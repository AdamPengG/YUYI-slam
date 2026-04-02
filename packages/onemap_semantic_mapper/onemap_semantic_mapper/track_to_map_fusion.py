from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .data_types import KeyframePacket, LocalCloudPacket, SensorConfig
from .proposal_association_3d import Proposal3D
from .persistent_instance_track_manager import TrackEvent


@dataclass
class TrackVoxelSubmap:
    voxel_keys: np.ndarray
    xyz_mean: np.ndarray
    hit_count: np.ndarray
    quality_sum: np.ndarray
    last_seen_stamp: np.ndarray
    rgb_mean: np.ndarray | None = None
    dirty_flag: bool = False
    _voxel_to_index: dict[tuple[int, int, int], int] = field(default_factory=dict, repr=False)

    @classmethod
    def empty(cls) -> "TrackVoxelSubmap":
        submap = cls(
            voxel_keys=np.zeros((0, 3), dtype=np.int32),
            xyz_mean=np.zeros((0, 3), dtype=np.float32),
            hit_count=np.zeros((0,), dtype=np.int32),
            quality_sum=np.zeros((0,), dtype=np.float32),
            last_seen_stamp=np.zeros((0,), dtype=np.float32),
            rgb_mean=None,
            dirty_flag=False,
        )
        submap._reindex()
        return submap

    def _reindex(self) -> None:
        self._voxel_to_index = {tuple(int(v) for v in key): idx for idx, key in enumerate(self.voxel_keys.tolist())}


class TrackToMapFusion:
    def __init__(self, *, voxel_size_m: float, output_dir: Path) -> None:
        self.voxel_size_m = float(max(voxel_size_m, 1e-3))
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.track_submaps_dir = self.output_dir / "track_submaps"
        self.track_submaps_dir.mkdir(parents=True, exist_ok=True)
        self._submaps: dict[str, TrackVoxelSubmap] = {}

    def load_from_metadata(self, payload: dict[str, Any]) -> None:
        raw_objects = payload.get("objects", payload) if isinstance(payload, dict) else {}
        if not isinstance(raw_objects, dict):
            return
        for track_id, raw in raw_objects.items():
            geometry_store_path = str(raw.get("geometry_store_path", "")).strip()
            if not geometry_store_path:
                continue
            geometry_path = Path(geometry_store_path)
            if not geometry_path.is_absolute():
                geometry_path = self.output_dir / geometry_path
            if not geometry_path.exists():
                continue
            self._submaps[str(track_id)] = self._load_submap(geometry_path)

    def fuse(
        self,
        *,
        keyframe: KeyframePacket,
        local_cloud: LocalCloudPacket,
        proposals: list[Proposal3D],
        track_events: list[TrackEvent],
    ) -> None:
        del local_cloud
        for event in track_events:
            if event.proposal_idx is None:
                continue
            if event.event_type not in {"new", "match", "reactivate", "fragment_attach"}:
                continue
            proposal = proposals[event.proposal_idx]
            self._fuse_track_points(
                track_id=event.track_id,
                xyz=proposal.support_xyz_world,
                stamp=float(keyframe.stamp_sec),
                quality=float(proposal.quality_mean),
            )

    def get_track_points(self, track_id: str) -> np.ndarray:
        submap = self._submaps.get(str(track_id))
        if submap is None or submap.xyz_mean.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return submap.xyz_mean.astype(np.float32, copy=False)

    def get_track_summary(self, track_id: str) -> dict[str, Any]:
        submap = self._submaps.get(str(track_id))
        if submap is None or submap.xyz_mean.size == 0:
            return {
                "geometry_store_path": f"track_submaps/{track_id}.npz",
                "fused_voxel_count": 0,
                "centroid_world": [0.0, 0.0, 0.0],
                "bbox_world": [0.0] * 6,
                "completeness_score": 0.0,
            }
        xyz = submap.xyz_mean.astype(np.float32, copy=False)
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
        completeness_score = min(1.0, float(xyz.shape[0]) / 250.0)
        return {
            "geometry_store_path": f"track_submaps/{track_id}.npz",
            "fused_voxel_count": int(xyz.shape[0]),
            "centroid_world": [float(v) for v in xyz.mean(axis=0).tolist()],
            "bbox_world": [float(v) for v in bbox_min.tolist() + bbox_max.tolist()],
            "completeness_score": float(completeness_score),
        }

    def save_dirty_submaps(self) -> None:
        for track_id, submap in self._submaps.items():
            if not submap.dirty_flag:
                continue
            path = self.track_submaps_dir / f"{track_id}.npz"
            np.savez_compressed(
                path,
                voxel_keys=submap.voxel_keys.astype(np.int32, copy=False),
                xyz_mean=submap.xyz_mean.astype(np.float32, copy=False),
                hit_count=submap.hit_count.astype(np.int32, copy=False),
                quality_sum=submap.quality_sum.astype(np.float32, copy=False),
                last_seen_stamp=submap.last_seen_stamp.astype(np.float32, copy=False),
            )
            submap.dirty_flag = False

    def surface_hit_ratio(self, track_id: str, xyz: np.ndarray) -> float:
        submap = self._submaps.get(str(track_id))
        if submap is None or submap.voxel_keys.size == 0 or xyz.size == 0:
            return 0.0
        proposal_voxels = self._voxel_keys(xyz)
        if proposal_voxels.size == 0:
            return 0.0
        proposal_keys = {tuple(int(v) for v in row) for row in proposal_voxels.tolist()}
        hit = sum(1 for key in proposal_keys if key in submap._voxel_to_index)
        return float(hit / max(len(proposal_keys), 1))

    def bbox3d_overlap(self, track_id: str, proposal_bbox_world: np.ndarray) -> float:
        submap = self._submaps.get(str(track_id))
        if submap is None or submap.xyz_mean.size == 0:
            return 0.0
        bbox_a = np.asarray(self.get_track_summary(track_id)["bbox_world"], dtype=np.float32)
        bbox_b = np.asarray(proposal_bbox_world, dtype=np.float32).reshape(-1)
        if bbox_a.size != 6 or bbox_b.size != 6:
            return 0.0
        inter_min = np.maximum(bbox_a[:3], bbox_b[:3])
        inter_max = np.minimum(bbox_a[3:], bbox_b[3:])
        inter_dim = np.maximum(inter_max - inter_min, 0.0)
        inter_vol = float(np.prod(inter_dim))
        if inter_vol <= 0.0:
            return 0.0
        vol_a = float(np.prod(np.maximum(bbox_a[3:] - bbox_a[:3], 0.0)))
        vol_b = float(np.prod(np.maximum(bbox_b[3:] - bbox_b[:3], 0.0)))
        denom = max(vol_a + vol_b - inter_vol, 1e-6)
        return float(inter_vol / denom)

    def reprojected_bbox_iou(
        self,
        *,
        track_id: str,
        proposal_bbox: list[int],
        keyframe: KeyframePacket,
        sensor_config: SensorConfig,
        dilate_px: int = 3,
    ) -> float:
        xyz = self.get_track_points(track_id)
        if xyz.shape[0] < 3:
            return 0.0
        c2w = np.asarray(keyframe.t_world_cam, dtype=np.float32).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        xyz_h = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)))
        xyz_cam = (w2c @ xyz_h.T).T[:, :3]
        z = xyz_cam[:, 2]
        valid = z > 0.05
        if not np.any(valid):
            return 0.0
        xyz_cam = xyz_cam[valid]
        fx = float(sensor_config.intrinsics["fx"])
        fy = float(sensor_config.intrinsics["fy"])
        cx = float(sensor_config.intrinsics["cx"])
        cy = float(sensor_config.intrinsics["cy"])
        width, height = int(sensor_config.image_size[0]), int(sensor_config.image_size[1])
        u = np.rint((xyz_cam[:, 0] * fx / xyz_cam[:, 2]) + cx).astype(np.int32)
        v = np.rint((xyz_cam[:, 1] * fy / xyz_cam[:, 2]) + cy).astype(np.int32)
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_image):
            return 0.0
        u = u[in_image]
        v = v[in_image]
        pred_mask = np.zeros((height, width), dtype=np.uint8)
        pred_mask[v, u] = 1
        if dilate_px > 0:
            kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
            pred_mask = cv2.dilate(pred_mask, kernel, iterations=1)
        ys, xs = np.nonzero(pred_mask)
        if xs.size == 0:
            return 0.0
        pred_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        return self._bbox_iou(pred_bbox, proposal_bbox)

    def _fuse_track_points(self, *, track_id: str, xyz: np.ndarray, stamp: float, quality: float) -> None:
        if xyz.size == 0:
            return
        submap = self._submaps.get(track_id)
        if submap is None:
            submap = TrackVoxelSubmap.empty()
            self._submaps[track_id] = submap
        voxel_keys = self._voxel_keys(xyz)
        if voxel_keys.size == 0:
            return
        unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
        voxel_xyz_mean = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        np.add.at(voxel_xyz_mean, inverse, xyz.astype(np.float64))
        counts = np.bincount(inverse).astype(np.int32)
        voxel_xyz_mean /= np.maximum(counts[:, None], 1)
        for idx, key_row in enumerate(unique_keys):
            key = tuple(int(v) for v in key_row.tolist())
            existing_idx = submap._voxel_to_index.get(key)
            if existing_idx is None:
                new_idx = submap.voxel_keys.shape[0]
                submap.voxel_keys = np.vstack((submap.voxel_keys, key_row.astype(np.int32)[None, :]))
                submap.xyz_mean = np.vstack((submap.xyz_mean, voxel_xyz_mean[idx].astype(np.float32)[None, :]))
                submap.hit_count = np.concatenate((submap.hit_count, np.asarray([counts[idx]], dtype=np.int32)))
                submap.quality_sum = np.concatenate(
                    (submap.quality_sum, np.asarray([float(quality) * float(counts[idx])], dtype=np.float32))
                )
                submap.last_seen_stamp = np.concatenate(
                    (submap.last_seen_stamp, np.asarray([float(stamp)], dtype=np.float32))
                )
                submap._voxel_to_index[key] = int(new_idx)
            else:
                prev_hits = float(submap.hit_count[existing_idx])
                new_hits = prev_hits + float(counts[idx])
                submap.xyz_mean[existing_idx] = (
                    (submap.xyz_mean[existing_idx].astype(np.float64) * prev_hits) + voxel_xyz_mean[idx] * counts[idx]
                ) / max(new_hits, 1.0)
                submap.hit_count[existing_idx] = int(new_hits)
                submap.quality_sum[existing_idx] = float(submap.quality_sum[existing_idx]) + float(quality) * float(
                    counts[idx]
                )
                submap.last_seen_stamp[existing_idx] = float(stamp)
        submap.dirty_flag = True

    def _load_submap(self, path: Path) -> TrackVoxelSubmap:
        with np.load(path, allow_pickle=False) as data:
            submap = TrackVoxelSubmap(
                voxel_keys=np.asarray(data.get("voxel_keys", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32),
                xyz_mean=np.asarray(data.get("xyz_mean", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32),
                hit_count=np.asarray(data.get("hit_count", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
                quality_sum=np.asarray(data.get("quality_sum", np.zeros((0,), dtype=np.float32)), dtype=np.float32),
                last_seen_stamp=np.asarray(
                    data.get("last_seen_stamp", np.zeros((0,), dtype=np.float32)), dtype=np.float32
                ),
                rgb_mean=None,
                dirty_flag=False,
            )
        submap._reindex()
        return submap

    def _voxel_keys(self, xyz: np.ndarray) -> np.ndarray:
        if xyz.size == 0:
            return np.zeros((0, 3), dtype=np.int32)
        return np.floor(np.asarray(xyz, dtype=np.float32) / self.voxel_size_m).astype(np.int32)

    @staticmethod
    def _bbox_iou(a: list[int], b: list[int]) -> float:
        ax0, ay0, ax1, ay1 = [int(v) for v in a]
        bx0, by0, bx1, by1 = [int(v) for v in b]
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter_w = max(0, inter_x1 - inter_x0 + 1)
        inter_h = max(0, inter_y1 - inter_y0 + 1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1)
        area_b = max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1)
        denom = max(area_a + area_b - inter, 1)
        return float(inter / denom)
