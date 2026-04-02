from __future__ import annotations

import argparse
import colorsys
import json
import os
import shutil
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .data_types import KeyframePacket, LocalCloudPacket
from .io.keyframe_manifest import load_keyframe_packets
from .io.local_cloud_manifest import load_local_cloud_packets
from .io.observation_manifest import append_observation_link
from .io.sensor_config import load_sensor_config
from .legacy_semantic_observer import MaskObservation, SemanticObserver
from .persistent_instance_track_manager import PersistentInstanceTrackManager
from .proposal_association_3d import ProposalAssociation3D
from .track_to_map_fusion import TrackToMapFusion
from .visibility_projector import VisibilityProjector
from .yoloe_helper_client import YOLOEHelperClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy 20260329-style online semantic observer with YOLO masks.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ovo-root", required=True)
    parser.add_argument("--ovo-config", default="data/working/configs/ovo_livo2_yoloe26x.yaml")
    parser.add_argument("--dataset-name", default="Replica")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--poll-period-sec", type=float, default=5.0)
    parser.add_argument("--min-keyframes", type=int, default=1)
    parser.add_argument("--process-every-new-keyframes", type=int, default=5)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk-labels", type=int, default=3)
    parser.add_argument("--visibility-depth-tolerance-m", type=float, default=0.08)
    parser.add_argument("--snapshot-voxel-size-m", type=float, default=0.0)
    parser.add_argument("--snapshot-max-points", type=int, default=1000000)
    parser.add_argument("--snapshot-mode", default="registered_frame")
    parser.add_argument("--min-observation-points", type=int, default=4)
    parser.add_argument("--min-mask-area", type=int, default=24)
    parser.add_argument("--merge-centroid-radius-m", type=float, default=0.75)
    parser.add_argument("--support-expansion-radius-m", type=float, default=0.20)
    parser.add_argument("--class-set", default="full")
    parser.add_argument("--near-ground-filter-height-m", type=float, default=0.08)
    parser.add_argument("--near-ground-floor-percentile", type=float, default=1.0)
    parser.add_argument("--assoc-score-min", type=float, default=0.42)
    parser.add_argument("--reproj-iou-min", type=float, default=0.08)
    parser.add_argument("--surface-hit-min", type=float, default=0.10)
    parser.add_argument("--reproj-dilate-px", type=int, default=3)
    parser.add_argument("--track-pending-hits", type=int, default=2)
    parser.add_argument("--track-dormant-after-sec", type=float, default=2.0)
    parser.add_argument("--track-delete-after-sec", type=float, default=30.0)
    parser.add_argument("--new-track-min-points", type=int, default=6)
    parser.add_argument("--fuse-voxel-size-m", type=float, default=0.03)
    parser.add_argument("--support-expansion-max-points", type=int, default=12000)
    parser.add_argument("--use-semantic-subset-projection", action="store_true")
    return parser.parse_args()


class LegacyYOLOMaskProposal:
    def __init__(self, semantic_cfg: dict[str, Any], class_names: list[str], device: str, topk_labels: int) -> None:
        self.semantic_cfg = semantic_cfg
        self.class_names = list(class_names)
        self.device = device
        self.topk_labels = max(int(topk_labels), 1)
        yoloe_cfg = dict(semantic_cfg.get("yoloe", {}))
        self.repo_root = Path(
            str(yoloe_cfg.get("repo_root", "/home/peng/isacc_slam/reference/YOLOE_official"))
        ).expanduser()
        self.helper_script = Path(
            str(
                yoloe_cfg.get(
                    "helper_script",
                    "/home/peng/isacc_slam/src/onemap_semantic_mapper/scripts/run_yoloe26x_region_infer.py",
                )
            )
        ).expanduser()
        self.conda_env = str(yoloe_cfg.get("conda_env", "yoloe_env")).strip()
        self.python_bin = str(yoloe_cfg.get("python_bin", "")).strip()
        self.cuda_visible_devices = str(yoloe_cfg.get("cuda_visible_devices", "1")).strip()
        self.model_path = str(yoloe_cfg.get("model_path", "yoloe-26x-seg.pt")).strip()
        self.conf_thresh = float(yoloe_cfg.get("conf_thresh", 0.10))
        self.iou_thresh = float(yoloe_cfg.get("iou_thresh", 0.50))
        self.max_det = int(yoloe_cfg.get("max_det", 100))
        self._client = YOLOEHelperClient(
            helper_script=self.helper_script,
            repo_root=self.repo_root,
            conda_env=self.conda_env,
            python_bin=self.python_bin,
            cuda_visible_devices=self.cuda_visible_devices,
            model_path=self.model_path,
            device=self.device,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            max_det=self.max_det,
            topk_labels=self.topk_labels,
        )

    def close(self) -> None:
        self._client.close()

    def get_masks(self, image_rgb: np.ndarray, frame_id: int) -> list[dict[str, Any]]:
        del frame_id
        results, masks = self._client.infer(image_rgb, self.class_names, include_masks=True)
        if masks is None or masks.shape[0] == 0 or not isinstance(results, list):
            return []
        count = min(len(results), masks.shape[0])
        observations: list[dict[str, Any]] = []
        for idx in range(count):
            rec = dict(results[idx])
            rec["binary_mask"] = masks[idx]
            observations.append(rec)
        return observations


class OnlineSemanticAdapter:
    def __init__(
        self,
        semantic_cfg: dict[str, Any],
        class_names: list[str],
        scene_name: str,
        device: str,
        topk_labels: int,
    ) -> None:
        self.device = device
        self.class_names = class_names
        self.topk_labels = max(int(topk_labels), 1)
        frontend = str(semantic_cfg.get("frontend", "textregion_pe")).strip().lower()
        if frontend == "yoloe26x":
            self.mask_generator = LegacyYOLOMaskProposal(semantic_cfg, class_names, device=device, topk_labels=topk_labels)
        else:
            raise RuntimeError(f"legacy_yolo observer only supports frontend=yoloe26x, got {frontend}")

    def close(self) -> None:
        close_fn = getattr(self.mask_generator, "close", None)
        if callable(close_fn):
            close_fn()

    def build_mask_observations(self, image_rgb: np.ndarray, frame_id: int) -> list[MaskObservation]:
        raw_masks = self.mask_generator.get_masks(image_rgb, frame_id)
        if not raw_masks:
            return []
        observations: list[MaskObservation] = []
        for raw_mask in raw_masks:
            binary_mask = np.asarray(raw_mask.get("binary_mask"), dtype=bool)
            if binary_mask.ndim != 2 or not bool(binary_mask.any()):
                continue
            ys, xs = np.nonzero(binary_mask)
            bbox_xyxy = raw_mask.get("bbox_xyxy")
            if bbox_xyxy is None or len(bbox_xyxy) != 4:
                bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            label_candidates = [str(v) for v in raw_mask.get("semantic_label_candidates", []) if str(v).strip()]
            label_scores = [float(v) for v in raw_mask.get("semantic_scores", [])]
            if not label_candidates:
                continue
            if not label_scores:
                label_scores = [1.0 / max(len(label_candidates), 1)] * len(label_candidates)
            observations.append(
                MaskObservation(
                    mask_id=int(raw_mask.get("mask_id", len(observations))),
                    binary_mask=binary_mask,
                    bbox_xyxy=[int(v) for v in bbox_xyxy],
                    semantic_label_candidates=label_candidates[: self.topk_labels],
                    semantic_scores=label_scores[: self.topk_labels],
                )
            )
        return observations


class OnlineSemanticObserverRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.scene_dir = Path(args.scene_dir).expanduser().resolve()
        self.export_dir = Path(args.export_dir).expanduser().resolve()
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.ovo_root = Path(args.ovo_root).expanduser().resolve()
        self.scene_name = args.scene_name or self.scene_dir.name
        os.chdir(self.ovo_root)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "observer_state_legacy_yolo.json"
        self.status_path = self.output_dir / "online_status.json"
        self.object_memory_path = self.output_dir / "object_memory.json"
        self.snapshot_path = self.output_dir / "semantic_snapshot.npz"
        self.snapshot_summary_path = self.output_dir / "semantic_snapshot_summary.txt"
        self.track_events_path = self.output_dir / "track_events.jsonl"
        self.debug_image_path = self.output_dir / "semantic_debug_latest.png"
        self.observation_manifest_path = self.export_dir / "observation_links.jsonl"

        self.processed_keyframe_ids: set[int] = set()
        self.processed_semantic_keyframe_ids: set[int] = set()
        self.total_observations = 0
        self.last_snapshot_processed = 0
        self.latest_registered_snapshot: dict[str, Any] | None = None
        self.cloud_cache: dict[str, np.ndarray] = {}
        self.local_cloud_lookup: dict[str, LocalCloudPacket] = {}
        self.support_expansion_radius_m = max(float(args.support_expansion_radius_m), 0.0)
        self.last_track_events_last_pass = 0
        self.last_recent_reactivations = 0

        if args.clear_output and not args.resume_if_exists:
            self._clear_runtime_outputs()

        if args.resume_if_exists:
            self._load_state()

        print("[legacy_yolo] loading semantic config", flush=True)
        semantic_cfg = self._load_ovo_config()
        print("[legacy_yolo] loading class list", flush=True)
        self.class_names = self._load_classes()
        print("[legacy_yolo] loading sensor config", flush=True)
        self.sensor_config = load_sensor_config(self.scene_dir / "sensor_config.yaml")
        print("[legacy_yolo] building projector/observer", flush=True)
        self.projector = VisibilityProjector(
            depth_tolerance_m=args.visibility_depth_tolerance_m,
            prefer_semantic_subset=args.use_semantic_subset_projection,
        )
        self.observer = SemanticObserver(
            min_mask_area=args.min_mask_area,
            min_hit_points=args.min_observation_points,
        )
        print("[legacy_yolo] loading persistent track metadata", flush=True)
        self.track_manager = self._load_track_manager()
        self.track_fuser = TrackToMapFusion(
            voxel_size_m=args.fuse_voxel_size_m,
            output_dir=self.output_dir,
        )
        self.track_fuser.load_from_metadata({"objects": self.track_manager.to_metadata_dict()})
        for track_id in list(self.track_manager.tracks.keys()):
            self.track_manager.apply_fusion_summary(track_id, self.track_fuser.get_track_summary(track_id))
        print("[legacy_yolo] building semantic adapter", flush=True)
        self.semantic_adapter = OnlineSemanticAdapter(
            semantic_cfg=semantic_cfg,
            class_names=self.class_names,
            scene_name=self.scene_name,
            device=args.device,
            topk_labels=args.topk_labels,
        )
        print("[legacy_yolo] building proposal association", flush=True)
        self.proposal_association = ProposalAssociation3D(
            assoc_score_min=args.assoc_score_min,
            reproj_iou_min=args.reproj_iou_min,
            surface_hit_min=args.surface_hit_min,
            reproj_dilate_px=args.reproj_dilate_px,
            use_appearance=True,
            use_depth_dense_fallback=False,
            support_expansion_radius_m=args.support_expansion_radius_m,
            support_expansion_max_points=args.support_expansion_max_points,
        )
        print("[legacy_yolo] semantic adapter ready", flush=True)

    def _clear_runtime_outputs(self) -> None:
        for path in [
            self.state_path,
            self.status_path,
            self.object_memory_path,
            self.snapshot_path,
            self.snapshot_summary_path,
            self.track_events_path,
            self.debug_image_path,
            self.observation_manifest_path,
        ]:
            if path.exists():
                path.unlink()
        track_submaps_dir = self.output_dir / "track_submaps"
        if track_submaps_dir.exists():
            shutil.rmtree(track_submaps_dir)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.processed_keyframe_ids = {int(v) for v in payload.get("processed_keyframe_ids", [])}
        self.processed_semantic_keyframe_ids = {int(v) for v in payload.get("processed_semantic_keyframe_ids", [])}
        self.total_observations = int(payload.get("total_observations", 0))
        self.last_snapshot_processed = int(payload.get("last_snapshot_processed", 0))

    def _save_state(self) -> None:
        payload = {
            "processed_keyframe_ids": sorted(self.processed_keyframe_ids),
            "processed_semantic_keyframe_ids": sorted(self.processed_semantic_keyframe_ids),
            "total_observations": int(self.total_observations),
            "last_snapshot_processed": int(self.last_snapshot_processed),
        }
        self.state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _load_track_manager(self) -> PersistentInstanceTrackManager:
        if self.args.resume_if_exists and self.object_memory_path.exists():
            payload = json.loads(self.object_memory_path.read_text(encoding="utf-8"))
            return PersistentInstanceTrackManager.from_metadata_dict(payload)
        return PersistentInstanceTrackManager(
            pending_hits=self.args.track_pending_hits,
            dormant_after_sec=self.args.track_dormant_after_sec,
            delete_after_sec=self.args.track_delete_after_sec,
            new_track_min_points=self.args.new_track_min_points,
        )

    def _save_object_memory(self) -> None:
        stats = self._object_state_stats()
        self.object_memory_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "objects": self.track_manager.to_metadata_dict(),
                    "stats": stats,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _load_ovo_config(self) -> dict[str, Any]:
        from ovo.utils import io_utils

        ovo_config_path = Path(self.args.ovo_config)
        if not ovo_config_path.is_absolute():
            ovo_config_path = self.ovo_root / ovo_config_path
        config = io_utils.load_config(str(ovo_config_path))
        return dict(config["semantic"])

    def _load_classes(self) -> list[str]:
        eval_info_path = self.ovo_root / "data" / "working" / "configs" / self.args.dataset_name / "eval_info.yaml"
        with eval_info_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        basic_indoor_allowlist = {
            "sofa",
            "chair",
            "desk",
            "table",
            "cabinet",
            "base-cabinet",
            "shelf",
            "bench",
            "stool",
            "bed",
            "nightstand",
            "wardrobe",
            "bin",
        }
        factory_shelf_box_allowlist = {
            "rack",
            "shelf",
            "box",
            "bin",
        }
        box_only_allowlist = {
            "box",
        }
        if self.args.class_set == "reduced":
            classes = list(payload.get("class_names_reduced", payload["class_names"]))
        elif self.args.class_set in {"basic_indoor", "basic_furniture", "basic_household"}:
            all_classes = [str(name) for name in payload["class_names"]]
            classes = [name for name in all_classes if name in basic_indoor_allowlist]
        elif self.args.class_set in {"factory_shelf_box", "rack_box", "shelf_box"}:
            all_classes = [str(name) for name in payload["class_names"]]
            classes = [name for name in all_classes if name in factory_shelf_box_allowlist]
        elif self.args.class_set in {"box_only", "box"}:
            all_classes = [str(name) for name in payload["class_names"]]
            classes = [name for name in all_classes if name in box_only_allowlist]
        else:
            classes = [str(name) for name in payload["class_names"]]
            classes = [name for name in classes if name and name != "0"]
        suppress = {
            "wall",
            "floor",
            "ceiling",
            "door",
            "window",
            "blinds",
            "clock",
            "lamp",
            "tv-screen",
            "wall-cabinet",
            "mirror",
            "poster",
            "picture",
            "picture-frame",
        }
        return [name for name in classes if str(name).strip().lower() not in suppress]

    def run(self) -> None:
        try:
            while True:
                processed = self.process_once()
                if self.args.run_once:
                    return
                if not processed:
                    time.sleep(self.args.poll_period_sec)
        finally:
            self.semantic_adapter.close()

    def process_once(self) -> bool:
        keyframes = load_keyframe_packets(self.export_dir / "keyframe_packets.jsonl")
        local_clouds = {
            packet.local_cloud_id: packet
            for packet in load_local_cloud_packets(self.export_dir / "local_cloud_packets.jsonl")
        }
        self.local_cloud_lookup = local_clouds

        total_keyframes = len(keyframes)
        if total_keyframes < self.args.min_keyframes:
            self._write_status(total_keyframes, "waiting_for_keyframes", 0)
            return False

        new_packets = [packet for packet in keyframes if packet.keyframe_id not in self.processed_keyframe_ids]
        if not new_packets:
            self._write_status(total_keyframes, "idle", 0)
            return False

        processed_now = 0
        skipped_now = 0
        semantic_stride = max(int(self.args.process_every_new_keyframes), 1)
        for packet in new_packets:
            if semantic_stride > 1 and (int(packet.keyframe_id) % semantic_stride) != 0:
                self.processed_keyframe_ids.add(packet.keyframe_id)
                skipped_now += 1
                continue
            print(
                f"[legacy_yolo] processing keyframe={packet.keyframe_id} local_cloud={packet.local_cloud_ref}",
                flush=True,
            )
            local_cloud = local_clouds.get(packet.local_cloud_ref)
            if local_cloud is None:
                continue
            if not Path(packet.rgb_path).exists():
                continue
            if packet.depth_path is not None and not Path(packet.depth_path).exists():
                continue

            self._process_keyframe(packet, local_cloud)
            self.processed_keyframe_ids.add(packet.keyframe_id)
            self.processed_semantic_keyframe_ids.add(packet.keyframe_id)
            processed_now += 1

        if processed_now == 0 and skipped_now == 0:
            self._write_status(total_keyframes, "waiting_for_local_clouds", 0)
            return False

        self.track_fuser.save_dirty_submaps()
        self._save_object_memory()
        self._save_state()

        processed_total = len(self.processed_semantic_keyframe_ids)
        force_initial_snapshot = processed_total > 0 and not self.snapshot_path.exists()
        if force_initial_snapshot or (processed_total - self.last_snapshot_processed >= self.args.process_every_new_keyframes):
            self._export_snapshot()
            self.last_snapshot_processed = processed_total
            self._save_state()

        self._write_status(total_keyframes, "updated", processed_now)
        return True

    def _process_keyframe(self, packet: KeyframePacket, local_cloud: LocalCloudPacket) -> None:
        image_rgb = self._load_image(packet.rgb_path)
        depth_m = self._load_depth(packet.depth_path)
        projection = self.projector.project(packet, local_cloud, self.sensor_config, depth_m)
        if projection.visible_point_indices.size == 0:
            print(f"[legacy_yolo] keyframe={packet.keyframe_id} no visible points", flush=True)
            return

        print(
            f"[legacy_yolo] keyframe={packet.keyframe_id} visible_points={projection.visible_point_indices.size} -> yolo",
            flush=True,
        )
        mask_observations = self.semantic_adapter.build_mask_observations(image_rgb, packet.keyframe_id)
        if not mask_observations:
            self._write_debug_image(image_rgb, [], {})
            print(f"[legacy_yolo] keyframe={packet.keyframe_id} no mask observations", flush=True)
            return

        print(
            f"[legacy_yolo] keyframe={packet.keyframe_id} masks={len(mask_observations)} -> observer",
            flush=True,
        )
        observations = self.observer.observe(
            keyframe_id=packet.keyframe_id,
            local_cloud_id=local_cloud.local_cloud_id,
            projection=projection,
            masks=mask_observations,
            existing_point_object_ids=None,
        )
        if not observations:
            self._write_debug_image(image_rgb, mask_observations, {})
            print(f"[legacy_yolo] keyframe={packet.keyframe_id} observer produced no observations", flush=True)
            return

        proposals = self.proposal_association.build_proposals(
            keyframe=packet,
            local_cloud=local_cloud,
            projection=projection,
            mask_observations=mask_observations,
            observations=observations,
            image_rgb=image_rgb,
            depth_m=depth_m,
        )
        if not proposals:
            self._write_debug_image(image_rgb, mask_observations, {})
            print(f"[legacy_yolo] keyframe={packet.keyframe_id} proposal builder produced no proposals", flush=True)
            return

        association = self.proposal_association.associate(
            keyframe=packet,
            sensor_config=self.sensor_config,
            depth_m=depth_m,
            proposals=proposals,
            track_manager=self.track_manager,
            track_fuser=self.track_fuser,
        )
        track_events = self.track_manager.update(
            keyframe=packet,
            proposals=proposals,
            association=association,
        )
        self.track_fuser.fuse(
            keyframe=packet,
            local_cloud=local_cloud,
            proposals=proposals,
            track_events=track_events,
        )
        for event in track_events:
            if event.proposal_idx is None:
                continue
            summary = self.track_fuser.get_track_summary(event.track_id)
            self.track_manager.apply_fusion_summary(event.track_id, summary)
        self._append_track_events(track_events)
        self.last_track_events_last_pass = len(track_events)
        self.last_recent_reactivations = sum(1 for event in track_events if event.event_type == "reactivate")

        proposal_track_map = {
            int(event.proposal_idx): str(event.track_id)
            for event in track_events
            if event.proposal_idx is not None
        }
        observation_track_map = {
            int(proposals[proposal_index].observation_index): track_id
            for proposal_index, track_id in proposal_track_map.items()
            if 0 <= int(proposal_index) < len(proposals)
        }
        self._write_debug_image(image_rgb, mask_observations, observation_track_map)
        for proposal_index, proposal in enumerate(proposals):
            observation = observations[proposal.observation_index]
            observation.candidate_object_id = proposal_track_map.get(proposal_index)
            append_observation_link(self.observation_manifest_path, observation)
            self.total_observations += 1
        self._update_latest_registered_snapshot(
            packet=packet,
            proposals=proposals,
            proposal_track_map=proposal_track_map,
        )
        print(
            "[legacy_yolo] "
            f"keyframe={packet.keyframe_id} proposals={len(proposals)} "
            f"tracks={len(self.track_manager.live_tracks())} "
            f"matches={len(association.matches)} new={sum(1 for event in track_events if event.event_type == 'new')}",
            flush=True,
        )

    def _update_latest_registered_snapshot(
        self,
        *,
        packet: KeyframePacket,
        proposals: list[Proposal3D],
        proposal_track_map: dict[int, str],
    ) -> None:
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        instance_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        instance_centers: list[np.ndarray] = []
        instance_labels: list[str] = []
        instance_ids: list[int] = []
        instance_global_ids: list[int] = []
        instance_states: list[str] = []
        instance_scores: list[float] = []
        instance_last_seen_stamp: list[float] = []
        instance_point_counts: list[int] = []
        instance_view_counts: list[int] = []

        class_lookup = {name: idx for idx, name in enumerate(self.class_names)}
        floor_z: float | None = None
        ground_keep_z: float | None = None
        total_ground_filtered = 0

        if self.args.near_ground_filter_height_m > 0.0:
            z_parts = [proposal.support_xyz_world[:, 2] for proposal in proposals if proposal.support_xyz_world.size > 0]
            if z_parts:
                all_z = np.concatenate(z_parts).astype(np.float32, copy=False)
                percentile = float(np.clip(self.args.near_ground_floor_percentile, 0.0, 100.0))
                floor_z = float(np.percentile(all_z, percentile))
                ground_keep_z = float(floor_z + self.args.near_ground_filter_height_m)

        for proposal_index, proposal in enumerate(proposals):
            track_id = proposal_track_map.get(int(proposal_index))
            if not track_id:
                continue
            track = self.track_manager.tracks.get(track_id)
            if track is None:
                continue

            xyz = np.asarray(proposal.support_xyz_world, dtype=np.float32)
            if xyz.size == 0:
                continue
            xyz = self._voxel_downsample(xyz, self.args.snapshot_voxel_size_m)
            if xyz.size == 0:
                continue
            if ground_keep_z is not None:
                keep = xyz[:, 2] >= ground_keep_z
                total_ground_filtered += int(xyz.shape[0] - int(np.count_nonzero(keep)))
                xyz = xyz[keep]
                if xyz.size == 0:
                    continue

            label = self._track_label(track)
            class_id = int(class_lookup.get(label, -1))
            semantic_rgb = self._semantic_color(class_id)

            xyz_parts.append(xyz)
            rgb_parts.append(np.tile(semantic_rgb[None, :], (xyz.shape[0], 1)))
            instance_parts.append(np.full((xyz.shape[0],), int(track.track_num_id), dtype=np.int32))
            class_parts.append(np.full((xyz.shape[0],), class_id, dtype=np.int32))
            instance_centers.append(xyz.mean(axis=0).astype(np.float32))
            instance_labels.append(label)
            instance_ids.append(int(track.track_num_id))
            instance_global_ids.append(int(track.track_num_id))
            instance_states.append(str(track.state))
            instance_scores.append(float(track.stability_score))
            instance_last_seen_stamp.append(float(track.last_seen_stamp))
            instance_point_counts.append(int(xyz.shape[0]))
            instance_view_counts.append(int(track.observation_count))

        if not xyz_parts:
            self.latest_registered_snapshot = None
            return

        xyz = np.vstack(xyz_parts).astype(np.float32, copy=False)
        rgb = np.vstack(rgb_parts).astype(np.uint8, copy=False)
        instance_id = np.concatenate(instance_parts)
        class_id = np.concatenate(class_parts)

        if xyz.shape[0] > self.args.snapshot_max_points:
            step = int(np.ceil(xyz.shape[0] / self.args.snapshot_max_points))
            keep = np.arange(0, xyz.shape[0], step, dtype=np.int64)
            xyz = xyz[keep]
            rgb = rgb[keep]
            instance_id = instance_id[keep]
            class_id = class_id[keep]

        self.latest_registered_snapshot = {
            "xyz": xyz,
            "rgb": rgb,
            "instance_id": instance_id,
            "class_id": class_id,
            "instance_ids": np.asarray(instance_ids, dtype=np.int32),
            "instance_centers": np.asarray(instance_centers, dtype=np.float32),
            "instance_labels": np.asarray(instance_labels, dtype=np.str_),
            "instance_global_ids": np.asarray(instance_global_ids, dtype=np.int32),
            "instance_states": np.asarray(instance_states, dtype=np.str_),
            "instance_scores": np.asarray(instance_scores, dtype=np.float32),
            "instance_last_seen_stamp": np.asarray(instance_last_seen_stamp, dtype=np.float32),
            "instance_point_counts": np.asarray(instance_point_counts, dtype=np.int32),
            "instance_view_counts": np.asarray(instance_view_counts, dtype=np.int32),
            "class_names": np.asarray(self.class_names, dtype=np.str_),
            "mode": np.asarray("semantic_registered_frame", dtype=np.str_),
            "ground_floor_z": float(floor_z) if floor_z is not None else None,
            "ground_keep_z": float(ground_keep_z) if ground_keep_z is not None else None,
            "ground_filtered_points": int(total_ground_filtered),
            "keyframe_id": int(packet.keyframe_id),
        }

    def _write_debug_image(
        self,
        image_rgb: np.ndarray,
        mask_observations: list[MaskObservation],
        observation_track_map: dict[int, str],
    ) -> None:
        canvas = image_rgb.copy()
        if canvas.ndim != 3 or canvas.shape[2] != 3:
            return
        for obs_index, obs in enumerate(mask_observations):
            mask = np.asarray(obs.binary_mask, dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            hue = (obs_index * 0.173) % 1.0
            color_rgb = np.asarray(colorsys.hsv_to_rgb(hue, 0.85, 1.0), dtype=np.float32)
            color = (color_rgb * 255.0).astype(np.uint8)
            alpha = 0.35
            canvas[mask] = np.clip((1.0 - alpha) * canvas[mask] + alpha * color[None, :], 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in color.tolist()), 2)
            label = obs.semantic_label_candidates[0] if obs.semantic_label_candidates else "unknown"
            track_id = observation_track_map.get(obs_index)
            if track_id:
                label = f"{label} {track_id}"
            x1, y1, x2, y2 = [int(v) for v in obs.bbox_xyxy]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), tuple(int(v) for v in color.tolist()), 2)
            cv2.putText(
                canvas,
                label[:48],
                (max(0, x1), max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                tuple(int(v) for v in color.tolist()),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(self.debug_image_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    def _load_image(self, image_path: str) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to load image {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _load_depth(self, depth_path: str | None) -> np.ndarray | None:
        if depth_path is None:
            return None
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"Failed to load depth {depth_path}")
        depth_scale = float(self.sensor_config.intrinsics.get("depth_scale", 1000.0))
        return depth_raw.astype(np.float32) / max(depth_scale, 1e-6)

    def _load_cloud_xyz(self, local_cloud: LocalCloudPacket) -> np.ndarray:
        cached = self.cloud_cache.get(local_cloud.local_cloud_id)
        if cached is not None:
            return cached
        with closing(np.load(Path(local_cloud.cloud_path))) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float32)
        self.cloud_cache[local_cloud.local_cloud_id] = xyz
        return xyz

    def _export_snapshot(self) -> None:
        if str(self.args.snapshot_mode).strip().lower() in {"registered_frame", "current_frame", "frame"}:
            if self._export_registered_frame_snapshot():
                return

        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        instance_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        instance_centers: list[np.ndarray] = []
        instance_labels: list[str] = []
        instance_ids: list[int] = []
        instance_global_ids: list[int] = []
        instance_states: list[str] = []
        instance_scores: list[float] = []
        instance_last_seen_stamp: list[float] = []
        instance_point_counts: list[int] = []
        instance_view_counts: list[int] = []

        class_lookup = {name: idx for idx, name in enumerate(self.class_names)}

        export_tracks = sorted(self.track_manager.live_tracks(), key=lambda track: track.track_num_id)
        floor_z: float | None = None
        ground_keep_z: float | None = None
        total_ground_filtered = 0
        if self.args.near_ground_filter_height_m > 0.0 and export_tracks:
            z_parts: list[np.ndarray] = []
            for track in export_tracks:
                xyz = self.track_fuser.get_track_points(track.track_id)
                if xyz.size == 0:
                    continue
                z_parts.append(xyz[:, 2].astype(np.float32, copy=False))
            if z_parts:
                all_z = np.concatenate(z_parts)
                percentile = float(np.clip(self.args.near_ground_floor_percentile, 0.0, 100.0))
                floor_z = float(np.percentile(all_z, percentile))
                ground_keep_z = float(floor_z + self.args.near_ground_filter_height_m)

        for object_index, track in enumerate(export_tracks):
            xyz = self.track_fuser.get_track_points(track.track_id)
            if xyz.size == 0:
                continue
            xyz = self._voxel_downsample(xyz, self.args.snapshot_voxel_size_m)
            if xyz.size == 0:
                continue
            if ground_keep_z is not None:
                keep = xyz[:, 2] >= ground_keep_z
                total_ground_filtered += int(xyz.shape[0] - int(np.count_nonzero(keep)))
                xyz = xyz[keep]
                if xyz.size == 0:
                    continue

            label = self._track_label(track)
            class_id = int(class_lookup.get(label, -1))
            semantic_rgb = self._semantic_color(class_id)

            xyz_parts.append(xyz)
            rgb_parts.append(np.tile(semantic_rgb[None, :], (xyz.shape[0], 1)))
            instance_parts.append(np.full((xyz.shape[0],), object_index, dtype=np.int32))
            class_parts.append(np.full((xyz.shape[0],), class_id, dtype=np.int32))
            instance_centers.append(xyz.mean(axis=0).astype(np.float32))
            instance_labels.append(label)
            instance_ids.append(object_index)
            instance_global_ids.append(int(track.track_num_id))
            instance_states.append(str(track.state))
            instance_scores.append(float(track.stability_score))
            instance_last_seen_stamp.append(float(track.last_seen_stamp))
            instance_point_counts.append(int(xyz.shape[0]))
            instance_view_counts.append(int(track.observation_count))

        if not xyz_parts:
            return

        xyz = np.vstack(xyz_parts).astype(np.float32, copy=False)
        rgb = np.vstack(rgb_parts).astype(np.uint8, copy=False)
        instance_id = np.concatenate(instance_parts)
        class_id = np.concatenate(class_parts)

        if xyz.shape[0] > self.args.snapshot_max_points:
            step = int(np.ceil(xyz.shape[0] / self.args.snapshot_max_points))
            keep = np.arange(0, xyz.shape[0], step, dtype=np.int64)
            xyz = xyz[keep]
            rgb = rgb[keep]
            instance_id = instance_id[keep]
            class_id = class_id[keep]

        tmp_path = self.snapshot_path.with_name(self.snapshot_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            xyz=xyz,
            rgb=rgb,
            instance_id=instance_id,
            class_id=class_id,
            instance_ids=np.asarray(instance_ids, dtype=np.int32),
            instance_centers=np.asarray(instance_centers, dtype=np.float32),
            instance_labels=np.asarray(instance_labels, dtype=np.str_),
            instance_global_ids=np.asarray(instance_global_ids, dtype=np.int32),
            instance_states=np.asarray(instance_states, dtype=np.str_),
            instance_scores=np.asarray(instance_scores, dtype=np.float32),
            instance_last_seen_stamp=np.asarray(instance_last_seen_stamp, dtype=np.float32),
            instance_point_counts=np.asarray(instance_point_counts, dtype=np.int32),
            instance_view_counts=np.asarray(instance_view_counts, dtype=np.int32),
            class_names=np.asarray(self.class_names, dtype=np.str_),
            classify_error=np.asarray("", dtype=np.str_),
            mode=np.asarray("semantic", dtype=np.str_),
        )
        tmp_path.replace(self.snapshot_path)

        lines = [
            f"num_points={xyz.shape[0]}",
            f"num_instances={len(instance_ids)}",
            f"processed_keyframes={len(self.processed_keyframe_ids)}",
            f"total_observations={self.total_observations}",
        ]
        if floor_z is not None and ground_keep_z is not None:
            lines.extend(
                [
                    f"ground_floor_z={floor_z:.6f}",
                    f"ground_keep_z={ground_keep_z:.6f}",
                    f"ground_filtered_points={total_ground_filtered}",
                ]
            )
        for label, center, point_count, view_count in zip(
            instance_labels[:50],
            instance_centers[:50],
            instance_point_counts[:50],
            instance_view_counts[:50],
            strict=False,
        ):
            lines.append(
                f"label={label}, points={point_count}, views={view_count}, "
                f"center={[round(float(v), 4) for v in center.tolist()]}"
            )
        self.snapshot_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"[legacy_yolo] snapshot exported points={xyz.shape[0]} instances={len(instance_ids)} path={self.snapshot_path}",
            flush=True,
        )

    def _export_registered_frame_snapshot(self) -> bool:
        payload = self.latest_registered_snapshot
        if not payload:
            return False

        xyz = np.asarray(payload["xyz"], dtype=np.float32, copy=False)
        rgb = np.asarray(payload["rgb"], dtype=np.uint8, copy=False)
        instance_id = np.asarray(payload["instance_id"], dtype=np.int32, copy=False)
        class_id = np.asarray(payload["class_id"], dtype=np.int32, copy=False)
        instance_ids = np.asarray(payload["instance_ids"], dtype=np.int32, copy=False)
        instance_centers = np.asarray(payload["instance_centers"], dtype=np.float32, copy=False)
        instance_labels = np.asarray(payload["instance_labels"], dtype=np.str_)
        instance_global_ids = np.asarray(payload["instance_global_ids"], dtype=np.int32, copy=False)
        instance_states = np.asarray(payload["instance_states"], dtype=np.str_)
        instance_scores = np.asarray(payload["instance_scores"], dtype=np.float32, copy=False)
        instance_last_seen_stamp = np.asarray(payload["instance_last_seen_stamp"], dtype=np.float32, copy=False)
        instance_point_counts = np.asarray(payload["instance_point_counts"], dtype=np.int32, copy=False)
        instance_view_counts = np.asarray(payload["instance_view_counts"], dtype=np.int32, copy=False)
        class_names = np.asarray(payload["class_names"], dtype=np.str_)
        keyframe_id = int(payload.get("keyframe_id", -1))

        tmp_path = self.snapshot_path.with_name(self.snapshot_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            xyz=xyz,
            rgb=rgb,
            instance_id=instance_id,
            class_id=class_id,
            instance_ids=instance_ids,
            instance_centers=instance_centers,
            instance_labels=instance_labels,
            instance_global_ids=instance_global_ids,
            instance_states=instance_states,
            instance_scores=instance_scores,
            instance_last_seen_stamp=instance_last_seen_stamp,
            instance_point_counts=instance_point_counts,
            instance_view_counts=instance_view_counts,
            class_names=class_names,
            classify_error=np.asarray("", dtype=np.str_),
            mode=np.asarray("semantic_registered_frame", dtype=np.str_),
        )
        tmp_path.replace(self.snapshot_path)

        lines = [
            f"snapshot_mode=registered_frame",
            f"keyframe_id={keyframe_id}",
            f"num_points={xyz.shape[0]}",
            f"num_instances={len(instance_ids)}",
            f"processed_keyframes={len(self.processed_keyframe_ids)}",
            f"total_observations={self.total_observations}",
        ]
        if payload.get("ground_floor_z") is not None and payload.get("ground_keep_z") is not None:
            lines.extend(
                [
                    f"ground_floor_z={float(payload['ground_floor_z']):.6f}",
                    f"ground_keep_z={float(payload['ground_keep_z']):.6f}",
                    f"ground_filtered_points={int(payload.get('ground_filtered_points', 0))}",
                ]
            )
        for label, center, point_count, view_count in zip(
            instance_labels[:50],
            instance_centers[:50],
            instance_point_counts[:50],
            instance_view_counts[:50],
            strict=False,
        ):
            lines.append(
                f"label={label}, points={point_count}, views={view_count}, "
                f"center={[round(float(v), 4) for v in center.tolist()]}"
            )
        self.snapshot_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"[legacy_yolo] snapshot exported mode=registered_frame points={xyz.shape[0]} "
            f"instances={len(instance_ids)} keyframe={keyframe_id} path={self.snapshot_path}",
            flush=True,
        )
        return True

    def _voxel_downsample(self, xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
        if voxel_size_m <= 0.0 or xyz.shape[0] <= 1:
            return xyz
        voxel = np.floor(xyz / voxel_size_m).astype(np.int64)
        _, inverse = np.unique(voxel, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inverse, xyz.astype(np.float64))
        downsampled = sums / np.maximum(counts[:, None], 1)
        return downsampled.astype(np.float32)

    def _track_label(self, track) -> str:
        if getattr(track, "top_label", ""):
            return str(track.top_label)
        if not track.label_votes:
            return "unknown"
        return max(track.label_votes.items(), key=lambda item: item[1])[0]

    def _semantic_color(self, class_id: int) -> np.ndarray:
        if class_id < 0:
            return np.array([180, 180, 180], dtype=np.uint8)
        hue = (class_id * 0.137) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        return np.asarray([int(255 * channel) for channel in rgb], dtype=np.uint8)

    def _append_track_events(self, track_events) -> None:
        if not track_events:
            return
        with self.track_events_path.open("a", encoding="utf-8") as handle:
            for event in track_events:
                handle.write(
                    json.dumps(
                        {
                            "event_type": str(event.event_type),
                            "track_id": str(event.track_id),
                            "proposal_idx": None if event.proposal_idx is None else int(event.proposal_idx),
                            "score": float(event.score),
                            "keyframe_id": int(event.keyframe_id),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _object_state_stats(self) -> dict[str, int]:
        return self.track_manager.state_stats()

    def _write_status(self, total_keyframes: int, status: str, processed_now: int) -> None:
        stats = self._object_state_stats()
        payload = {
            "scene_dir": str(self.scene_dir),
            "export_dir": str(self.export_dir),
            "output_dir": str(self.output_dir),
            "status": status,
            "total_keyframes_seen": int(total_keyframes),
            "processed_keyframes": int(len(self.processed_keyframe_ids)),
            "processed_in_last_pass": int(processed_now),
            "num_objects": int(stats["num_objects"]),
            "num_active_tracks": int(stats["num_active"]),
            "num_pending_objects": int(stats["num_pending"]),
            "num_dormant_tracks": int(stats["num_dormant"]),
            "num_recent_reactivations": int(self.last_recent_reactivations),
            "num_track_events_last_pass": int(self.last_track_events_last_pass),
            "num_observations": int(self.total_observations),
            "artifact_path": str(self.snapshot_path),
            "semantic_frontend": "legacy_yolo_only_persistent_tracks",
        }
        self.status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runner = OnlineSemanticObserverRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
