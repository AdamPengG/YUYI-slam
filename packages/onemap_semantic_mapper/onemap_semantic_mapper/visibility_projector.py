from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data_types import KeyframePacket, LocalCloudPacket, SensorConfig


@dataclass
class VisibleProjection:
    visible_point_indices: np.ndarray
    projected_uv: np.ndarray
    projected_depth: np.ndarray
    depth_residual: np.ndarray
    zbuffer_rank: np.ndarray
    distance_to_depth_edge: np.ndarray
    visibility_score: np.ndarray
    quality_score: np.ndarray
    point_id_buffer: np.ndarray | None = None
    zbuffer_depth_buffer: np.ndarray | None = None


class VisibilityProjector:
    def __init__(self, depth_tolerance_m: float = 0.03, prefer_semantic_subset: bool = True) -> None:
        self.depth_tolerance_m = depth_tolerance_m
        self.prefer_semantic_subset = bool(prefer_semantic_subset)

    def load_local_cloud(self, local_cloud: LocalCloudPacket) -> dict[str, np.ndarray]:
        cloud_path = Path(local_cloud.cloud_path)
        with np.load(cloud_path) as data:
            if "xyz" not in data:
                raise KeyError(f"Local cloud {cloud_path} is missing xyz")
            payload = {key: np.asarray(data[key]) for key in data.files}
        payload["xyz"] = np.asarray(payload["xyz"], dtype=np.float32)
        return payload

    def _empty_projection(self) -> VisibleProjection:
        empty_i = np.zeros((0,), dtype=np.int32)
        empty_f = np.zeros((0,), dtype=np.float32)
        return VisibleProjection(
            visible_point_indices=empty_i,
            projected_uv=np.zeros((0, 2), dtype=np.int32),
            projected_depth=empty_f,
            depth_residual=empty_f,
            zbuffer_rank=empty_i,
            distance_to_depth_edge=empty_f,
            visibility_score=empty_f,
            quality_score=empty_f,
            point_id_buffer=None,
            zbuffer_depth_buffer=None,
        )

    def _build_projection_buffers(
        self,
        width: int,
        height: int,
        point_indices: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_id_buffer = np.full((height, width), -1, dtype=np.int32)
        zbuffer_depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
        if point_indices.size == 0:
            return point_id_buffer, zbuffer_depth_buffer
        point_id_buffer[v, u] = point_indices.astype(np.int32, copy=False)
        zbuffer_depth_buffer[v, u] = depth.astype(np.float32, copy=False)
        return point_id_buffer, zbuffer_depth_buffer

    def _depth_edge_distance(self, depth_m: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth_m) & (depth_m > 0.05)
        if not np.any(valid):
            return np.zeros_like(depth_m, dtype=np.float32)
        depth_filled = depth_m.astype(np.float32, copy=True)
        depth_filled[~valid] = 0.0
        grad_x = cv2.Sobel(depth_filled, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_filled, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
        edge_mask = (~valid) | (grad_mag > max(self.depth_tolerance_m * 1.5, 0.03))
        safe_region = (~edge_mask).astype(np.uint8)
        return cv2.distanceTransform(safe_region, cv2.DIST_L2, 3).astype(np.float32)

    def project(
        self,
        keyframe: KeyframePacket,
        local_cloud: LocalCloudPacket,
        sensor_config: SensorConfig,
        depth_m: np.ndarray | None = None,
    ) -> VisibleProjection:
        cloud_data = self.load_local_cloud(local_cloud)
        xyz_world = cloud_data["xyz"]
        if xyz_world.size == 0:
            return self._empty_projection()

        if (
            self.prefer_semantic_subset
            and
            "semantic_point_indices" in cloud_data
            and "semantic_projected_uv" in cloud_data
            and "semantic_projected_depth" in cloud_data
        ):
            point_indices = np.asarray(cloud_data["semantic_point_indices"], dtype=np.int32).reshape(-1)
            visibility_score = np.asarray(
                cloud_data.get("semantic_visibility_score", np.ones((point_indices.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            quality_score = np.asarray(
                cloud_data.get("semantic_quality_score", visibility_score),
                dtype=np.float32,
            ).reshape(-1)
            depth_residual = np.asarray(
                cloud_data.get("semantic_depth_residual", np.zeros((point_indices.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            zbuffer_rank = np.asarray(
                cloud_data.get("semantic_zbuffer_rank", np.zeros((point_indices.shape[0],), dtype=np.int32)),
                dtype=np.int32,
            ).reshape(-1)
            distance_to_depth_edge = np.asarray(
                cloud_data.get("semantic_distance_to_depth_edge", np.zeros((point_indices.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            projected_uv = np.asarray(cloud_data["semantic_projected_uv"], dtype=np.int32).reshape(-1, 2)
            projected_depth = np.asarray(cloud_data["semantic_projected_depth"], dtype=np.float32).reshape(-1)

            valid = np.ones((point_indices.shape[0],), dtype=bool)
            valid &= point_indices >= 0
            valid &= point_indices < xyz_world.shape[0]
            if "semantic_visibility_flag" in cloud_data:
                valid &= np.asarray(cloud_data["semantic_visibility_flag"], dtype=np.uint8).reshape(-1) > 0
            if depth_m is not None and depth_m.size > 0 and projected_uv.size > 0:
                u = projected_uv[:, 0]
                v = projected_uv[:, 1]
                sampled_depth = depth_m[v, u]
                finite = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
                residual = sampled_depth - projected_depth
                valid &= finite & (np.abs(residual) <= self.depth_tolerance_m)
            point_indices = point_indices[valid]
            if point_indices.size == 0:
                return self._empty_projection()
            width, height = int(sensor_config.image_size[0]), int(sensor_config.image_size[1])
            point_id_buffer, zbuffer_depth_buffer = self._build_projection_buffers(
                width=width,
                height=height,
                point_indices=point_indices.astype(np.int32, copy=False),
                u=projected_uv[valid][:, 0].astype(np.int32, copy=False),
                v=projected_uv[valid][:, 1].astype(np.int32, copy=False),
                depth=projected_depth[valid].astype(np.float32, copy=False),
            )
            return VisibleProjection(
                visible_point_indices=point_indices.astype(np.int32),
                projected_uv=projected_uv[valid].astype(np.int32),
                projected_depth=projected_depth[valid].astype(np.float32),
                depth_residual=depth_residual[valid].astype(np.float32),
                zbuffer_rank=zbuffer_rank[valid].astype(np.int32),
                distance_to_depth_edge=distance_to_depth_edge[valid].astype(np.float32),
                visibility_score=visibility_score[valid].astype(np.float32),
                quality_score=quality_score[valid].astype(np.float32),
                point_id_buffer=point_id_buffer,
                zbuffer_depth_buffer=zbuffer_depth_buffer,
            )

        if (
            local_cloud.quality_fields_present
            and local_cloud.has_uv
            and local_cloud.has_visibility_flag
            and "projected_uv" in cloud_data
            and "projected_depth" in cloud_data
        ):
            point_indices = np.arange(xyz_world.shape[0], dtype=np.int32)
            visibility_score = np.asarray(
                cloud_data.get("visibility_score", np.ones((xyz_world.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            quality_score = np.asarray(
                cloud_data.get("quality_score", visibility_score),
                dtype=np.float32,
            ).reshape(-1)
            depth_residual = np.asarray(
                cloud_data.get("depth_residual", np.zeros((xyz_world.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            zbuffer_rank = np.asarray(
                cloud_data.get("zbuffer_rank", np.zeros((xyz_world.shape[0],), dtype=np.int32)),
                dtype=np.int32,
            ).reshape(-1)
            distance_to_depth_edge = np.asarray(
                cloud_data.get("distance_to_depth_edge", np.zeros((xyz_world.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            projected_uv = np.asarray(cloud_data["projected_uv"], dtype=np.int32).reshape(-1, 2)
            projected_depth = np.asarray(cloud_data["projected_depth"], dtype=np.float32).reshape(-1)

            valid = np.ones((point_indices.shape[0],), dtype=bool)
            if "visibility_flag" in cloud_data:
                valid &= np.asarray(cloud_data["visibility_flag"], dtype=np.uint8).reshape(-1) > 0
            if depth_m is not None and depth_m.size > 0 and projected_uv.size > 0:
                u = projected_uv[:, 0]
                v = projected_uv[:, 1]
                sampled_depth = depth_m[v, u]
                finite = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
                residual = sampled_depth - projected_depth
                valid &= finite & (np.abs(residual) <= self.depth_tolerance_m)
            point_indices = point_indices[valid]
            if point_indices.size == 0:
                return self._empty_projection()
            width, height = int(sensor_config.image_size[0]), int(sensor_config.image_size[1])
            point_id_buffer, zbuffer_depth_buffer = self._build_projection_buffers(
                width=width,
                height=height,
                point_indices=point_indices.astype(np.int32, copy=False),
                u=projected_uv[valid][:, 0].astype(np.int32, copy=False),
                v=projected_uv[valid][:, 1].astype(np.int32, copy=False),
                depth=projected_depth[valid].astype(np.float32, copy=False),
            )
            return VisibleProjection(
                visible_point_indices=point_indices.astype(np.int32),
                projected_uv=projected_uv[valid].astype(np.int32),
                projected_depth=projected_depth[valid].astype(np.float32),
                depth_residual=depth_residual[valid].astype(np.float32),
                zbuffer_rank=zbuffer_rank[valid].astype(np.int32),
                distance_to_depth_edge=distance_to_depth_edge[valid].astype(np.float32),
                visibility_score=visibility_score[valid].astype(np.float32),
                quality_score=quality_score[valid].astype(np.float32),
                point_id_buffer=point_id_buffer,
                zbuffer_depth_buffer=zbuffer_depth_buffer,
            )

        c2w = np.asarray(keyframe.t_world_cam, dtype=np.float32).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        xyz_h = np.hstack((xyz_world, np.ones((xyz_world.shape[0], 1), dtype=np.float32)))
        xyz_cam = (w2c @ xyz_h.T).T[:, :3]

        z = xyz_cam[:, 2]
        in_front = z > 0.05
        if not np.any(in_front):
            return self._empty_projection()

        xyz_cam = xyz_cam[in_front]
        point_indices = np.nonzero(in_front)[0]
        fx = float(sensor_config.intrinsics["fx"])
        fy = float(sensor_config.intrinsics["fy"])
        cx = float(sensor_config.intrinsics["cx"])
        cy = float(sensor_config.intrinsics["cy"])
        width, height = int(sensor_config.image_size[0]), int(sensor_config.image_size[1])

        u = np.rint((xyz_cam[:, 0] * fx / xyz_cam[:, 2]) + cx).astype(np.int32)
        v = np.rint((xyz_cam[:, 1] * fy / xyz_cam[:, 2]) + cy).astype(np.int32)
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_image):
            return self._empty_projection()

        u = u[in_image]
        v = v[in_image]
        z = xyz_cam[in_image, 2]
        point_indices = point_indices[in_image]

        pixel_id = v.astype(np.int64) * width + u.astype(np.int64)
        order = np.lexsort((z, pixel_id))
        pixel_id = pixel_id[order]
        z = z[order]
        u = u[order]
        v = v[order]
        point_indices = point_indices[order]
        zbuffer_rank = np.zeros_like(z, dtype=np.int32)
        for idx in range(1, pixel_id.shape[0]):
            if pixel_id[idx] == pixel_id[idx - 1]:
                zbuffer_rank[idx] = zbuffer_rank[idx - 1] + 1
        _, unique_indices = np.unique(pixel_id, return_index=True)

        point_indices = point_indices[unique_indices]
        u = u[unique_indices]
        v = v[unique_indices]
        z = z[unique_indices]
        zbuffer_rank = zbuffer_rank[unique_indices]

        visibility = np.ones_like(z, dtype=np.float32)
        depth_residual = np.zeros_like(z, dtype=np.float32)
        depth_edge_distance = np.full_like(z, 10.0, dtype=np.float32)
        quality_score = np.ones_like(z, dtype=np.float32)
        if depth_m is not None and depth_m.size > 0:
            depth_edge = self._depth_edge_distance(depth_m)
            sampled_depth = depth_m[v, u]
            finite = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
            visibility = np.zeros_like(z, dtype=np.float32)
            depth_residual = sampled_depth - z
            depth_edge_distance = depth_edge[v, u]
            visibility[finite] = np.clip(
                1.0 - (np.abs(depth_residual[finite]) / max(self.depth_tolerance_m, 1e-4)),
                0.0,
                1.0,
            )
            depth_consistent = finite & (np.abs(sampled_depth - z) <= self.depth_tolerance_m)
            point_indices = point_indices[depth_consistent]
            u = u[depth_consistent]
            v = v[depth_consistent]
            z = z[depth_consistent]
            zbuffer_rank = zbuffer_rank[depth_consistent]
            depth_residual = depth_residual[depth_consistent]
            depth_edge_distance = depth_edge_distance[depth_consistent]
            visibility = visibility[depth_consistent]
            edge_score = np.clip(depth_edge_distance / 4.0, 0.0, 1.0)
            rank_score = np.clip(1.0 - (zbuffer_rank.astype(np.float32) * 0.25), 0.0, 1.0)
            quality_score = np.clip((visibility * 0.65) + (edge_score * 0.25) + (rank_score * 0.10), 0.0, 1.0)
        else:
            quality_score = np.clip(1.0 - (zbuffer_rank.astype(np.float32) * 0.10), 0.1, 1.0)

        point_id_buffer, zbuffer_depth_buffer = self._build_projection_buffers(
            width=width,
            height=height,
            point_indices=point_indices.astype(np.int32, copy=False),
            u=u.astype(np.int32, copy=False),
            v=v.astype(np.int32, copy=False),
            depth=z.astype(np.float32, copy=False),
        )

        return VisibleProjection(
            visible_point_indices=point_indices.astype(np.int32),
            projected_uv=np.column_stack((u, v)).astype(np.int32),
            projected_depth=z.astype(np.float32),
            depth_residual=depth_residual.astype(np.float32),
            zbuffer_rank=zbuffer_rank.astype(np.int32),
            distance_to_depth_edge=depth_edge_distance.astype(np.float32),
            visibility_score=visibility.astype(np.float32),
            quality_score=quality_score.astype(np.float32),
            point_id_buffer=point_id_buffer,
            zbuffer_depth_buffer=zbuffer_depth_buffer,
        )
