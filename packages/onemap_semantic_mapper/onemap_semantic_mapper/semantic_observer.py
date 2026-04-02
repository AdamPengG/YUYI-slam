from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

from .data_types import ObservationLink
from .visibility_projector import VisibleProjection


@dataclass
class MaskObservation:
    mask_id: int
    binary_mask: np.ndarray
    bbox_xyxy: list[int]
    semantic_label_candidates: list[str]
    semantic_scores: list[float]
    semantic_embedding: np.ndarray | None = None
    semantic_embedding_variant: str = ""
    yolo_label_candidates: list[str] = field(default_factory=list)
    yolo_scores: list[float] = field(default_factory=list)
    detection_score: float = 0.0
    observation_kind: str = "thing"
    view_quality: float = 0.0


class SemanticObserver:
    def __init__(
        self,
        min_mask_area: int = 64,
        min_hit_points: int = 12,
        mask_erosion_px: int = 0,
        abstain_margin: float = 0.15,
        min_binding_score: float = 0.45,
        foreground_depth_override_enabled: bool = True,
        foreground_depth_quantile: float = 0.12,
        foreground_depth_band_m: float = 0.06,
        foreground_override_min_pixels: int = 48,
    ) -> None:
        self.min_mask_area = max(int(min_mask_area), 1)
        self.min_hit_points = max(int(min_hit_points), 1)
        self.mask_erosion_px = max(int(mask_erosion_px), 0)
        self.abstain_margin = max(float(abstain_margin), 0.0)
        self.min_binding_score = max(float(min_binding_score), 0.0)
        self.foreground_depth_override_enabled = bool(foreground_depth_override_enabled)
        self.foreground_depth_quantile = float(np.clip(foreground_depth_quantile, 0.01, 0.50))
        self.foreground_depth_band_m = max(float(foreground_depth_band_m), 0.01)
        self.foreground_override_min_pixels = max(int(foreground_override_min_pixels), 8)

    def observe(
        self,
        keyframe_id: int,
        local_cloud_id: str,
        projection: VisibleProjection,
        masks: Iterable[MaskObservation | dict[str, Any]],
        existing_point_object_ids: np.ndarray | None = None,
        depth_m: np.ndarray | None = None,
    ) -> list[ObservationLink]:
        observations: list[ObservationLink] = []
        if projection.visible_point_indices.size == 0:
            return observations

        uv = projection.projected_uv
        visibility = projection.visibility_score
        quality = projection.quality_score
        projected_depth = projection.projected_depth
        depth_residual = projection.depth_residual
        depth_edge_distance = projection.distance_to_depth_edge

        for raw_mask in masks:
            mask = self._normalize_mask(raw_mask)
            mask_area = int(mask.binary_mask.sum())
            if mask_area < self.min_mask_area:
                continue

            core_mask = self._build_core_mask(mask.binary_mask, mask_area, mask.bbox_xyxy)
            hits = core_mask[uv[:, 1], uv[:, 0]]
            if hits.dtype != np.bool_:
                hits = hits.astype(bool)
            effective_min_hit_points = self._effective_min_hit_points(mask_area, mask.bbox_xyxy)
            if int(hits.sum()) < effective_min_hit_points:
                hits = mask.binary_mask[uv[:, 1], uv[:, 0]]
                if hits.dtype != np.bool_:
                    hits = hits.astype(bool)
            if int(hits.sum()) < effective_min_hit_points:
                continue

            point_indices = projection.visible_point_indices[hits]
            hit_uv = uv[hits]
            hit_visibility = visibility[hits]
            hit_quality = quality[hits]
            hit_depth = projected_depth[hits]
            hit_depth_residual = depth_residual[hits]
            hit_depth_edge_distance = depth_edge_distance[hits]
            dense_fg = self._dense_foreground_depth(mask.binary_mask, core_mask, depth_m)
            (
                point_indices,
                hit_uv,
                hit_visibility,
                hit_quality,
                hit_depth,
                hit_depth_residual,
                hit_depth_edge_distance,
            ) = self._apply_foreground_override(
                point_indices=point_indices,
                hit_uv=hit_uv,
                visibility=hit_visibility,
                quality=hit_quality,
                projected_depth=hit_depth,
                depth_residual=hit_depth_residual,
                depth_edge_distance=hit_depth_edge_distance,
                effective_min_hit_points=effective_min_hit_points,
                dense_foreground=dense_fg,
            )
            if point_indices.size < effective_min_hit_points:
                continue
            point_weights, fg_depth_median, fg_depth_p10 = self._compute_point_weights(
                hit_uv=hit_uv,
                bbox_xyxy=mask.bbox_xyxy,
                visibility=hit_visibility,
                quality=hit_quality,
                projected_depth=hit_depth,
                depth_residual=hit_depth_residual,
                depth_edge_distance=hit_depth_edge_distance,
                dense_foreground=dense_fg,
            )
            candidate_object_id = None
            candidate_object_scores: dict[str, float] = {}
            vote_count = int(hits.sum())
            abstained = False
            abstain_reason = ""
            if existing_point_object_ids is not None and existing_point_object_ids.size > 0:
                point_object_ids = existing_point_object_ids[point_indices]
                weighted_scores: dict[str, float] = {}
                weighted_counts: dict[str, int] = {}
                for obj_id, point_weight in zip(point_object_ids.tolist(), point_weights.tolist(), strict=False):
                    if obj_id is None or not str(obj_id):
                        continue
                    object_id = str(obj_id)
                    weighted_scores[object_id] = weighted_scores.get(object_id, 0.0) + float(point_weight)
                    weighted_counts[object_id] = weighted_counts.get(object_id, 0) + 1
                if weighted_scores:
                    total_weight = max(float(sum(weighted_scores.values())), 1e-6)
                    candidate_object_scores = {
                        object_id: float(score) / total_weight for object_id, score in weighted_scores.items()
                    }
                    ranked = sorted(candidate_object_scores.items(), key=lambda item: item[1], reverse=True)
                    best_object_id, best_score = ranked[0]
                    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                    if best_score >= self.min_binding_score:
                        candidate_object_id = str(best_object_id)
                        vote_count = int(weighted_counts.get(best_object_id, 0))
                        if (best_score - second_score) < self.abstain_margin:
                            abstain_reason = "weak_existing_binding"
                    elif len(ranked) > 0:
                        candidate_object_id = str(best_object_id)
                        vote_count = int(weighted_counts.get(best_object_id, 0))
                        abstain_reason = "fallback_existing_binding"

            observations.append(
                ObservationLink(
                    keyframe_id=keyframe_id,
                    mask_id=mask.mask_id,
                    local_cloud_id=local_cloud_id,
                    point_indices=[int(v) for v in point_indices.tolist()],
                    semantic_label_candidates=mask.semantic_label_candidates,
                    semantic_scores=mask.semantic_scores,
                    candidate_object_id=candidate_object_id,
                    vote_count=vote_count,
                    visibility_score=float(np.mean(visibility[hits])) if np.any(hits) else 0.0,
                    quality_score=float(np.mean(point_weights)) if point_weights.size > 0 else 0.0,
                    bbox_xyxy=mask.bbox_xyxy,
                    mask_area=mask_area,
                    foreground_depth_median=dense_fg["median"] if dense_fg is not None else fg_depth_median,
                    foreground_depth_p10=dense_fg["front"] if dense_fg is not None else fg_depth_p10,
                    candidate_object_scores=candidate_object_scores,
                    abstained=abstained,
                    abstain_reason=abstain_reason,
                    semantic_embedding=[]
                    if mask.semantic_embedding is None
                    else [float(v) for v in np.asarray(mask.semantic_embedding, dtype=np.float32).tolist()],
                    semantic_embedding_variant=str(mask.semantic_embedding_variant),
                    yolo_label_candidates=list(mask.yolo_label_candidates),
                    yolo_scores=[float(v) for v in mask.yolo_scores],
                    detection_score=float(mask.detection_score),
                    observation_kind=str(mask.observation_kind),
                    view_quality=float(mask.view_quality),
                )
            )
        return observations

    def _compute_point_weights(
        self,
        hit_uv: np.ndarray,
        bbox_xyxy: list[int],
        visibility: np.ndarray,
        quality: np.ndarray,
        projected_depth: np.ndarray,
        depth_residual: np.ndarray,
        depth_edge_distance: np.ndarray,
        dense_foreground: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, float | None, float | None]:
        if hit_uv.size == 0:
            return np.zeros((0,), dtype=np.float32), None, None
        bbox_min = np.asarray([bbox_xyxy[0], bbox_xyxy[1]], dtype=np.float32)
        bbox_max = np.asarray([bbox_xyxy[2], bbox_xyxy[3]], dtype=np.float32)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bbox_size = np.maximum(bbox_max - bbox_min, 1.0)
        norm_xy = (hit_uv.astype(np.float32) - bbox_center[None, :]) / bbox_size[None, :]
        center_dist = np.linalg.norm(norm_xy, axis=1)
        center_weight = np.clip(1.0 - center_dist, 0.15, 1.0)

        fg_depth_median = float(np.median(projected_depth)) if projected_depth.size > 0 else None
        fg_depth_p10 = float(np.percentile(projected_depth, 10.0)) if projected_depth.size > 0 else None
        if dense_foreground is not None:
            front_depth = float(dense_foreground["front"])
            foreground_band = float(dense_foreground["band"])
            foreground_limit = float(dense_foreground["limit"])
            foreground_weight = np.clip(
                1.0
                - np.maximum(projected_depth - foreground_limit, 0.0)
                / max(foreground_band, 1e-4),
                0.01,
                1.0,
            )
            fg_depth_median = float(dense_foreground["median"])
            fg_depth_p10 = front_depth
        elif fg_depth_p10 is None:
            foreground_weight = np.ones_like(projected_depth, dtype=np.float32)
        else:
            foreground_margin = np.maximum(0.05, np.std(projected_depth) * 0.5)
            foreground_weight = np.clip(
                1.0 - np.maximum(projected_depth - (fg_depth_p10 + foreground_margin), 0.0) / max(foreground_margin, 1e-4),
                0.05,
                1.0,
            )

        residual_weight = np.clip(1.0 - (np.abs(depth_residual) / 0.03), 0.05, 1.0)
        edge_weight = np.clip(depth_edge_distance / 4.0, 0.05, 1.0)
        point_weights = visibility * quality * center_weight * foreground_weight * residual_weight * edge_weight
        point_weights = np.clip(point_weights.astype(np.float32), 0.01, 1.0)
        return point_weights, fg_depth_median, fg_depth_p10

    def _build_core_mask(self, binary_mask: np.ndarray, mask_area: int, bbox_xyxy: list[int]) -> np.ndarray:
        effective_core_px = self._effective_core_erosion_px(mask_area, bbox_xyxy)
        if effective_core_px <= 0 or not binary_mask.any():
            return binary_mask
        kernel_size = (effective_core_px * 2) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
        if int(eroded.sum()) <= 0:
            return binary_mask
        return eroded.astype(bool)

    def _dense_foreground_depth(
        self,
        binary_mask: np.ndarray,
        core_mask: np.ndarray,
        depth_m: np.ndarray | None,
    ) -> dict[str, float] | None:
        if not self.foreground_depth_override_enabled or depth_m is None or depth_m.size == 0:
            return None
        stats_mask = core_mask if int(core_mask.sum()) >= self.foreground_override_min_pixels else binary_mask
        sampled_depth = depth_m[stats_mask]
        valid_depth = sampled_depth[np.isfinite(sampled_depth) & (sampled_depth > 0.05)]
        if valid_depth.size < self.foreground_override_min_pixels:
            return None
        front_depth = float(np.percentile(valid_depth, self.foreground_depth_quantile * 100.0))
        median_depth = float(np.median(valid_depth))
        robust_spread = float(max(median_depth - front_depth, 0.0))
        adaptive_band = float(np.clip(max(self.foreground_depth_band_m, robust_spread * 0.75), self.foreground_depth_band_m, 0.15))
        return {
            "front": front_depth,
            "median": median_depth,
            "band": adaptive_band,
            "limit": front_depth + adaptive_band,
        }

    def _apply_foreground_override(
        self,
        point_indices: np.ndarray,
        hit_uv: np.ndarray,
        visibility: np.ndarray,
        quality: np.ndarray,
        projected_depth: np.ndarray,
        depth_residual: np.ndarray,
        depth_edge_distance: np.ndarray,
        effective_min_hit_points: int,
        dense_foreground: dict[str, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if dense_foreground is None or projected_depth.size == 0:
            return (
                point_indices,
                hit_uv,
                visibility,
                quality,
                projected_depth,
                depth_residual,
                depth_edge_distance,
            )
        foreground_limit = float(dense_foreground["limit"])
        keep_foreground = projected_depth <= foreground_limit
        relaxed_min_points = max(6, min(effective_min_hit_points, 8))
        if int(np.count_nonzero(keep_foreground)) < relaxed_min_points:
            return (
                point_indices,
                hit_uv,
                visibility,
                quality,
                projected_depth,
                depth_residual,
                depth_edge_distance,
            )
        return (
            point_indices[keep_foreground],
            hit_uv[keep_foreground],
            visibility[keep_foreground],
            quality[keep_foreground],
            projected_depth[keep_foreground],
            depth_residual[keep_foreground],
            depth_edge_distance[keep_foreground],
        )

    def _normalize_mask(self, mask: MaskObservation | dict[str, Any]) -> MaskObservation:
        if isinstance(mask, MaskObservation):
            raw_binary_mask = np.asarray(mask.binary_mask, dtype=bool)
            bbox = [int(v) for v in mask.bbox_xyxy]
            mask_id = int(mask.mask_id)
            candidates = [str(v) for v in mask.semantic_label_candidates]
            scores = [float(v) for v in mask.semantic_scores]
            yolo_candidates = [str(v) for v in mask.yolo_label_candidates]
            yolo_scores = [float(v) for v in mask.yolo_scores]
            detection_score = float(mask.detection_score)
            observation_kind = str(mask.observation_kind)
            view_quality = float(mask.view_quality)
        else:
            raw_binary_mask = np.asarray(mask.get("binary_mask", mask.get("segmentation")), dtype=bool)
            bbox = mask.get("bbox_xyxy")
            if bbox is None and "bbox" in mask:
                bbox_raw = mask["bbox"]
                if len(bbox_raw) == 4:
                    x0, y0, w, h = bbox_raw
                    bbox = [int(x0), int(y0), int(x0 + w), int(y0 + h)]
            mask_id = int(mask.get("mask_id", mask.get("id", 0)))
            candidates = mask.get("semantic_label_candidates", [])
            scores = mask.get("semantic_scores", [])
            if not candidates and "label" in mask:
                candidates = [str(mask["label"])]
                scores = [float(mask.get("score", 1.0))]
            yolo_candidates = mask.get("yolo_label_candidates", candidates)
            yolo_scores = mask.get("yolo_scores", scores)
            detection_score = float(mask.get("detection_score", mask.get("score", 0.0)))
            observation_kind = str(mask.get("observation_kind", "thing"))
            view_quality = float(mask.get("view_quality", 0.0))

        binary_mask = raw_binary_mask
        raw_mask_area = int(binary_mask.sum())
        if bbox is None:
            ys, xs = np.nonzero(binary_mask)
            if xs.size == 0:
                bbox = [0, 0, 0, 0]
            else:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        effective_erosion_px = self._effective_mask_erosion_px(raw_mask_area, bbox)
        if effective_erosion_px > 0 and binary_mask.any():
            kernel_size = (effective_erosion_px * 2) + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
            if int(eroded.sum()) > 0:
                binary_mask = eroded.astype(bool)

        return MaskObservation(
            mask_id=mask_id,
            binary_mask=binary_mask,
            bbox_xyxy=[int(v) for v in bbox],
            semantic_label_candidates=[str(v) for v in candidates],
            semantic_scores=[float(v) for v in scores],
            semantic_embedding=None
            if getattr(mask, "semantic_embedding", None) is None
            else np.asarray(mask.semantic_embedding, dtype=np.float32),
            semantic_embedding_variant=str(getattr(mask, "semantic_embedding_variant", "")),
            yolo_label_candidates=[str(v) for v in yolo_candidates],
            yolo_scores=[float(v) for v in yolo_scores],
            detection_score=float(detection_score),
            observation_kind=str(observation_kind),
            view_quality=float(view_quality),
        )

    def _effective_mask_erosion_px(self, mask_area: int, bbox_xyxy: list[int]) -> int:
        if self.mask_erosion_px <= 0:
            return 0
        bbox_w = max(int(bbox_xyxy[2]) - int(bbox_xyxy[0]), 0)
        bbox_h = max(int(bbox_xyxy[3]) - int(bbox_xyxy[1]), 0)
        min_side = min(bbox_w, bbox_h)
        if mask_area < max(self.min_mask_area * 3, 256) or min_side < 18:
            return 0
        if mask_area < max(self.min_mask_area * 8, 768) or min_side < 36:
            return min(self.mask_erosion_px, 1)
        return self.mask_erosion_px

    def _effective_core_erosion_px(self, mask_area: int, bbox_xyxy: list[int]) -> int:
        bbox_w = max(int(bbox_xyxy[2]) - int(bbox_xyxy[0]), 0)
        bbox_h = max(int(bbox_xyxy[3]) - int(bbox_xyxy[1]), 0)
        min_side = min(bbox_w, bbox_h)
        if mask_area < max(self.min_mask_area * 3, 256) or min_side < 18:
            return 0
        if mask_area < max(self.min_mask_area * 8, 768) or min_side < 36:
            return 1
        return 2

    def _effective_min_hit_points(self, mask_area: int, bbox_xyxy: list[int]) -> int:
        bbox_w = max(int(bbox_xyxy[2]) - int(bbox_xyxy[0]), 0)
        bbox_h = max(int(bbox_xyxy[3]) - int(bbox_xyxy[1]), 0)
        min_side = min(bbox_w, bbox_h)
        if mask_area < max(self.min_mask_area * 3, 256) or min_side < 18:
            return max(2, min(self.min_hit_points, 2))
        if mask_area < max(self.min_mask_area * 8, 768) or min_side < 36:
            return max(3, min(self.min_hit_points, 3))
        return self.min_hit_points
