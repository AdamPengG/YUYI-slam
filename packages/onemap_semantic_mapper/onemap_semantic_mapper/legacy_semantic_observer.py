from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

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


class SemanticObserver:
    def __init__(self, min_mask_area: int = 64, min_hit_points: int = 12) -> None:
        self.min_mask_area = int(min_mask_area)
        self.min_hit_points = int(min_hit_points)

    def observe(
        self,
        keyframe_id: int,
        local_cloud_id: str,
        projection: VisibleProjection,
        masks: Iterable[MaskObservation | dict[str, Any]],
        existing_point_object_ids: np.ndarray | None = None,
    ) -> list[ObservationLink]:
        observations: list[ObservationLink] = []
        if projection.visible_point_indices.size == 0:
            return observations

        uv = projection.projected_uv
        visibility = projection.visibility_score

        for raw_mask in masks:
            mask = self._normalize_mask(raw_mask)
            mask_area = int(mask.binary_mask.sum())
            if mask_area < self.min_mask_area:
                continue

            hits = mask.binary_mask[uv[:, 1], uv[:, 0]]
            if hits.dtype != np.bool_:
                hits = hits.astype(bool)
            if int(hits.sum()) < self.min_hit_points:
                continue

            point_indices = projection.visible_point_indices[hits]
            candidate_object_id = None
            vote_count = int(hits.sum())
            if existing_point_object_ids is not None and existing_point_object_ids.size > 0:
                point_object_ids = existing_point_object_ids[point_indices]
                valid_obj_ids = point_object_ids[point_object_ids >= 0]
                if valid_obj_ids.size > 0:
                    values, counts = np.unique(valid_obj_ids, return_counts=True)
                    best_idx = int(np.argmax(counts))
                    candidate_object_id = str(int(values[best_idx]))
                    vote_count = int(counts[best_idx])

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
                    bbox_xyxy=mask.bbox_xyxy,
                    mask_area=mask_area,
                )
            )
        return observations

    def _normalize_mask(self, mask: MaskObservation | dict[str, Any]) -> MaskObservation:
        if isinstance(mask, MaskObservation):
            return mask

        binary_mask = np.asarray(mask.get("binary_mask", mask.get("segmentation")), dtype=bool)
        bbox = mask.get("bbox_xyxy")
        if bbox is None and "bbox" in mask:
            bbox_raw = mask["bbox"]
            if len(bbox_raw) == 4:
                x0, y0, w, h = bbox_raw
                bbox = [int(x0), int(y0), int(x0 + w), int(y0 + h)]
        if bbox is None:
            ys, xs = np.nonzero(binary_mask)
            if xs.size == 0:
                bbox = [0, 0, 0, 0]
            else:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        candidates = mask.get("semantic_label_candidates", [])
        scores = mask.get("semantic_scores", [])
        if not candidates and "label" in mask:
            candidates = [str(mask["label"])]
            scores = [float(mask.get("score", 1.0))]

        return MaskObservation(
            mask_id=int(mask.get("mask_id", mask.get("id", 0))),
            binary_mask=binary_mask,
            bbox_xyxy=[int(v) for v in bbox],
            semantic_label_candidates=[str(v) for v in candidates],
            semantic_scores=[float(v) for v in scores],
        )
