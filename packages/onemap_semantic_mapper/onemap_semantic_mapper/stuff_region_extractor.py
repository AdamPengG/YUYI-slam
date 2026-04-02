from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .semantic_observer import MaskObservation


class StuffRegionExtractor:
    """Lightweight structural-region proposer for floor/wall/ceiling style nodes."""

    def __init__(
        self,
        class_names: Iterable[str],
        min_region_area: int = 4096,
        thing_exclusion_dilate_px: int = 9,
    ) -> None:
        self.class_names = {str(name) for name in class_names}
        self.min_region_area = max(int(min_region_area), 256)
        self.thing_exclusion_dilate_px = max(int(thing_exclusion_dilate_px), 0)

    def build_mask_observations(
        self,
        image_rgb: np.ndarray,
        depth_m: np.ndarray | None,
        thing_masks: list[np.ndarray] | None = None,
        start_mask_id: int = 10_000,
    ) -> list[MaskObservation]:
        if depth_m is None or depth_m.size == 0:
            return []
        height, width = depth_m.shape[:2]
        valid_depth = np.isfinite(depth_m) & (depth_m > 0.05)
        if not np.any(valid_depth):
            return []

        exclude_mask = np.zeros((height, width), dtype=np.uint8)
        for thing_mask in thing_masks or []:
            if thing_mask.shape != valid_depth.shape:
                continue
            exclude_mask |= thing_mask.astype(np.uint8)
        if self.thing_exclusion_dilate_px > 0 and np.any(exclude_mask):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.thing_exclusion_dilate_px * 2 + 1, self.thing_exclusion_dilate_px * 2 + 1),
            )
            exclude_mask = cv2.dilate(exclude_mask, kernel)
        exclude = exclude_mask.astype(bool)

        observations: list[MaskObservation] = []
        next_mask_id = int(start_mask_id)
        band_specs = [
            ("ceiling", slice(0, max(int(height * 0.22), 1)), ("ceiling", "beam", "wall")),
            ("wall", slice(int(height * 0.15), int(height * 0.82)), ("wall", "window", "door", "blinds")),
            ("floor", slice(int(height * 0.62), height), ("floor", "rug", "mat", "carpet")),
        ]

        for region_kind, row_slice, priors in band_specs:
            band_mask = np.zeros_like(valid_depth, dtype=bool)
            band_mask[row_slice, :] = True
            mask = valid_depth & band_mask & (~exclude)
            mask = self._clean_region(mask)
            if int(mask.sum()) < self.min_region_area:
                continue
            components = self._largest_components(mask, max_components=2)
            for component in components:
                if int(component.sum()) < self.min_region_area:
                    continue
                ys, xs = np.nonzero(component)
                if xs.size == 0:
                    continue
                labels, scores = self._priors_for_region(region_kind, priors)
                if not labels:
                    continue
                area_ratio = float(component.sum()) / float(max(height * width, 1))
                observations.append(
                    MaskObservation(
                        mask_id=next_mask_id,
                        binary_mask=component,
                        bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                        semantic_label_candidates=labels,
                        semantic_scores=scores,
                        semantic_embedding=None,
                        semantic_embedding_variant="stuff_prior",
                        yolo_label_candidates=labels,
                        yolo_scores=scores,
                        detection_score=float(scores[0]) if scores else 0.0,
                        observation_kind="stuff",
                        view_quality=float(np.clip(area_ratio * 4.0, 0.15, 1.0)),
                    )
                )
                next_mask_id += 1
        return observations

    def _priors_for_region(self, region_kind: str, priors: tuple[str, ...]) -> tuple[list[str], list[float]]:
        labels = [label for label in priors if label in self.class_names]
        if not labels:
            fallback = {
                "ceiling": ["ceiling", "wall"],
                "wall": ["wall"],
                "floor": ["floor"],
            }.get(region_kind, [region_kind])
            labels = [label for label in fallback if label in self.class_names]
        if not labels:
            return [], []
        base = {
            "ceiling": [0.88, 0.40, 0.18],
            "wall": [0.82, 0.30, 0.24, 0.18],
            "floor": [0.88, 0.32, 0.24, 0.16],
        }.get(region_kind, [0.70] * len(labels))
        scores = [float(base[min(idx, len(base) - 1)]) for idx in range(len(labels))]
        return labels, scores

    def _clean_region(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed > 0

    def _largest_components(self, mask: np.ndarray, max_components: int = 2) -> list[np.ndarray]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return [mask.astype(bool)]
        candidates: list[tuple[int, np.ndarray]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_region_area:
                continue
            candidates.append((area, labels == label))
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [component.astype(bool) for _, component in candidates[: max_components]]
