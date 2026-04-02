from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .data_types import KeyframePacket, LocalCloudPacket, ObservationLink, SensorConfig
from .legacy_semantic_observer import MaskObservation
from .visibility_projector import VisibleProjection


@dataclass
class Proposal3D:
    keyframe_id: int
    local_cloud_id: str
    mask_id: int
    observation_index: int

    top_label: str
    label_candidates: list[str]
    label_scores: list[float]

    bbox_xyxy: list[int]
    mask_area: int
    binary_mask: np.ndarray

    point_indices: np.ndarray
    support_xyz_world: np.ndarray
    projected_uv: np.ndarray

    centroid_world: np.ndarray
    bbox_world_aabb: np.ndarray

    point_count: int
    visibility_mean: float
    quality_mean: float

    appearance_feat: np.ndarray | None
    candidate_track_num_id: int | None


@dataclass
class AssociationEdge:
    proposal_idx: int
    track_id: str
    score: float
    reproj_iou: float
    surface_hit_ratio: float
    bbox3d_overlap: float
    centroid_dist: float
    appearance_score: float


@dataclass
class AssociationResult:
    matches: list[tuple[int, str]]
    fragment_attaches: list[tuple[int, str]]
    unmatched_proposals: list[int]
    unmatched_tracks: list[str]
    scored_edges: list[AssociationEdge]


class ProposalAssociation3D:
    def __init__(
        self,
        *,
        assoc_score_min: float = 0.42,
        reproj_iou_min: float = 0.08,
        surface_hit_min: float = 0.20,
        reproj_dilate_px: int = 3,
        use_appearance: bool = True,
        use_depth_dense_fallback: bool = False,
        support_expansion_radius_m: float = 0.20,
        support_expansion_max_points: int = 6000,
    ) -> None:
        self.assoc_score_min = float(assoc_score_min)
        self.reproj_iou_min = float(reproj_iou_min)
        self.surface_hit_min = float(surface_hit_min)
        self.reproj_dilate_px = int(max(reproj_dilate_px, 0))
        self.use_appearance = bool(use_appearance)
        self.use_depth_dense_fallback = bool(use_depth_dense_fallback)
        self.support_expansion_radius_m = float(max(support_expansion_radius_m, 0.0))
        self.support_expansion_max_points = int(max(support_expansion_max_points, 0))

    def build_proposals(
        self,
        *,
        keyframe: KeyframePacket,
        local_cloud: LocalCloudPacket,
        projection: VisibleProjection,
        mask_observations: list[MaskObservation],
        observations: list[ObservationLink],
        image_rgb: np.ndarray,
        depth_m: np.ndarray | None,
    ) -> list[Proposal3D]:
        del depth_m
        with np.load(local_cloud.cloud_path, allow_pickle=False) as data:
            xyz_world = np.asarray(data["xyz"], dtype=np.float32)

        if xyz_world.size == 0 or projection.visible_point_indices.size == 0:
            return []

        mask_by_id = {int(mask.mask_id): mask for mask in mask_observations}
        visible_slot_by_point_index = {
            int(point_idx): slot for slot, point_idx in enumerate(projection.visible_point_indices.tolist())
        }
        proposals: list[Proposal3D] = []

        for obs_index, obs in enumerate(observations):
            point_indices = np.unique(np.asarray(obs.point_indices, dtype=np.int32))
            valid = point_indices[(point_indices >= 0) & (point_indices < xyz_world.shape[0])]
            if valid.size == 0:
                continue

            slots = np.asarray(
                [visible_slot_by_point_index[idx] for idx in valid.tolist() if idx in visible_slot_by_point_index],
                dtype=np.int32,
            )
            expanded_indices = self._expand_support_indices(
                xyz_world=xyz_world,
                support_indices=valid,
            )
            support_xyz_world = xyz_world[expanded_indices]
            if support_xyz_world.size == 0:
                continue

            mask = mask_by_id.get(int(obs.mask_id))
            if mask is None:
                continue

            projected_uv = (
                projection.projected_uv[slots].astype(np.int32, copy=False)
                if slots.size > 0
                else np.zeros((0, 2), dtype=np.int32)
            )
            visibility_mean = (
                float(np.mean(projection.visibility_score[slots])) if slots.size > 0 else float(obs.visibility_score)
            )
            quality_mean = float(np.mean(projection.quality_score[slots])) if slots.size > 0 else 0.0
            bbox_world_aabb = np.concatenate(
                [
                    support_xyz_world.min(axis=0).astype(np.float32, copy=False),
                    support_xyz_world.max(axis=0).astype(np.float32, copy=False),
                ]
            ).astype(np.float32, copy=False)
            centroid_world = support_xyz_world.mean(axis=0, dtype=np.float32)
            appearance_feat = self._compute_appearance(image_rgb, mask.binary_mask) if self.use_appearance else None

            proposals.append(
                Proposal3D(
                    keyframe_id=int(keyframe.keyframe_id),
                    local_cloud_id=str(local_cloud.local_cloud_id),
                    mask_id=int(obs.mask_id),
                    observation_index=int(obs_index),
                    top_label=str(obs.semantic_label_candidates[0]) if obs.semantic_label_candidates else "unknown",
                    label_candidates=[str(v) for v in obs.semantic_label_candidates],
                    label_scores=[float(v) for v in obs.semantic_scores],
                    bbox_xyxy=[int(v) for v in obs.bbox_xyxy],
                    mask_area=int(obs.mask_area),
                    binary_mask=np.asarray(mask.binary_mask, dtype=bool),
                    point_indices=expanded_indices.astype(np.int32, copy=False),
                    support_xyz_world=support_xyz_world.astype(np.float32, copy=False),
                    projected_uv=projected_uv,
                    centroid_world=centroid_world,
                    bbox_world_aabb=bbox_world_aabb,
                    point_count=int(valid.size),
                    visibility_mean=float(visibility_mean),
                    quality_mean=float(quality_mean),
                    appearance_feat=appearance_feat,
                    candidate_track_num_id=None,
                )
            )

        return proposals

    def associate(
        self,
        *,
        keyframe: KeyframePacket,
        sensor_config: SensorConfig,
        depth_m: np.ndarray | None,
        proposals: list[Proposal3D],
        track_manager: Any,
        track_fuser: Any,
    ) -> AssociationResult:
        del depth_m
        live_tracks = list(track_manager.live_tracks())
        if not proposals or not live_tracks:
            unmatched_tracks = [str(track.track_id) for track in live_tracks]
            return AssociationResult([], [], list(range(len(proposals))), unmatched_tracks, [])

        edges: list[AssociationEdge] = []
        for proposal_idx, proposal in enumerate(proposals):
            for track in live_tracks:
                class_score = self._class_score(proposal.top_label, track.top_label, track.label_votes)
                if class_score <= 0.0:
                    continue

                centroid_dist = float(
                    np.linalg.norm(np.asarray(track.centroid_world, dtype=np.float32) - proposal.centroid_world)
                )
                adaptive_gate = self._adaptive_gate(track.bbox_world)
                reproj_iou = float(
                    track_fuser.reprojected_bbox_iou(
                        track_id=track.track_id,
                        proposal_bbox=proposal.bbox_xyxy,
                        keyframe=keyframe,
                        sensor_config=sensor_config,
                        dilate_px=self.reproj_dilate_px,
                    )
                )
                surface_hit_ratio = float(track_fuser.surface_hit_ratio(track.track_id, proposal.support_xyz_world))
                bbox3d_overlap = float(track_fuser.bbox3d_overlap(track.track_id, proposal.bbox_world_aabb))
                appearance_score = float(
                    self._appearance_score(track.appearance_bank, proposal.appearance_feat) if self.use_appearance else 0.0
                )

                if not (
                    reproj_iou >= 0.05
                    or surface_hit_ratio >= self.surface_hit_min
                    or centroid_dist <= adaptive_gate
                ):
                    continue

                score = (
                    0.40 * reproj_iou
                    + 0.25 * surface_hit_ratio
                    + 0.15 * class_score
                    + 0.10 * bbox3d_overlap
                    + 0.10 * appearance_score
                )
                if score < self.assoc_score_min:
                    continue
                edges.append(
                    AssociationEdge(
                        proposal_idx=int(proposal_idx),
                        track_id=str(track.track_id),
                        score=float(score),
                        reproj_iou=float(reproj_iou),
                        surface_hit_ratio=float(surface_hit_ratio),
                        bbox3d_overlap=float(bbox3d_overlap),
                        centroid_dist=float(centroid_dist),
                        appearance_score=float(appearance_score),
                    )
                )

        matches: list[tuple[int, str]] = []
        matched_proposals: set[int] = set()
        matched_tracks: set[str] = set()
        for edge in sorted(edges, key=lambda item: item.score, reverse=True):
            if edge.proposal_idx in matched_proposals or edge.track_id in matched_tracks:
                continue
            matches.append((int(edge.proposal_idx), str(edge.track_id)))
            matched_proposals.add(int(edge.proposal_idx))
            matched_tracks.add(str(edge.track_id))

        fragment_attaches: list[tuple[int, str]] = []
        primary_tracks = {track_id for _, track_id in matches}
        for edge in sorted(edges, key=lambda item: item.score, reverse=True):
            if edge.proposal_idx in matched_proposals:
                continue
            if edge.track_id not in primary_tracks:
                continue
            if edge.score < max(self.assoc_score_min + 0.08, 0.55):
                continue
            fragment_attaches.append((int(edge.proposal_idx), str(edge.track_id)))
            matched_proposals.add(int(edge.proposal_idx))

        unmatched_proposals = [idx for idx in range(len(proposals)) if idx not in matched_proposals]
        unmatched_tracks = [track.track_id for track in live_tracks if track.track_id not in matched_tracks]
        return AssociationResult(matches, fragment_attaches, unmatched_proposals, unmatched_tracks, edges)

    def _expand_support_indices(self, *, xyz_world: np.ndarray, support_indices: np.ndarray) -> np.ndarray:
        support_indices = np.unique(np.asarray(support_indices, dtype=np.int32))
        if self.support_expansion_radius_m <= 0.0 or support_indices.size == 0 or xyz_world.size == 0:
            return support_indices

        support_xyz = np.asarray(xyz_world[support_indices], dtype=np.float32)
        radius = float(self.support_expansion_radius_m)
        bbox_min = support_xyz.min(axis=0) - radius
        bbox_max = support_xyz.max(axis=0) + radius
        candidate_mask = np.all((xyz_world >= bbox_min[None, :]) & (xyz_world <= bbox_max[None, :]), axis=1)
        candidate_indices = np.nonzero(candidate_mask)[0].astype(np.int32, copy=False)
        if candidate_indices.size == 0:
            return support_indices
        if self.support_expansion_max_points > 0 and candidate_indices.size > self.support_expansion_max_points:
            center = support_xyz.mean(axis=0)
            candidate_xyz = np.asarray(xyz_world[candidate_indices], dtype=np.float32)
            dist2 = np.sum((candidate_xyz - center[None, :]) ** 2, axis=1)
            keep = np.argsort(dist2)[: self.support_expansion_max_points]
            candidate_indices = candidate_indices[keep]

        candidate_xyz = np.asarray(xyz_world[candidate_indices], dtype=np.float32)
        radius2 = radius * radius
        matched = np.zeros((candidate_xyz.shape[0],), dtype=bool)
        chunk = 256
        for start in range(0, support_xyz.shape[0], chunk):
            support_chunk = support_xyz[start : start + chunk]
            diff = candidate_xyz[:, None, :] - support_chunk[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            matched |= np.any(dist2 <= radius2, axis=1)
            if bool(np.all(matched)):
                break
        expanded = np.unique(
            np.concatenate((support_indices, candidate_indices[matched].astype(np.int32, copy=False))),
        )
        return expanded.astype(np.int32, copy=False)

    def _adaptive_gate(self, bbox_world: list[float]) -> float:
        bbox = np.asarray(bbox_world, dtype=np.float32).reshape(-1)
        if bbox.size != 6:
            return 0.6
        diag = float(np.linalg.norm(bbox[3:] - bbox[:3]))
        return float(np.clip(0.6 * diag, 0.35, 1.20))

    def _class_score(self, proposal_label: str, track_top_label: str, track_votes: dict[str, float]) -> float:
        label = str(proposal_label).strip().lower()
        track_label = str(track_top_label).strip().lower()
        if not label:
            return 0.0
        if label == track_label:
            return 1.0
        if label in {str(name).strip().lower() for name in track_votes.keys()}:
            return 0.4
        return 0.0

    def _compute_appearance(self, image_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray | None:
        mask = np.asarray(binary_mask, dtype=bool)
        if mask.ndim != 2 or not bool(mask.any()):
            return None
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            mask.astype(np.uint8),
            [8, 4, 2],
            [0, 180, 0, 256, 0, 256],
        ).flatten()
        total = float(hist.sum())
        if total <= 1e-6:
            return None
        return (hist / total).astype(np.float32)

    def _appearance_score(self, appearance_bank: list[list[float]] | None, appearance_feat: np.ndarray | None) -> float:
        if not appearance_bank or appearance_feat is None or appearance_feat.size == 0:
            return 0.0
        bank = np.asarray(appearance_bank, dtype=np.float32)
        if bank.ndim != 2 or bank.shape[1] != appearance_feat.shape[0]:
            return 0.0
        overlaps = np.minimum(bank, appearance_feat[None, :]).sum(axis=1)
        return float(np.clip(np.max(overlaps), 0.0, 1.0))
