from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data_types import KeyframePacket
from .proposal_association_3d import AssociationResult, Proposal3D


@dataclass
class PersistentInstanceTrack:
    track_id: str
    track_num_id: int
    state: str

    first_seen_stamp: float
    last_seen_stamp: float
    last_update_keyframe_id: int

    hits: int
    misses: int
    consecutive_hits: int
    observation_count: int

    label_votes: dict[str, float]
    top_label: str
    stability_score: float
    completeness_score: float

    centroid_world: list[float]
    bbox_world: list[float]

    best_view_keyframes: list[int]
    appearance_bank: list[list[float]] | None

    last_assoc_score: float
    last_reproj_iou: float
    dirty_flag: bool

    geometry_store_path: str
    fused_voxel_count: int


@dataclass
class TrackEvent:
    event_type: str
    track_id: str
    proposal_idx: int | None
    score: float
    keyframe_id: int


class PersistentInstanceTrackManager:
    def __init__(
        self,
        *,
        pending_hits: int = 2,
        dormant_after_sec: float = 2.0,
        delete_after_sec: float = 30.0,
        new_track_min_points: int = 12,
    ) -> None:
        self.pending_hits = int(max(pending_hits, 1))
        self.dormant_after_sec = float(max(dormant_after_sec, 0.0))
        self.delete_after_sec = float(max(delete_after_sec, self.dormant_after_sec))
        self.new_track_min_points = int(max(new_track_min_points, 1))
        self.tracks: dict[str, PersistentInstanceTrack] = {}
        self._next_track_num_id = 0

    def live_tracks(self) -> list[PersistentInstanceTrack]:
        return [track for track in self.tracks.values() if track.state != "dead"]

    def update(
        self,
        *,
        keyframe: KeyframePacket,
        proposals: list[Proposal3D],
        association: AssociationResult,
    ) -> list[TrackEvent]:
        stamp = float(keyframe.stamp_sec)
        keyframe_id = int(keyframe.keyframe_id)
        events: list[TrackEvent] = []

        score_by_match = {(idx, track_id): 0.0 for idx, track_id in association.matches}
        reproj_by_match = {(idx, track_id): 0.0 for idx, track_id in association.matches}
        for edge in association.scored_edges:
            key = (int(edge.proposal_idx), str(edge.track_id))
            if key in score_by_match and edge.score >= score_by_match[key]:
                score_by_match[key] = float(edge.score)
                reproj_by_match[key] = float(edge.reproj_iou)

        matched_tracks: set[str] = set()
        for proposal_idx, track_id in association.matches:
            proposal = proposals[proposal_idx]
            track = self.tracks.get(track_id)
            if track is None:
                continue
            event_type = "match"
            if track.state == "dormant":
                track.state = "active"
                event_type = "reactivate"
            self._apply_hit(
                track=track,
                proposal=proposal,
                keyframe_id=keyframe_id,
                stamp=stamp,
                assoc_score=score_by_match.get((proposal_idx, track_id), 0.0),
                reproj_iou=reproj_by_match.get((proposal_idx, track_id), 0.0),
            )
            matched_tracks.add(track_id)
            events.append(
                TrackEvent(
                    event_type=event_type,
                    track_id=track_id,
                    proposal_idx=int(proposal_idx),
                    score=float(score_by_match.get((proposal_idx, track_id), 0.0)),
                    keyframe_id=keyframe_id,
                )
            )

        for proposal_idx, track_id in association.fragment_attaches:
            proposal = proposals[proposal_idx]
            track = self.tracks.get(track_id)
            if track is None:
                continue
            self._apply_hit(
                track=track,
                proposal=proposal,
                keyframe_id=keyframe_id,
                stamp=stamp,
                assoc_score=0.0,
                reproj_iou=0.0,
            )
            matched_tracks.add(track_id)
            events.append(
                TrackEvent(
                    event_type="fragment_attach",
                    track_id=track_id,
                    proposal_idx=int(proposal_idx),
                    score=0.0,
                    keyframe_id=keyframe_id,
                )
            )

        for proposal_idx in association.unmatched_proposals:
            proposal = proposals[proposal_idx]
            if proposal.point_count < self.new_track_min_points or not proposal.top_label:
                continue
            track = self._create_track(keyframe=keyframe, proposal=proposal)
            events.append(
                TrackEvent(
                    event_type="new",
                    track_id=track.track_id,
                    proposal_idx=int(proposal_idx),
                    score=1.0,
                    keyframe_id=keyframe_id,
                )
            )

        to_delete: list[str] = []
        matched_or_created_tracks = {event.track_id for event in events if event.proposal_idx is not None}
        for track_id, track in self.tracks.items():
            if track_id in matched_or_created_tracks:
                continue
            track.misses += 1
            track.consecutive_hits = 0
            age_sec = stamp - float(track.last_seen_stamp)
            if track.state in {"active", "pending"} and age_sec >= self.dormant_after_sec:
                track.state = "dormant"
                track.dirty_flag = True
            if track.state == "dormant" and age_sec >= self.delete_after_sec:
                track.state = "dead"
                track.dirty_flag = True
                to_delete.append(track_id)
                events.append(
                    TrackEvent(
                        event_type="delete",
                        track_id=track_id,
                        proposal_idx=None,
                        score=0.0,
                        keyframe_id=keyframe_id,
                    )
                )

        for track_id in to_delete:
            self.tracks.pop(track_id, None)
        return events

    def apply_fusion_summary(self, track_id: str, summary: dict[str, Any]) -> None:
        track = self.tracks.get(track_id)
        if track is None:
            return
        centroid_world = summary.get("centroid_world")
        bbox_world = summary.get("bbox_world")
        if centroid_world is not None:
            track.centroid_world = [float(v) for v in centroid_world]
        if bbox_world is not None:
            track.bbox_world = [float(v) for v in bbox_world]
        track.geometry_store_path = str(summary.get("geometry_store_path", track.geometry_store_path))
        track.fused_voxel_count = int(summary.get("fused_voxel_count", track.fused_voxel_count))
        track.completeness_score = max(
            float(track.completeness_score),
            float(summary.get("completeness_score", track.completeness_score)),
        )
        track.dirty_flag = True

    def to_metadata_dict(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for track_id, track in sorted(self.tracks.items(), key=lambda item: item[1].track_num_id):
            out[track_id] = {
                "object_id": track.track_id,
                "track_num_id": int(track.track_num_id),
                "state": str(track.state),
                "first_seen_stamp": float(track.first_seen_stamp),
                "last_seen_stamp": float(track.last_seen_stamp),
                "last_update_keyframe_id": int(track.last_update_keyframe_id),
                "hits": int(track.hits),
                "misses": int(track.misses),
                "consecutive_hits": int(track.consecutive_hits),
                "observation_count": int(track.observation_count),
                "label_votes": {str(k): float(v) for k, v in track.label_votes.items()},
                "top_label": str(track.top_label),
                "stability_score": float(track.stability_score),
                "completeness_score": float(track.completeness_score),
                "centroid_world": [float(v) for v in track.centroid_world],
                "bbox_world": [float(v) for v in track.bbox_world],
                "best_view_keyframes": [int(v) for v in track.best_view_keyframes],
                "appearance_bank": track.appearance_bank,
                "last_assoc_score": float(track.last_assoc_score),
                "last_reproj_iou": float(track.last_reproj_iou),
                "dirty_flag": bool(track.dirty_flag),
                "geometry_store_path": str(track.geometry_store_path),
                "fused_voxel_count": int(track.fused_voxel_count),
            }
        return out

    @classmethod
    def from_metadata_dict(cls, payload: dict) -> "PersistentInstanceTrackManager":
        manager = cls()
        raw_objects = payload.get("objects", payload) if isinstance(payload, dict) else {}
        if not isinstance(raw_objects, dict):
            raw_objects = {}
        max_track_num_id = -1
        for raw_track_id, raw in raw_objects.items():
            track_num_id = int(raw.get("track_num_id", cls._parse_track_num_id(raw_track_id)))
            max_track_num_id = max(max_track_num_id, track_num_id)
            appearance_bank = raw.get("appearance_bank")
            manager.tracks[str(raw_track_id)] = PersistentInstanceTrack(
                track_id=str(raw.get("object_id", raw_track_id)),
                track_num_id=track_num_id,
                state=str(raw.get("state", "active")),
                first_seen_stamp=float(raw.get("first_seen_stamp", raw.get("last_seen_stamp", 0.0))),
                last_seen_stamp=float(raw.get("last_seen_stamp", 0.0)),
                last_update_keyframe_id=int(raw.get("last_update_keyframe_id", -1)),
                hits=int(raw.get("hits", raw.get("observation_count", 0))),
                misses=int(raw.get("misses", 0)),
                consecutive_hits=int(raw.get("consecutive_hits", 0)),
                observation_count=int(raw.get("observation_count", 0)),
                label_votes={str(k): float(v) for k, v in raw.get("label_votes", {}).items()},
                top_label=str(raw.get("top_label", "")),
                stability_score=float(raw.get("stability_score", 0.0)),
                completeness_score=float(raw.get("completeness_score", 0.0)),
                centroid_world=[float(v) for v in raw.get("centroid_world", [0.0, 0.0, 0.0])],
                bbox_world=[float(v) for v in raw.get("bbox_world", [0.0] * 6)],
                best_view_keyframes=[int(v) for v in raw.get("best_view_keyframes", [])],
                appearance_bank=appearance_bank if isinstance(appearance_bank, list) else None,
                last_assoc_score=float(raw.get("last_assoc_score", 0.0)),
                last_reproj_iou=float(raw.get("last_reproj_iou", 0.0)),
                dirty_flag=bool(raw.get("dirty_flag", False)),
                geometry_store_path=str(raw.get("geometry_store_path", "")),
                fused_voxel_count=int(raw.get("fused_voxel_count", 0)),
            )
        manager._next_track_num_id = max_track_num_id + 1
        return manager

    def state_stats(self) -> dict[str, int]:
        num_pending = 0
        num_active = 0
        num_dormant = 0
        for track in self.tracks.values():
            state = str(track.state).strip().lower()
            if state == "pending":
                num_pending += 1
            elif state == "dormant":
                num_dormant += 1
            elif state != "dead":
                num_active += 1
        return {
            "num_objects": int(len(self.live_tracks())),
            "num_active": int(num_active),
            "num_pending": int(num_pending),
            "num_dormant": int(num_dormant),
        }

    def _apply_hit(
        self,
        *,
        track: PersistentInstanceTrack,
        proposal: Proposal3D,
        keyframe_id: int,
        stamp: float,
        assoc_score: float,
        reproj_iou: float,
    ) -> None:
        track.last_seen_stamp = float(stamp)
        track.last_update_keyframe_id = int(keyframe_id)
        track.hits += 1
        track.misses = 0
        track.consecutive_hits += 1
        track.observation_count += 1
        track.last_assoc_score = float(assoc_score)
        track.last_reproj_iou = float(reproj_iou)
        vote_weight = max(0.05, 0.5 * float(proposal.quality_mean) + 0.5 * float(max(assoc_score, 0.2)))
        if proposal.top_label:
            track.label_votes[proposal.top_label] = track.label_votes.get(proposal.top_label, 0.0) + vote_weight
        if proposal.appearance_feat is not None:
            if track.appearance_bank is None:
                track.appearance_bank = []
            track.appearance_bank = (
                [[float(v) for v in proposal.appearance_feat.tolist()]] + list(track.appearance_bank)
            )[:5]
        track.best_view_keyframes = ([int(keyframe_id)] + [v for v in track.best_view_keyframes if v != keyframe_id])[:5]
        track.top_label = self._top_label(track.label_votes)
        track.stability_score = self._stability(track.label_votes)
        if track.state == "pending" and track.consecutive_hits >= self.pending_hits:
            track.state = "active"
        track.dirty_flag = True

    def _create_track(self, *, keyframe: KeyframePacket, proposal: Proposal3D) -> PersistentInstanceTrack:
        track_num_id = int(self._next_track_num_id)
        self._next_track_num_id += 1
        track_id = f"obj_{track_num_id:05d}"
        label_votes = {}
        if proposal.top_label:
            score = float(proposal.label_scores[0]) if proposal.label_scores else 1.0
            label_votes[proposal.top_label] = max(score, 0.05)
        track = PersistentInstanceTrack(
            track_id=track_id,
            track_num_id=track_num_id,
            state="active" if self.pending_hits <= 1 else "pending",
            first_seen_stamp=float(keyframe.stamp_sec),
            last_seen_stamp=float(keyframe.stamp_sec),
            last_update_keyframe_id=int(keyframe.keyframe_id),
            hits=1,
            misses=0,
            consecutive_hits=1,
            observation_count=1,
            label_votes=label_votes,
            top_label=str(proposal.top_label),
            stability_score=self._stability(label_votes),
            completeness_score=min(1.0, float(proposal.point_count) / max(float(self.new_track_min_points) * 4.0, 1.0)),
            centroid_world=[float(v) for v in proposal.centroid_world.tolist()],
            bbox_world=[float(v) for v in proposal.bbox_world_aabb.tolist()],
            best_view_keyframes=[int(keyframe.keyframe_id)],
            appearance_bank=(
                [[float(v) for v in proposal.appearance_feat.tolist()]] if proposal.appearance_feat is not None else None
            ),
            last_assoc_score=1.0,
            last_reproj_iou=0.0,
            dirty_flag=True,
            geometry_store_path=f"track_submaps/{track_id}.npz",
            fused_voxel_count=0,
        )
        self.tracks[track_id] = track
        return track

    @staticmethod
    def _parse_track_num_id(track_id: str) -> int:
        try:
            return int(str(track_id).split("_")[-1])
        except Exception:
            return 0

    @staticmethod
    def _top_label(label_votes: dict[str, float]) -> str:
        if not label_votes:
            return ""
        return max(label_votes.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _stability(label_votes: dict[str, float]) -> float:
        if not label_votes:
            return 0.0
        total = float(sum(label_votes.values()))
        if total <= 1e-6:
            return 0.0
        best = float(max(label_votes.values()))
        return float(best / total)
