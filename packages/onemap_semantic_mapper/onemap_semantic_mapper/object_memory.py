from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_types import LocalCloudPacket, ObjectMemory, ObservationLink

STRUCTURAL_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "blinds",
}


@dataclass
class ObjectMemoryUpdateResult:
    updated_object_ids: list[str]
    created_object_ids: list[str]
    ignored_observations: int


class ObjectMemoryStore:
    def __init__(
        self,
        min_points_per_observation: int = 12,
        min_mask_area: int = 64,
        max_best_views: int = 5,
        merge_centroid_radius_m: float = 0.75,
    ) -> None:
        self.min_points_per_observation = min_points_per_observation
        self.min_mask_area = min_mask_area
        self.max_best_views = max_best_views
        self.merge_centroid_radius_m = merge_centroid_radius_m
        self.thing_merge_centroid_radius_m = 0.35
        self.point_score_increment = 1.0
        self.point_score_decrement = 0.10
        self.point_keep_threshold = 0.03
        self.point_reassign_margin = 0.05
        self.point_stale_time_sec = 1800.0
        self.label_vote_decay = 0.97
        self.spatial_voxel_size_m = 0.12
        self.cross_cloud_reassign_radius_m = 0.18
        self.cross_cloud_positive_weight = 0.60
        self.cross_cloud_negative_weight = 0.35
        self.max_cross_cloud_refs_per_point = 24
        self.pending_commit_observations = 1
        self.pending_commit_evidence = 0.20
        self.pending_seed_points = 3
        self.descriptor_confidence_floor = 0.0
        self.observation_descriptor_match_min = 0.18
        self.observation_descriptor_conflict_max = 0.06
        self.merge_descriptor_similarity_min = 0.20
        self.merge_descriptor_conflict_max = 0.10
        self.object_stale_time_sec = 900.0
        self.object_delete_time_sec = 7200.0
        self.merge_bbox_overlap_min = 0.05
        self.merge_compact_centroid_radius_m = 0.45
        self.posterior_descriptor_weight = 0.85
        self.posterior_yolo_weight = 0.35
        self.posterior_vote_weight = 0.20
        self.objects: dict[str, ObjectMemory] = {}
        self.point_assignments: dict[str, dict[int, dict[str, object]]] = {}
        self.local_cloud_paths: dict[str, str] = {}
        self.local_cloud_xyz_cache: dict[str, np.ndarray] = {}
        self.global_voxel_index: dict[tuple[int, int, int], list[tuple[str, int]]] = {}
        self._next_object_id = 0

    def update(
        self,
        observations: list[ObservationLink],
        keyframe_stamp_sec: float,
        keyframe_id: int,
        local_cloud: LocalCloudPacket,
        visible_point_indices: list[int] | np.ndarray | None = None,
    ) -> ObjectMemoryUpdateResult:
        updated: list[str] = []
        created: list[str] = []
        ignored = 0
        xyz = self._load_xyz(local_cloud)
        self.local_cloud_paths[local_cloud.local_cloud_id] = str(local_cloud.cloud_path)
        local_states = self.point_assignments.setdefault(local_cloud.local_cloud_id, {})
        affected_object_ids: set[str] = set()
        visible_points = self._normalize_visible_points(visible_point_indices, xyz.shape[0])
        effective_observations: list[tuple[ObservationLink, dict[str, float]]] = []
        self._rebuild_global_voxel_index()

        for obs in observations:
            min_points_required = self._effective_min_points_per_observation(obs.mask_area, obs.bbox_xyxy)
            if len(obs.point_indices) < min_points_required or obs.mask_area < self.min_mask_area:
                ignored += 1
                continue

            support_xyz = xyz[np.asarray(obs.point_indices, dtype=np.int32)]
            centroid = support_xyz.mean(axis=0).astype(np.float32)
            bbox_min = support_xyz.min(axis=0).astype(np.float32)
            bbox_max = support_xyz.max(axis=0).astype(np.float32)

            object_id = self._resolve_object_id(obs, centroid)
            label_votes = self._votes_from_observation(obs)
            object_id, was_created = self._ensure_object_id(
                object_id=object_id,
                centroid=centroid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                keyframe_stamp_sec=keyframe_stamp_sec,
                keyframe_id=keyframe_id,
                label_votes=label_votes,
                node_type=self._node_type_from_observation(obs),
                yolo_label_candidates=list(obs.yolo_label_candidates),
                yolo_scores=[float(v) for v in obs.yolo_scores],
            )
            obs.candidate_object_id = object_id
            if was_created:
                created.append(object_id)
            effective_observations.append((obs, label_votes))
            affected_object_ids.add(object_id)

        observed_point_indices: set[int] = set()
        for obs, _label_votes in effective_observations:
            target_object_id = str(obs.candidate_object_id)
            if target_object_id not in self.objects:
                continue
            for point_index in obs.point_indices:
                if point_index < 0 or point_index >= xyz.shape[0]:
                    continue
                observed_point_indices.add(int(point_index))
                state = self._get_or_create_point_state(local_states, int(point_index), keyframe_stamp_sec)
                previous_object_id = state.get("current_object_id")
                self._apply_positive_evidence(state, target_object_id, self.point_score_increment)
                new_object_id = self._reconcile_point_assignment(state)
                state["last_seen_stamp"] = float(keyframe_stamp_sec)
                if previous_object_id and previous_object_id != new_object_id:
                    affected_object_ids.add(str(previous_object_id))
                if new_object_id:
                    affected_object_ids.add(str(new_object_id))

            obj = self.objects[target_object_id]
            obj.observation_count += 1
            obj.last_seen_stamp = float(keyframe_stamp_sec)
            obj.best_view_keyframes = self._update_best_views(obj.best_view_keyframes, keyframe_id)
            self._apply_label_evidence(obj, label_votes)
            self._apply_yolo_evidence(obj, obs.yolo_label_candidates, obs.yolo_scores)
            self._update_descriptor_views(obj, obs, keyframe_id)
            obj.promotion_evidence += max(float(obs.quality_score), 0.05)
            obj.dirty_flag = True
            updated.append(target_object_id)
            self._propagate_to_historical_neighbors(
                target_object_id=target_object_id,
                source_local_cloud_id=local_cloud.local_cloud_id,
                source_xyz=xyz,
                observed_point_indices=obs.point_indices,
                stamp_sec=keyframe_stamp_sec,
                affected_object_ids=affected_object_ids,
            )

        for point_index in sorted(visible_points - observed_point_indices):
            state = local_states.get(int(point_index))
            if state is None:
                continue
            previous_object_id = state.get("current_object_id")
            if not previous_object_id:
                continue
            self._apply_negative_evidence(state, str(previous_object_id), self.point_score_decrement * 0.5)
            new_object_id = self._reconcile_point_assignment(state)
            state["last_seen_stamp"] = float(keyframe_stamp_sec)
            affected_object_ids.add(str(previous_object_id))
            if new_object_id:
                affected_object_ids.add(str(new_object_id))

        self._refresh_supports_for_local_cloud(local_cloud.local_cloud_id)
        self._recompute_objects(affected_object_ids)
        self._prune_point_assignments(keyframe_stamp_sec)
        self._refresh_all_supports()
        self._recompute_objects(set(self.objects.keys()))
        self._update_object_states(keyframe_stamp_sec)
        self._merge_compatible_objects()
        self._refresh_all_supports()
        self._recompute_objects(set(self.objects.keys()))
        self._update_object_states(keyframe_stamp_sec)
        self._prune_objects()
        self._rebuild_global_voxel_index()

        return ObjectMemoryUpdateResult(updated, created, ignored)

    def to_dict(self) -> dict[str, dict]:
        return {
            "objects": {object_id: obj.to_dict() for object_id, obj in self.objects.items()},
            "local_cloud_paths": dict(self.local_cloud_paths),
            "point_assignments": {
                local_cloud_id: {
                    str(point_index): {
                        "current_object_id": payload.get("current_object_id"),
                        "current_score": float(payload.get("current_score", 0.0)),
                        "object_scores": {
                            str(object_id): float(score)
                            for object_id, score in dict(payload.get("object_scores", {})).items()
                            if float(score) > 1e-6
                        },
                        "last_seen_stamp": float(payload.get("last_seen_stamp", 0.0)),
                    }
                    for point_index, payload in point_map.items()
                }
                for local_cloud_id, point_map in self.point_assignments.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, dict]) -> "ObjectMemoryStore":
        store = cls()
        object_payloads = payload.get("objects", payload)
        for object_id, object_payload in object_payloads.items():
            store.objects[str(object_id)] = ObjectMemory.from_dict(object_payload)
        store.local_cloud_paths = {
            str(local_cloud_id): str(cloud_path)
            for local_cloud_id, cloud_path in payload.get("local_cloud_paths", {}).items()
        }
        point_payloads = payload.get("point_assignments", {})
        for local_cloud_id, point_map in point_payloads.items():
            local_states: dict[int, dict[str, object]] = {}
            for point_index, state_payload in point_map.items():
                local_states[int(point_index)] = {
                    "current_object_id": None
                    if state_payload.get("current_object_id") is None
                    else str(state_payload["current_object_id"]),
                    "current_score": float(state_payload.get("current_score", 0.0)),
                    "object_scores": {
                        str(object_id): float(score)
                        for object_id, score in state_payload.get("object_scores", {}).items()
                    },
                    "last_seen_stamp": float(state_payload.get("last_seen_stamp", 0.0)),
                }
            store.point_assignments[str(local_cloud_id)] = local_states
        if store.objects:
            max_numeric = max(int(obj_id.split("_")[-1]) for obj_id in store.objects)
            store._next_object_id = max_numeric + 1
        return store

    def _resolve_object_id(self, observation: ObservationLink, centroid: np.ndarray) -> str | None:
        if observation.abstained:
            return None
        if observation.candidate_object_id is not None and observation.candidate_object_id in self.objects:
            return observation.candidate_object_id

        observation_kind = str(getattr(observation, "observation_kind", "thing") or "thing").strip().lower()
        observation_primary = self._observation_primary_nonstruct_label(observation)
        merge_radius = self.thing_merge_centroid_radius_m if observation_kind == "thing" else self.merge_centroid_radius_m

        if observation.candidate_object_scores:
            best_id = None
            best_score = None
            for object_id, score in observation.candidate_object_scores.items():
                obj = self.objects.get(str(object_id))
                if obj is None or not self._labels_compatible(observation, obj):
                    continue
                obj_primary = self._object_primary_nonstruct_label(obj)
                if (
                    observation_kind == "thing"
                    and observation_primary
                    and obj_primary is None
                    and obj.observation_count >= 1
                ):
                    continue
                dist = float(np.linalg.norm(np.asarray(obj.centroid_world, dtype=np.float32) - centroid))
                if dist > merge_radius:
                    continue
                combined_score = float(score) - min(dist / max(merge_radius, 1e-4), 1.0) * 0.25
                if best_score is None or combined_score > best_score:
                    best_id = str(object_id)
                    best_score = combined_score
            if best_id is not None:
                return best_id

        best_id = None
        best_dist = None
        for object_id, obj in self.objects.items():
            if not self._labels_compatible(observation, obj):
                continue
            obj_primary = self._object_primary_nonstruct_label(obj)
            if (
                observation_kind == "thing"
                and observation_primary
                and obj_primary is None
                and obj.observation_count >= 1
            ):
                continue
            dist = float(np.linalg.norm(np.asarray(obj.centroid_world, dtype=np.float32) - centroid))
            if dist <= merge_radius and (best_dist is None or dist < best_dist):
                best_id = object_id
                best_dist = dist
        return best_id

    def point_object_ids_for_local_cloud(self, local_cloud_id: str, point_count: int) -> np.ndarray:
        output = np.empty((max(int(point_count), 0),), dtype=object)
        output[:] = None
        local_states = self.point_assignments.get(local_cloud_id, {})
        for point_index, state in local_states.items():
            if point_index < 0 or point_index >= point_count:
                continue
            object_id = state.get("current_object_id")
            if object_id:
                output[point_index] = str(object_id)
        return output

    def register_support_view(
        self,
        object_id: str,
        keyframe_id: int,
        stamp_sec: float,
        semantic_label_candidates: list[str],
        semantic_scores: list[float],
        yolo_label_candidates: list[str] | None,
        yolo_scores: list[float] | None,
        semantic_embedding: list[float] | np.ndarray | None,
        quality_score: float,
    ) -> bool:
        obj = self.objects.get(str(object_id))
        if obj is None:
            return False
        if any(int(item.get("keyframe_id", -1)) == int(keyframe_id) for item in obj.descriptor_views):
            return False
        label_votes = {
            str(label): float(score) * 0.35
            for label, score in zip(semantic_label_candidates, semantic_scores)
            if float(score) > 1e-6
        }
        if label_votes:
            self._apply_label_evidence(obj, label_votes)
        self._apply_yolo_evidence(obj, yolo_label_candidates or [], yolo_scores or [])
        if semantic_embedding is not None:
            pseudo_observation = ObservationLink(
                keyframe_id=int(keyframe_id),
                mask_id=-1,
                local_cloud_id="support_view",
                point_indices=[],
                semantic_label_candidates=[str(v) for v in semantic_label_candidates],
                semantic_scores=[float(v) for v in semantic_scores],
                candidate_object_id=str(object_id),
                vote_count=0,
                visibility_score=float(quality_score),
                bbox_xyxy=[0, 0, 0, 0],
                mask_area=0,
                quality_score=float(quality_score),
                semantic_embedding=[]
                if semantic_embedding is None
                else [float(v) for v in np.asarray(semantic_embedding, dtype=np.float32).tolist()],
                semantic_embedding_variant="support_view",
                yolo_label_candidates=[str(v) for v in (yolo_label_candidates or [])],
                yolo_scores=[float(v) for v in (yolo_scores or [])],
                detection_score=float((yolo_scores or [quality_score])[0]) if (yolo_scores or [quality_score]) else 0.0,
                observation_kind="thing",
                view_quality=float(quality_score),
            )
            self._update_descriptor_views(obj, pseudo_observation, int(keyframe_id))
        obj.best_view_keyframes = self._update_best_views(obj.best_view_keyframes, int(keyframe_id))
        obj.observation_count += 1
        obj.promotion_evidence += max(float(quality_score) * 0.60, 0.05)
        obj.last_seen_stamp = max(float(obj.last_seen_stamp), float(stamp_sec))
        obj.dirty_flag = True
        return True

    def _create_object_id(self) -> str:
        object_id = f"obj_{self._next_object_id:05d}"
        self._next_object_id += 1
        return object_id

    def _ensure_object_id(
        self,
        object_id: str | None,
        centroid: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        keyframe_stamp_sec: float,
        keyframe_id: int,
        label_votes: dict[str, float],
        node_type: str = "thing",
        yolo_label_candidates: list[str] | None = None,
        yolo_scores: list[float] | None = None,
    ) -> tuple[str, bool]:
        if object_id is not None and object_id in self.objects:
            obj = self.objects[object_id]
            incoming_type = str(node_type or "thing").strip().lower()
            if incoming_type == "thing" and str(getattr(obj, "node_type", "thing")).strip().lower() != "thing":
                obj.node_type = "thing"
                obj.node_status = "tentative"
                obj.posterior = {}
                obj.unknown_score = max(float(getattr(obj, "unknown_score", 1.0)), 0.4)
            return object_id, False
        object_id = self._create_object_id()
        self.objects[object_id] = ObjectMemory(
            object_id=object_id,
            point_support_refs=[],
            centroid_world=[float(v) for v in centroid.tolist()],
            bbox_world=[float(v) for v in bbox_min.tolist() + bbox_max.tolist()],
            label_votes=dict(label_votes),
            embedding_path=None,
            best_view_keyframes=[int(keyframe_id)],
            observation_count=0,
            stability_score=1.0 if label_votes else 0.0,
            completeness_score=0.0,
            dirty_flag=True,
            last_seen_stamp=float(keyframe_stamp_sec),
            state="pending",
            pending_since_keyframe=int(keyframe_id),
            promotion_evidence=max(sum(label_votes.values()), 0.0),
            negative_evidence_score=0.0,
            descriptor_views=[],
            semantic_descriptor=None,
            semantic_descriptor_keyframe=None,
            semantic_descriptor_confidence=0.0,
            node_type=str(node_type or "thing"),
            node_status="tentative",
            posterior=dict(label_votes),
            unknown_score=1.0,
            reject_score=0.0,
            yolo_logit_sum={},
            descriptor_bank=[],
        )
        if yolo_label_candidates and yolo_scores:
            self._apply_yolo_evidence(
                self.objects[object_id],
                [str(v) for v in yolo_label_candidates],
                [float(v) for v in yolo_scores],
            )
        return object_id, True

    def birth_node(
        self,
        centroid: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        keyframe_stamp_sec: float,
        keyframe_id: int,
        label_votes: dict[str, float] | None = None,
        node_type: str = "thing",
    ) -> str:
        object_id, _ = self._ensure_object_id(
            object_id=None,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            keyframe_stamp_sec=keyframe_stamp_sec,
            keyframe_id=keyframe_id,
            label_votes=label_votes or {},
            node_type=node_type,
        )
        return object_id

    def merge_node(self, keep_id: str, drop_id: str, reason: str = "manual_merge") -> bool:
        if keep_id not in self.objects or drop_id not in self.objects or keep_id == drop_id:
            return False
        self._merge_object_into(
            keep_id,
            drop_id,
            reason,
            centroid_dist=0.0,
            overlap=0.0,
            support_overlap=0.0,
            best_view_similarity=0.0,
            temporal_continuity=0.0,
            descriptor_similarity=0.0,
        )
        return True

    def split_node(self, object_id: str, min_component_points: int = 12) -> list[str]:
        obj = self.objects.get(str(object_id))
        if obj is None:
            return []
        xyz, refs = self._collect_object_xyz_refs(obj.point_support_refs)
        if xyz.shape[0] < max(int(min_component_points) * 2, 4):
            return []
        components = self._connected_components_from_xyz(xyz, min_component_points=max(int(min_component_points), 2))
        if len(components) <= 1:
            return []
        components.sort(key=len, reverse=True)
        keep_component = set(int(idx) for idx in components[0])
        created_ids: list[str] = []
        affected_object_ids = {str(object_id)}
        for component in components[1:]:
            component = [int(idx) for idx in component]
            component_refs = [refs[idx] for idx in component]
            component_xyz = xyz[np.asarray(component, dtype=np.int32)]
            bbox_min = component_xyz.min(axis=0).astype(np.float32)
            bbox_max = component_xyz.max(axis=0).astype(np.float32)
            centroid = component_xyz.mean(axis=0).astype(np.float32)
            new_object_id = self.birth_node(
                centroid=centroid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                keyframe_stamp_sec=float(obj.last_seen_stamp),
                keyframe_id=int(obj.best_view_keyframes[0]) if obj.best_view_keyframes else -1,
                label_votes=dict(obj.label_votes),
                node_type=str(getattr(obj, "node_type", "thing")),
            )
            new_obj = self.objects[new_object_id]
            new_obj.state = str(obj.state)
            new_obj.node_status = "tentative"
            new_obj.promotion_evidence = float(obj.promotion_evidence) * 0.5
            new_obj.negative_evidence_score = float(obj.negative_evidence_score)
            new_obj.node_type = str(getattr(obj, "node_type", "thing"))
            new_obj.posterior = {}
            new_obj.yolo_logit_sum = {}
            new_obj.descriptor_views = []
            new_obj.descriptor_bank = []
            new_obj.semantic_descriptor = None
            new_obj.semantic_descriptor_keyframe = None
            new_obj.semantic_descriptor_confidence = 0.0
            new_obj.merge_reason = "split_child"
            created_ids.append(str(new_object_id))
            affected_object_ids.add(new_object_id)
            for local_cloud_id, point_index in component_refs:
                state = self.point_assignments.get(local_cloud_id, {}).get(int(point_index))
                if state is None:
                    continue
                state["current_object_id"] = str(new_object_id)
                state["current_score"] = max(float(state.get("current_score", 0.0)), self.point_score_increment)
                state["object_scores"] = {str(new_object_id): float(state["current_score"])}
        for idx, (local_cloud_id, point_index) in enumerate(refs):
            if idx in keep_component:
                continue
            state = self.point_assignments.get(local_cloud_id, {}).get(int(point_index))
            if state is None:
                continue
            if state.get("current_object_id") == str(object_id):
                continue
            if str(object_id) in dict(state.get("object_scores", {})):
                scores = dict(state.get("object_scores", {}))
                scores.pop(str(object_id), None)
                state["object_scores"] = scores

        self._refresh_all_supports()
        self._recompute_objects(affected_object_ids)
        self._update_object_states(float(obj.last_seen_stamp))
        return created_ids

    def update_posterior(self, rank_descriptor_fn, structural_labels: set[str] | None = None) -> None:
        self.refresh_node_semantics(rank_descriptor_fn, structural_labels)

    def reassign_points_from_posterior(self, unknown_threshold: float = 0.95) -> int:
        changed = 0
        confident_object_ids = {
            object_id
            for object_id, obj in self.objects.items()
            if float(getattr(obj, "unknown_score", 1.0)) < float(unknown_threshold)
            and float(getattr(obj, "reject_score", 0.0)) < 0.85
        }
        self._rebuild_global_voxel_index()
        for local_states in self.point_assignments.values():
            for state in local_states.values():
                object_id = state.get("current_object_id")
                should_clear = False
                if object_id:
                    obj = self.objects.get(str(object_id))
                    if obj is not None and float(getattr(obj, "unknown_score", 1.0)) >= float(unknown_threshold):
                        should_clear = True
                if should_clear and state.get("current_object_id") is not None:
                    state["current_object_id"] = None
                    state["current_score"] = 0.0
                    changed += 1
        for local_cloud_id, local_states in self.point_assignments.items():
            xyz = self._load_xyz_by_id(local_cloud_id)
            if xyz is None:
                continue
            for point_index, state in local_states.items():
                current_object_id = state.get("current_object_id")
                if current_object_id and str(current_object_id) in confident_object_ids:
                    continue
                ref_xyz = xyz[int(point_index)]
                votes: dict[str, float] = {}
                for voxel_key in self._neighbor_voxel_keys(self._voxel_key(ref_xyz)):
                    for neighbor_cloud_id, neighbor_point_idx in self.global_voxel_index.get(voxel_key, []):
                        neighbor_state = self.point_assignments.get(neighbor_cloud_id, {}).get(int(neighbor_point_idx))
                        if neighbor_state is None:
                            continue
                        neighbor_object_id = neighbor_state.get("current_object_id")
                        if not neighbor_object_id or str(neighbor_object_id) not in confident_object_ids:
                            continue
                        neighbor_xyz = self._point_xyz(neighbor_cloud_id, int(neighbor_point_idx))
                        if neighbor_xyz is None:
                            continue
                        dist = float(np.linalg.norm(neighbor_xyz - ref_xyz))
                        if dist > self.cross_cloud_reassign_radius_m:
                            continue
                        weight = 1.0 / max(dist, 0.03)
                        votes[str(neighbor_object_id)] = votes.get(str(neighbor_object_id), 0.0) + weight
                if not votes:
                    continue
                new_object_id, score = max(votes.items(), key=lambda item: item[1])
                if state.get("current_object_id") != new_object_id:
                    state["current_object_id"] = str(new_object_id)
                    state["current_score"] = float(score)
                    state["object_scores"] = {str(new_object_id): float(score)}
                    changed += 1
        self._refresh_all_supports()
        self._recompute_objects(set(self.objects.keys()))
        return changed

    def _votes_from_observation(self, observation: ObservationLink) -> dict[str, float]:
        if observation.semantic_label_candidates and observation.semantic_scores:
            return {
                str(label): float(score)
                for label, score in zip(observation.semantic_label_candidates, observation.semantic_scores)
            }
        if observation.semantic_label_candidates:
            weight = 1.0 / max(len(observation.semantic_label_candidates), 1)
            return {str(label): weight for label in observation.semantic_label_candidates}
        return {}

    def _node_type_from_observation(self, observation: ObservationLink) -> str:
        kind = str(getattr(observation, "observation_kind", "thing")).strip().lower()
        if kind in {"thing", "stuff"}:
            return kind
        if observation.semantic_label_candidates:
            top_label = str(observation.semantic_label_candidates[0])
            if top_label in STRUCTURAL_LABELS:
                return "stuff"
        return "thing"

    def _load_xyz(self, local_cloud: LocalCloudPacket) -> np.ndarray:
        cached = self.local_cloud_xyz_cache.get(local_cloud.local_cloud_id)
        if cached is not None:
            return cached
        data = np.load(Path(local_cloud.cloud_path))
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        self.local_cloud_xyz_cache[local_cloud.local_cloud_id] = xyz
        return xyz

    def _running_average(self, current: list[float], new_value: list[float], count: int) -> list[float]:
        if count <= 1:
            return [float(v) for v in new_value]
        alpha = 1.0 / float(count)
        return [float((1.0 - alpha) * c + alpha * n) for c, n in zip(current, new_value)]

    def _merge_bbox(self, current_bbox: list[float], bbox_min: np.ndarray, bbox_max: np.ndarray) -> list[float]:
        current_min = np.asarray(current_bbox[:3], dtype=np.float32)
        current_max = np.asarray(current_bbox[3:], dtype=np.float32)
        merged_min = np.minimum(current_min, bbox_min)
        merged_max = np.maximum(current_max, bbox_max)
        return [float(v) for v in merged_min.tolist() + merged_max.tolist()]

    def _update_best_views(self, existing: list[int], keyframe_id: int) -> list[int]:
        out = [int(v) for v in existing if int(v) != int(keyframe_id)]
        out.insert(0, int(keyframe_id))
        return out[: self.max_best_views]

    def _update_descriptor_views(self, obj: ObjectMemory, observation: ObservationLink, keyframe_id: int) -> None:
        if not observation.semantic_embedding:
            return
        embedding = np.asarray(observation.semantic_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-6:
            return
        embedding = embedding / norm
        quality = float(observation.quality_score)
        record = {
            "keyframe_id": int(keyframe_id),
            "quality_score": quality,
            "embedding": [float(v) for v in embedding.tolist()],
            "semantic_label_candidates": [str(v) for v in observation.semantic_label_candidates[: self.max_best_views]],
            "semantic_scores": [float(v) for v in observation.semantic_scores[: self.max_best_views]],
            "source": str(observation.semantic_embedding_variant or "descriptor"),
            "bbox_xyxy": [int(v) for v in observation.bbox_xyxy],
            "view_quality": float(observation.view_quality),
            "observation_kind": str(observation.observation_kind),
            "yolo_label_candidates": [str(v) for v in observation.yolo_label_candidates[: self.max_best_views]],
            "yolo_scores": [float(v) for v in observation.yolo_scores[: self.max_best_views]],
        }
        kept: list[dict[str, object]] = []
        replaced = False
        for existing in obj.descriptor_views:
            if int(existing.get("keyframe_id", -1)) == int(keyframe_id):
                if float(existing.get("quality_score", 0.0)) <= quality:
                    kept.append(record)
                else:
                    kept.append(existing)
                replaced = True
            else:
                kept.append(existing)
        if not replaced:
            kept.append(record)
        kept.sort(key=lambda item: float(item.get("quality_score", 0.0)), reverse=True)
        obj.descriptor_views = kept[: self.max_best_views]
        obj.descriptor_bank = [dict(item) for item in obj.descriptor_views]
        self._recompute_semantic_descriptor(obj)

    def _recompute_semantic_descriptor(self, obj: ObjectMemory) -> None:
        if not obj.descriptor_views:
            obj.semantic_descriptor = None
            obj.semantic_descriptor_keyframe = None
            obj.semantic_descriptor_confidence = 0.0
            return
        embeddings: list[np.ndarray] = []
        keyframe_ids: list[int] = []
        qualities: list[float] = []
        for item in obj.descriptor_views:
            vec = np.asarray(item.get("embedding", []), dtype=np.float32)
            if vec.size == 0:
                continue
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-6:
                continue
            embeddings.append(vec / norm)
            keyframe_ids.append(int(item.get("keyframe_id", -1)))
            qualities.append(float(item.get("quality_score", 0.0)))
        if not embeddings:
            obj.semantic_descriptor = None
            obj.semantic_descriptor_keyframe = None
            obj.semantic_descriptor_confidence = 0.0
            return
        embed_mat = np.vstack(embeddings).astype(np.float32, copy=False)
        if embed_mat.shape[0] == 1:
            descriptor = embed_mat[0]
            selected_idx = 0
        else:
            l1_distances = np.abs(embed_mat[:, None, :] - embed_mat[None, :, :]).sum(axis=(1, 2))
            selected_idx = int(np.argmin(l1_distances))
            descriptor = embed_mat[selected_idx]
        obj.semantic_descriptor = [float(v) for v in descriptor.tolist()]
        obj.semantic_descriptor_keyframe = int(keyframe_ids[selected_idx]) if selected_idx < len(keyframe_ids) else None
        obj.semantic_descriptor_confidence = float(max(qualities[selected_idx], 0.0)) if selected_idx < len(qualities) else 0.0

    def _compute_stability(self, obj: ObjectMemory) -> float:
        if not obj.label_votes:
            return 0.0
        total = float(sum(obj.label_votes.values()))
        best = float(max(obj.label_votes.values()))
        if total <= 1e-6:
            return 0.0
        return best / total

    def _effective_min_points_per_observation(self, mask_area: int, bbox_xyxy: list[int]) -> int:
        bbox_w = max(int(bbox_xyxy[2]) - int(bbox_xyxy[0]), 0)
        bbox_h = max(int(bbox_xyxy[3]) - int(bbox_xyxy[1]), 0)
        min_side = min(bbox_w, bbox_h)
        if mask_area < max(self.min_mask_area * 3, 256) or min_side < 18:
            return max(2, min(self.min_points_per_observation, 2))
        if mask_area < max(self.min_mask_area * 8, 768) or min_side < 36:
            return max(3, min(self.min_points_per_observation, 3))
        return self.min_points_per_observation

    def _labels_compatible(self, observation: ObservationLink, obj: ObjectMemory) -> bool:
        if str(getattr(observation, "observation_kind", "thing")).strip().lower() != str(
            getattr(obj, "node_type", "thing")
        ).strip().lower():
            return False

        obs_primary = self._observation_primary_nonstruct_label(observation)
        obj_primary = self._object_primary_nonstruct_label(obj)
        if obs_primary and obj_primary and obs_primary != obj_primary and obj.observation_count >= 1:
            return False

        descriptor_similarity = self._observation_object_descriptor_similarity(observation, obj)
        if descriptor_similarity is not None:
            if descriptor_similarity >= self.observation_descriptor_match_min:
                return True
            if obj.state == "active" and descriptor_similarity <= self.observation_descriptor_conflict_max:
                return False
        if not observation.semantic_label_candidates or not obj.label_votes:
            return True
        if obj.state == "pending":
            return True
        object_top_label = max(obj.label_votes.items(), key=lambda item: item[1])[0]
        observation_top_labels = {str(label) for label in observation.semantic_label_candidates[:2]}
        if object_top_label in observation_top_labels:
            return True
        if self._compute_stability(obj) < 0.55:
            return True
        return False

    def _observation_primary_nonstruct_label(self, observation: ObservationLink) -> str | None:
        for label in [*observation.yolo_label_candidates, *observation.semantic_label_candidates]:
            value = str(label).strip()
            if value and value not in STRUCTURAL_LABELS:
                return value
        return None

    def _object_primary_nonstruct_label(self, obj: ObjectMemory) -> str | None:
        yolo_scores = self._normalize_score_dict(getattr(obj, "yolo_logit_sum", {}))
        yolo_scores = {
            str(label): float(score)
            for label, score in yolo_scores.items()
            if str(label) not in STRUCTURAL_LABELS and float(score) > 1e-6
        }
        if yolo_scores:
            return max(yolo_scores.items(), key=lambda item: item[1])[0]
        label_votes = {
            str(label): float(score)
            for label, score in dict(getattr(obj, "label_votes", {})).items()
            if str(label) not in STRUCTURAL_LABELS and float(score) > 1e-6
        }
        if label_votes:
            return max(label_votes.items(), key=lambda item: item[1])[0]
        posterior = {
            str(label): float(score)
            for label, score in dict(getattr(obj, "posterior", {})).items()
            if str(label) not in STRUCTURAL_LABELS and float(score) > 1e-6
        }
        if posterior:
            return max(posterior.items(), key=lambda item: item[1])[0]
        return None

    def _observation_object_descriptor_similarity(
        self,
        observation: ObservationLink,
        obj: ObjectMemory,
    ) -> float | None:
        if not observation.semantic_embedding or not obj.semantic_descriptor:
            return None
        obs_vec = self._normalized_vector(observation.semantic_embedding)
        obj_vec = self._normalized_vector(obj.semantic_descriptor)
        if obs_vec is None or obj_vec is None:
            return None
        return float(np.clip(np.dot(obs_vec, obj_vec), -1.0, 1.0))

    def _object_descriptor_similarity(self, obj_a: ObjectMemory, obj_b: ObjectMemory) -> float | None:
        if not obj_a.semantic_descriptor or not obj_b.semantic_descriptor:
            return None
        vec_a = self._normalized_vector(obj_a.semantic_descriptor)
        vec_b = self._normalized_vector(obj_b.semantic_descriptor)
        if vec_a is None or vec_b is None:
            return None
        return float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))

    def _normalized_vector(self, values: list[float] | np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        vec = np.asarray(values, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return None
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return None
        return vec / norm

    def _top_label(self, obj: ObjectMemory) -> str:
        if not obj.label_votes:
            return "unknown"
        return max(obj.label_votes.items(), key=lambda item: item[1])[0]

    def _normalize_score_dict(self, scores: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for label, score in scores.items():
            value = float(score)
            if not np.isfinite(value) or value <= 1e-6:
                continue
            normalized[str(label)] = value
        if not normalized:
            return {}
        total = float(sum(normalized.values()))
        if total <= 1e-6:
            return {}
        return {label: float(value / total) for label, value in normalized.items()}

    def _filter_scores_for_node_type(
        self,
        scores: dict[str, float],
        node_type: str,
        structural_labels: set[str],
    ) -> dict[str, float]:
        node_kind = str(node_type or "thing").strip().lower()
        if node_kind == "thing":
            return {
                str(label): float(score)
                for label, score in scores.items()
                if str(label) not in structural_labels and float(score) > 1e-6
            }
        if node_kind == "stuff":
            return {
                str(label): float(score)
                for label, score in scores.items()
                if str(label) in structural_labels and float(score) > 1e-6
            }
        return {
            str(label): float(score)
            for label, score in scores.items()
            if float(score) > 1e-6
        }

    def _merge_bounded_score_maps(
        self,
        keep_scores: dict[str, float] | None,
        drop_scores: dict[str, float] | None,
    ) -> dict[str, float]:
        merged: dict[str, float] = {}
        for label, score in dict(keep_scores or {}).items():
            value = float(score)
            if np.isfinite(value) and value > 1e-6:
                merged[str(label)] = value
        for label, score in dict(drop_scores or {}).items():
            value = float(score)
            if not np.isfinite(value) or value <= 1e-6:
                continue
            merged[str(label)] = max(float(merged.get(str(label), 0.0)), value)
        return self._normalize_score_dict(merged)

    def refresh_node_semantics(
        self,
        rank_descriptor_fn,
        structural_labels: set[str] | None = None,
    ) -> None:
        structural = structural_labels or STRUCTURAL_LABELS
        for obj in self.objects.values():
            stable_node_type = str(getattr(obj, "node_type", "thing") or "thing").strip().lower()
            if stable_node_type not in {"thing", "stuff"}:
                stable_node_type = "thing"
            raw_scores: dict[str, float] = {}

            label_scores = self._filter_scores_for_node_type(dict(obj.label_votes), stable_node_type, structural)
            for label, score in label_scores.items():
                raw_scores[str(label)] = raw_scores.get(str(label), 0.0) + float(score) * self.posterior_vote_weight

            yolo_scores = self._filter_scores_for_node_type(
                self._normalize_score_dict(getattr(obj, "yolo_logit_sum", {})),
                stable_node_type,
                structural,
            )
            for label, score in yolo_scores.items():
                raw_scores[str(label)] = raw_scores.get(str(label), 0.0) + float(score) * self.posterior_yolo_weight

            if obj.semantic_descriptor:
                descriptor_scores = {
                    str(label): float(score) for label, score in rank_descriptor_fn(obj.semantic_descriptor, topk=5)
                }
                descriptor_scores = self._filter_scores_for_node_type(descriptor_scores, stable_node_type, structural)
                for label, score in descriptor_scores.items():
                    raw_scores[str(label)] = raw_scores.get(str(label), 0.0) + float(score) * self.posterior_descriptor_weight
            if raw_scores:
                clipped_scores = {label: max(float(score), 1e-6) for label, score in raw_scores.items()}
                total = float(sum(clipped_scores.values()))
                posterior = {label: float(score) / max(total, 1e-6) for label, score in clipped_scores.items()}
                ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
                best_prob = float(ranked[0][1]) if ranked else 0.0
                second_prob = float(ranked[1][1]) if len(ranked) > 1 else 0.0
                support_count = sum(len(indices) for _cloud_id, indices in obj.point_support_refs)
                unknown_score = max(0.0, 1.0 - min(best_prob * 1.6, 1.0))
                if obj.observation_count < 2 or support_count < self.pending_seed_points:
                    unknown_score = max(unknown_score, 0.45)
                reject_score = float(np.clip(1.0 - max(best_prob - second_prob, 0.0) * 4.0, 0.0, 1.0))
                obj.posterior = posterior
                obj.unknown_score = float(np.clip(unknown_score, 0.0, 1.0))
                obj.reject_score = float(np.clip(reject_score, 0.0, 1.0))
                obj.node_type = stable_node_type
            else:
                obj.posterior = {}
                obj.unknown_score = 1.0
                obj.reject_score = 0.0
                obj.node_type = stable_node_type
            obj.node_status = self._infer_node_status(obj)

    def _infer_node_status(self, obj: ObjectMemory) -> str:
        posterior = dict(getattr(obj, "posterior", {}))
        ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
        best_prob = float(ranked[0][1]) if ranked else 0.0
        second_prob = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        support_count = sum(len(indices) for _local_cloud_id, indices in obj.point_support_refs)
        if float(getattr(obj, "reject_score", 0.0)) >= 0.65:
            return "reject"
        if float(getattr(obj, "unknown_score", 1.0)) >= 0.75 and best_prob < 0.45:
            return "unknown"
        if obj.observation_count < max(self.pending_commit_observations, 2):
            return "tentative"
        if support_count < max(self.pending_seed_points, 4):
            return "tentative"
        if best_prob < 0.55 or (best_prob - second_prob) < 0.10:
            return "tentative"
        return "confirmed"

    def _normalize_visible_points(self, visible_point_indices: list[int] | np.ndarray | None, point_count: int) -> set[int]:
        if visible_point_indices is None:
            return set()
        indices = np.asarray(visible_point_indices, dtype=np.int32).reshape(-1)
        indices = indices[(indices >= 0) & (indices < point_count)]
        return {int(v) for v in indices.tolist()}

    def _get_or_create_point_state(
        self, local_states: dict[int, dict[str, object]], point_index: int, stamp_sec: float
    ) -> dict[str, object]:
        state = local_states.get(point_index)
        if state is None:
            state = {
                "current_object_id": None,
                "current_score": 0.0,
                "object_scores": {},
                "last_seen_stamp": float(stamp_sec),
            }
            local_states[point_index] = state
        return state

    def _apply_positive_evidence(self, state: dict[str, object], object_id: str, weight: float) -> None:
        scores = dict(state.get("object_scores", {}))
        scores[object_id] = float(scores.get(object_id, 0.0)) + float(weight)
        for other_id in list(scores.keys()):
            if other_id == object_id:
                continue
            scores[other_id] = max(0.0, float(scores.get(other_id, 0.0)) - (self.point_score_decrement * 0.5))
        state["object_scores"] = scores

    def _apply_negative_evidence(self, state: dict[str, object], object_id: str, weight: float) -> None:
        scores = dict(state.get("object_scores", {}))
        if object_id not in scores:
            return
        scores[object_id] = max(0.0, float(scores[object_id]) - float(weight))
        state["object_scores"] = scores

    def _reconcile_point_assignment(self, state: dict[str, object]) -> str | None:
        scores = {
            str(object_id): float(score)
            for object_id, score in dict(state.get("object_scores", {})).items()
            if float(score) > 1e-6
        }
        if not scores:
            state["current_object_id"] = None
            state["current_score"] = 0.0
            state["object_scores"] = {}
            return None

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_object_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        current_object_id = state.get("current_object_id")
        current_score = float(scores.get(str(current_object_id), 0.0)) if current_object_id else 0.0

        if best_score < self.point_keep_threshold:
            state["current_object_id"] = None
            state["current_score"] = 0.0
            state["object_scores"] = scores
            return None

        if current_object_id and str(current_object_id) in scores:
            if best_object_id != current_object_id and (best_score - current_score) < self.point_reassign_margin:
                state["current_object_id"] = str(current_object_id)
                state["current_score"] = current_score
                state["object_scores"] = scores
                return str(current_object_id)

        if (best_score - second_score) < self.point_reassign_margin and current_object_id:
            state["current_object_id"] = str(current_object_id)
            state["current_score"] = current_score
            state["object_scores"] = scores
            return str(current_object_id)

        state["current_object_id"] = best_object_id
        state["current_score"] = best_score
        state["object_scores"] = scores
        return best_object_id

    def _apply_label_evidence(self, obj: ObjectMemory, label_votes: dict[str, float]) -> None:
        previous_top_label = None
        if obj.label_votes:
            previous_top_label = max(obj.label_votes.items(), key=lambda item: item[1])[0]
        if obj.label_votes:
            for label in list(obj.label_votes.keys()):
                obj.label_votes[label] = float(obj.label_votes[label]) * self.label_vote_decay
                if obj.label_votes[label] <= 1e-6:
                    obj.label_votes.pop(label, None)
        for label, score in label_votes.items():
            obj.label_votes[label] = obj.label_votes.get(label, 0.0) + float(score)
        if label_votes:
            current_top_label = max(label_votes.items(), key=lambda item: item[1])[0]
            if previous_top_label and current_top_label != previous_top_label:
                obj.negative_evidence_score += 1.0
            else:
                obj.negative_evidence_score *= 0.5
        obj.stability_score = self._compute_stability(obj)

    def _apply_yolo_evidence(self, obj: ObjectMemory, labels: list[str], scores: list[float]) -> None:
        ema_decay = float(np.clip(self.label_vote_decay, 0.0, 0.99))
        current_scores = self._normalize_score_dict(getattr(obj, "yolo_logit_sum", {}))
        if current_scores:
            for label in list(current_scores.keys()):
                current_scores[label] = float(current_scores[label]) * ema_decay
                if current_scores[label] <= 1e-6:
                    current_scores.pop(label, None)

        incoming_raw: dict[str, float] = {}
        for label, score in zip(labels, scores, strict=False):
            value = float(score)
            if not np.isfinite(value) or value <= 1e-6:
                continue
            incoming_raw[str(label)] = max(float(incoming_raw.get(str(label), 0.0)), value)
        incoming_scores = self._normalize_score_dict(incoming_raw)
        for label, score in incoming_scores.items():
            current_scores[label] = current_scores.get(label, 0.0) + (1.0 - ema_decay) * float(score)
        obj.yolo_logit_sum = self._normalize_score_dict(current_scores)

    def _refresh_supports_for_local_cloud(self, local_cloud_id: str) -> None:
        grouped_indices: dict[str, list[int]] = {}
        local_states = self.point_assignments.get(local_cloud_id, {})
        for point_index, state in local_states.items():
            object_id = state.get("current_object_id")
            current_score = float(state.get("current_score", 0.0))
            if not object_id or current_score < self.point_keep_threshold:
                continue
            grouped_indices.setdefault(str(object_id), []).append(int(point_index))

        for obj in self.objects.values():
            obj.point_support_refs = [
                (support_local_cloud_id, point_indices)
                for support_local_cloud_id, point_indices in obj.point_support_refs
                if support_local_cloud_id != local_cloud_id
            ]

        for object_id, point_indices in grouped_indices.items():
            if object_id not in self.objects:
                continue
            self.objects[object_id].point_support_refs.append(
                (local_cloud_id, sorted(set(int(v) for v in point_indices)))
            )

    def _refresh_all_supports(self) -> None:
        for obj in self.objects.values():
            obj.point_support_refs = []
        for local_cloud_id in list(self.point_assignments.keys()):
            self._refresh_supports_for_local_cloud(local_cloud_id)

    def _rebuild_global_voxel_index(self) -> None:
        voxel_index: dict[tuple[int, int, int], list[tuple[str, int]]] = {}
        for local_cloud_id, local_states in self.point_assignments.items():
            cloud_path = self._cloud_path_for_local_cloud(local_cloud_id)
            if cloud_path is None or not cloud_path.exists():
                continue
            xyz = self._load_xyz_by_id(local_cloud_id)
            if xyz is None or xyz.size == 0:
                continue
            for point_index, state in local_states.items():
                object_id = state.get("current_object_id")
                current_score = float(state.get("current_score", 0.0))
                if not object_id or current_score < self.point_keep_threshold:
                    continue
                if point_index < 0 or point_index >= xyz.shape[0]:
                    continue
                voxel_key = self._voxel_key(xyz[point_index])
                voxel_index.setdefault(voxel_key, []).append((local_cloud_id, int(point_index)))
        self.global_voxel_index = voxel_index

    def _propagate_to_historical_neighbors(
        self,
        target_object_id: str,
        source_local_cloud_id: str,
        source_xyz: np.ndarray,
        observed_point_indices: list[int],
        stamp_sec: float,
        affected_object_ids: set[str],
    ) -> None:
        visited_refs: set[tuple[str, int]] = set()
        for point_index in observed_point_indices:
            if point_index < 0 or point_index >= source_xyz.shape[0]:
                continue
            query_xyz = source_xyz[point_index]
            matched = 0
            for neighbor_voxel in self._neighbor_voxel_keys(self._voxel_key(query_xyz)):
                for ref_local_cloud_id, ref_point_index in self.global_voxel_index.get(neighbor_voxel, []):
                    ref_key = (ref_local_cloud_id, int(ref_point_index))
                    if ref_key in visited_refs:
                        continue
                    if ref_local_cloud_id == source_local_cloud_id and int(ref_point_index) == int(point_index):
                        continue
                    ref_xyz = self._point_xyz(ref_local_cloud_id, int(ref_point_index))
                    if ref_xyz is None:
                        continue
                    if float(np.linalg.norm(ref_xyz - query_xyz)) > self.cross_cloud_reassign_radius_m:
                        continue
                    state = self.point_assignments.get(ref_local_cloud_id, {}).get(int(ref_point_index))
                    if state is None:
                        continue
                    previous_object_id = state.get("current_object_id")
                    self._apply_positive_evidence(state, target_object_id, self.cross_cloud_positive_weight)
                    if previous_object_id and previous_object_id != target_object_id:
                        self._apply_negative_evidence(
                            state,
                            str(previous_object_id),
                            self.cross_cloud_negative_weight,
                        )
                    new_object_id = self._reconcile_point_assignment(state)
                    state["last_seen_stamp"] = float(stamp_sec)
                    if previous_object_id:
                        affected_object_ids.add(str(previous_object_id))
                    if new_object_id:
                        affected_object_ids.add(str(new_object_id))
                    visited_refs.add(ref_key)
                    matched += 1
                    if matched >= self.max_cross_cloud_refs_per_point:
                        break
                if matched >= self.max_cross_cloud_refs_per_point:
                    break

    def _collect_object_xyz(self, point_support_refs: list[tuple[str, list[int]]]) -> np.ndarray:
        xyz, _ = self._collect_object_xyz_refs(point_support_refs)
        return xyz

    def _collect_object_xyz_refs(
        self,
        point_support_refs: list[tuple[str, list[int]]],
    ) -> tuple[np.ndarray, list[tuple[str, int]]]:
        xyz_parts: list[np.ndarray] = []
        refs: list[tuple[str, int]] = []
        for local_cloud_id, point_indices in point_support_refs:
            if not point_indices:
                continue
            cloud_path = self._cloud_path_for_local_cloud(local_cloud_id)
            if cloud_path is None or not cloud_path.exists():
                continue
            data = np.load(cloud_path)
            xyz = np.asarray(data["xyz"], dtype=np.float32)
            valid = np.asarray(sorted(set(int(v) for v in point_indices)), dtype=np.int32)
            valid = valid[(valid >= 0) & (valid < xyz.shape[0])]
            if valid.size == 0:
                continue
            xyz_parts.append(xyz[valid])
            refs.extend((str(local_cloud_id), int(point_index)) for point_index in valid.tolist())
        if not xyz_parts:
            return np.zeros((0, 3), dtype=np.float32), []
        return np.vstack(xyz_parts).astype(np.float32, copy=False), refs

    def _connected_components_from_xyz(self, xyz: np.ndarray, min_component_points: int) -> list[list[int]]:
        if xyz.size == 0:
            return []
        voxel = np.floor(xyz / self.spatial_voxel_size_m).astype(np.int32)
        unique_voxel, inverse = np.unique(voxel, axis=0, return_inverse=True)
        if unique_voxel.shape[0] <= 1:
            return [list(range(xyz.shape[0]))]
        voxel_to_points: dict[int, list[int]] = {}
        for point_idx, voxel_idx in enumerate(inverse.tolist()):
            voxel_to_points.setdefault(int(voxel_idx), []).append(int(point_idx))
        coord_to_index = {tuple(coord.tolist()): int(idx) for idx, coord in enumerate(unique_voxel)}
        components: list[list[int]] = []
        visited: set[int] = set()
        for voxel_idx in range(unique_voxel.shape[0]):
            if voxel_idx in visited:
                continue
            queue = [int(voxel_idx)]
            visited.add(int(voxel_idx))
            component_points: list[int] = []
            while queue:
                current = queue.pop()
                component_points.extend(voxel_to_points.get(current, []))
                base = unique_voxel[current]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            neighbor = coord_to_index.get((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)))
                            if neighbor is None or neighbor in visited:
                                continue
                            visited.add(neighbor)
                            queue.append(neighbor)
            component_points = sorted(set(component_points))
            if len(component_points) >= int(min_component_points):
                components.append(component_points)
        return components

    def auto_split_nodes(
        self,
        min_component_points: int = 12,
        min_extent_m: float = 1.8,
        reject_threshold: float = 0.85,
    ) -> list[str]:
        created_ids: list[str] = []
        for object_id, obj in list(self.objects.items()):
            if obj.state == "stale":
                continue
            bbox = np.asarray(obj.bbox_world, dtype=np.float32)
            extent = bbox[3:] - bbox[:3]
            largest_extent = float(np.max(extent)) if extent.size == 3 else 0.0
            should_try = largest_extent >= float(min_extent_m) or float(getattr(obj, "reject_score", 0.0)) >= float(reject_threshold)
            if not should_try:
                continue
            created_ids.extend(self.split_node(str(object_id), min_component_points=min_component_points))
        return created_ids

    def _recompute_objects(self, object_ids: set[str]) -> None:
        for object_id in sorted(object_ids):
            obj = self.objects.get(object_id)
            if obj is None:
                continue
            xyz = self._collect_object_xyz(obj.point_support_refs)
            if xyz.size == 0:
                obj.completeness_score = 0.0
                continue
            centroid = xyz.mean(axis=0).astype(np.float32)
            bbox_min = xyz.min(axis=0).astype(np.float32)
            bbox_max = xyz.max(axis=0).astype(np.float32)
            obj.centroid_world = [float(v) for v in centroid.tolist()]
            obj.bbox_world = [float(v) for v in bbox_min.tolist() + bbox_max.tolist()]
            obj.completeness_score = min(1.0, float(xyz.shape[0]) / 200.0)
            obj.stability_score = self._compute_stability(obj)
            obj.label_compatibility_score = obj.stability_score
            obj.dirty_flag = True

    def _bbox_overlap_score(self, bbox_a: list[float], bbox_b: list[float]) -> float:
        a_min = np.asarray(bbox_a[:3], dtype=np.float32)
        a_max = np.asarray(bbox_a[3:], dtype=np.float32)
        b_min = np.asarray(bbox_b[:3], dtype=np.float32)
        b_max = np.asarray(bbox_b[3:], dtype=np.float32)
        inter_min = np.maximum(a_min, b_min)
        inter_max = np.minimum(a_max, b_max)
        inter_size = np.maximum(inter_max - inter_min, 0.0)
        inter_vol = float(np.prod(inter_size))
        if inter_vol <= 0.0:
            return 0.0
        vol_a = float(np.prod(np.maximum(a_max - a_min, 1e-4)))
        vol_b = float(np.prod(np.maximum(b_max - b_min, 1e-4)))
        return inter_vol / max(min(vol_a, vol_b), 1e-4)

    def _support_overlap_score(self, obj_a: ObjectMemory, obj_b: ObjectMemory) -> float:
        supports_a: dict[str, set[int]] = {
            str(local_cloud_id): {int(v) for v in point_indices}
            for local_cloud_id, point_indices in obj_a.point_support_refs
        }
        supports_b: dict[str, set[int]] = {
            str(local_cloud_id): {int(v) for v in point_indices}
            for local_cloud_id, point_indices in obj_b.point_support_refs
        }
        shared_clouds = set(supports_a.keys()) & set(supports_b.keys())
        if not shared_clouds:
            return 0.0
        overlap = 0
        denom = 0
        for local_cloud_id in shared_clouds:
            pts_a = supports_a[local_cloud_id]
            pts_b = supports_b[local_cloud_id]
            if not pts_a or not pts_b:
                continue
            overlap += len(pts_a & pts_b)
            denom += max(min(len(pts_a), len(pts_b)), 1)
        if denom <= 0:
            return 0.0
        return float(overlap) / float(denom)

    def _best_view_similarity(self, obj_a: ObjectMemory, obj_b: ObjectMemory) -> float:
        views_a = {int(v) for v in obj_a.best_view_keyframes}
        views_b = {int(v) for v in obj_b.best_view_keyframes}
        if not views_a or not views_b:
            return 0.0
        inter = len(views_a & views_b)
        union = len(views_a | views_b)
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    def _temporal_continuity_score(self, obj_a: ObjectMemory, obj_b: ObjectMemory) -> float:
        delta = abs(float(obj_a.last_seen_stamp) - float(obj_b.last_seen_stamp))
        if delta <= 0.5:
            return 1.0
        if delta >= 10.0:
            return 0.0
        return float(max(0.0, 1.0 - (delta / 10.0)))

    def _merge_compatible_objects(self) -> None:
        object_ids = sorted(self.objects.keys())
        merged_pairs: list[tuple[str, str, str, float, float, float, float, float, float]] = []
        for idx, object_id_a in enumerate(object_ids):
            obj_a = self.objects.get(object_id_a)
            if obj_a is None or obj_a.state != "active":
                continue
            label_a = self._top_label(obj_a)
            for object_id_b in object_ids[idx + 1 :]:
                obj_b = self.objects.get(object_id_b)
                if obj_b is None or obj_b.state != "active":
                    continue
                label_b = self._top_label(obj_b)
                centroid_dist = float(
                    np.linalg.norm(
                        np.asarray(obj_a.centroid_world, dtype=np.float32)
                        - np.asarray(obj_b.centroid_world, dtype=np.float32)
                    )
                )
                overlap = self._bbox_overlap_score(obj_a.bbox_world, obj_b.bbox_world)
                support_overlap = self._support_overlap_score(obj_a, obj_b)
                best_view_similarity = self._best_view_similarity(obj_a, obj_b)
                temporal_continuity = self._temporal_continuity_score(obj_a, obj_b)
                descriptor_similarity = self._object_descriptor_similarity(obj_a, obj_b)
                same_label = label_a == label_b and label_a != "unknown"
                if not same_label and (
                    descriptor_similarity is None or descriptor_similarity < self.merge_descriptor_similarity_min
                ):
                    continue
                if descriptor_similarity is not None and descriptor_similarity <= self.merge_descriptor_conflict_max:
                    continue
                max_dist = (
                    self.merge_centroid_radius_m if label_a in STRUCTURAL_LABELS else self.merge_compact_centroid_radius_m
                )
                allow_merge = False
                if centroid_dist <= max_dist and temporal_continuity >= 0.40:
                    allow_merge = True
                if overlap >= self.merge_bbox_overlap_min:
                    allow_merge = True
                if support_overlap >= 0.15:
                    allow_merge = True
                if best_view_similarity >= 0.35 and centroid_dist <= (max_dist * 1.25):
                    allow_merge = True
                if descriptor_similarity is not None and descriptor_similarity >= self.merge_descriptor_similarity_min:
                    allow_merge = True
                if not allow_merge:
                    continue
                merged_pairs.append(
                    (
                        object_id_a,
                        object_id_b,
                        f"{'same_label' if same_label else 'descriptor_merge'}:{label_a}|{label_b}",
                        centroid_dist,
                        overlap,
                        support_overlap,
                        best_view_similarity,
                        temporal_continuity,
                        -1.0 if descriptor_similarity is None else descriptor_similarity,
                    )
                )

        for (
            keep_id,
            drop_id,
            reason,
            centroid_dist,
            overlap,
            support_overlap,
            best_view_similarity,
            temporal_continuity,
            descriptor_similarity,
        ) in merged_pairs:
            if keep_id not in self.objects or drop_id not in self.objects:
                continue
            self._merge_object_into(
                keep_id,
                drop_id,
                reason,
                centroid_dist,
                overlap,
                support_overlap,
                best_view_similarity,
                temporal_continuity,
                descriptor_similarity,
            )

    def _merge_object_into(
        self,
        keep_id: str,
        drop_id: str,
        reason: str,
        centroid_dist: float,
        overlap: float,
        support_overlap: float,
        best_view_similarity: float,
        temporal_continuity: float,
        descriptor_similarity: float,
    ) -> None:
        keep = self.objects.get(keep_id)
        drop = self.objects.get(drop_id)
        if keep is None or drop is None:
            return
        for local_states in self.point_assignments.values():
            for state in local_states.values():
                scores = dict(state.get("object_scores", {}))
                if drop_id not in scores:
                    continue
                scores[keep_id] = max(float(scores.get(keep_id, 0.0)), float(scores.get(drop_id, 0.0)))
                scores.pop(drop_id, None)
                state["object_scores"] = scores
                if state.get("current_object_id") == drop_id:
                    state["current_object_id"] = keep_id
                self._reconcile_point_assignment(state)
        for label, score in drop.label_votes.items():
            keep.label_votes[label] = keep.label_votes.get(label, 0.0) + float(score)
        keep.yolo_logit_sum = self._merge_bounded_score_maps(
            getattr(keep, "yolo_logit_sum", {}),
            getattr(drop, "yolo_logit_sum", {}),
        )
        keep.posterior = self._merge_bounded_score_maps(
            getattr(keep, "posterior", {}),
            getattr(drop, "posterior", {}),
        )
        keep.observation_count += int(drop.observation_count)
        keep.promotion_evidence += float(drop.promotion_evidence)
        keep.negative_evidence_score = min(keep.negative_evidence_score, drop.negative_evidence_score)
        keep.best_view_keyframes = self._update_best_views(
            keep.best_view_keyframes + drop.best_view_keyframes,
            drop.best_view_keyframes[0] if drop.best_view_keyframes else 0,
        )
        merged_views = list(keep.descriptor_views) + list(drop.descriptor_views)
        merged_views.sort(key=lambda item: float(item.get("quality_score", 0.0)), reverse=True)
        keep.descriptor_views = merged_views[: self.max_best_views]
        merged_bank = list(getattr(keep, "descriptor_bank", [])) + list(getattr(drop, "descriptor_bank", []))
        merged_bank.sort(key=lambda item: float(item.get("quality_score", 0.0)), reverse=True)
        keep.descriptor_bank = merged_bank[: max(self.max_best_views, 8)]
        self._recompute_semantic_descriptor(keep)
        keep.last_seen_stamp = max(float(keep.last_seen_stamp), float(drop.last_seen_stamp))
        keep.merge_reason = reason
        keep.label_compatibility_score = max(
            keep.label_compatibility_score,
            keep.stability_score,
            float(best_view_similarity),
            float(temporal_continuity),
            float(descriptor_similarity),
        )
        keep.support_overlap_score = max(keep.support_overlap_score, float(overlap), float(support_overlap))
        keep.dirty_flag = True
        self.objects.pop(drop_id, None)

    def _update_object_states(self, stamp_sec: float) -> None:
        for obj in self.objects.values():
            support_count = sum(len(point_indices) for _local_cloud_id, point_indices in obj.point_support_refs)
            if support_count <= 0:
                obj.state = "stale"
                if getattr(obj, "node_status", "") not in {"reject", "unknown"}:
                    obj.node_status = "tentative"
                continue
            if (stamp_sec - float(obj.last_seen_stamp)) > self.object_stale_time_sec:
                obj.state = "stale"
                if getattr(obj, "node_status", "") not in {"reject", "unknown"}:
                    obj.node_status = "tentative"
                continue
            if obj.state == "pending":
                if (
                    obj.observation_count >= self.pending_commit_observations
                    and obj.promotion_evidence >= self.pending_commit_evidence
                    and support_count >= self.pending_seed_points
                ):
                    obj.state = "active"
            elif obj.state == "stale":
                if obj.observation_count > 0 and support_count >= self.pending_seed_points:
                    obj.state = "active"
            obj.node_status = self._infer_node_status(obj)

    def _prune_point_assignments(self, stamp_sec: float) -> None:
        for local_cloud_id in list(self.point_assignments.keys()):
            local_states = self.point_assignments[local_cloud_id]
            for point_index in list(local_states.keys()):
                state = local_states[point_index]
                scores = {
                    str(object_id): float(score)
                    for object_id, score in dict(state.get("object_scores", {})).items()
                    if float(score) > 1e-6
                }
                if (stamp_sec - float(state.get("last_seen_stamp", 0.0))) > self.point_stale_time_sec:
                    scores = {
                        object_id: max(0.0, score - self.point_score_decrement)
                        for object_id, score in scores.items()
                    }
                    scores = {object_id: score for object_id, score in scores.items() if score > 1e-6}
                state["object_scores"] = scores
                self._reconcile_point_assignment(state)
                if not state.get("current_object_id") and not scores:
                    local_states.pop(point_index, None)
            if not local_states:
                self.point_assignments.pop(local_cloud_id, None)

    def _prune_objects(self) -> None:
        empty_objects: list[str] = []
        for object_id, obj in self.objects.items():
            has_support = any(len(point_indices) > 0 for _local_cloud_id, point_indices in obj.point_support_refs)
            if not has_support:
                empty_objects.append(object_id)
                continue
            if obj.state == "stale" and obj.negative_evidence_score > 8.0:
                empty_objects.append(object_id)
        for object_id in empty_objects:
            self.objects.pop(object_id, None)

    def _cloud_path_for_local_cloud(self, local_cloud_id: str) -> Path | None:
        cloud_path = self.local_cloud_paths.get(local_cloud_id)
        if not cloud_path:
            return None
        return Path(cloud_path)

    def _load_xyz_by_id(self, local_cloud_id: str) -> np.ndarray | None:
        cached = self.local_cloud_xyz_cache.get(local_cloud_id)
        if cached is not None:
            return cached
        cloud_path = self._cloud_path_for_local_cloud(local_cloud_id)
        if cloud_path is None or not cloud_path.exists():
            return None
        data = np.load(cloud_path)
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        self.local_cloud_xyz_cache[local_cloud_id] = xyz
        return xyz

    def _point_xyz(self, local_cloud_id: str, point_index: int) -> np.ndarray | None:
        xyz = self._load_xyz_by_id(local_cloud_id)
        if xyz is None or point_index < 0 or point_index >= xyz.shape[0]:
            return None
        return xyz[point_index]

    def _voxel_key(self, xyz: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(np.asarray(xyz, dtype=np.float32) / self.spatial_voxel_size_m).astype(np.int32).tolist())

    def _neighbor_voxel_keys(self, voxel_key: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        keys: list[tuple[int, int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    keys.append((voxel_key[0] + dx, voxel_key[1] + dy, voxel_key[2] + dz))
        return keys
