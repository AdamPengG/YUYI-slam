from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_types import LocalCloudPacket, ObjectMemory, ObservationLink


@dataclass
class ObjectMemoryUpdateResult:
    updated_object_ids: list[str]
    created_object_ids: list[str]
    ignored_observations: int


class ObjectMemoryStore:
    def __init__(
        self,
        min_points_per_observation: int = 24,
        min_mask_area: int = 96,
        max_best_views: int = 5,
        merge_centroid_radius_m: float = 0.75,
    ) -> None:
        self.min_points_per_observation = int(min_points_per_observation)
        self.min_mask_area = int(min_mask_area)
        self.max_best_views = int(max_best_views)
        self.merge_centroid_radius_m = float(merge_centroid_radius_m)
        self.objects: dict[str, ObjectMemory] = {}
        self._next_object_id = 0

    def update(
        self,
        observations: list[ObservationLink],
        keyframe_stamp_sec: float,
        keyframe_id: int,
        local_cloud: LocalCloudPacket,
    ) -> ObjectMemoryUpdateResult:
        updated: list[str] = []
        created: list[str] = []
        ignored = 0
        xyz = self._load_xyz(local_cloud)

        for obs in observations:
            if len(obs.point_indices) < self.min_points_per_observation or obs.mask_area < self.min_mask_area:
                ignored += 1
                continue

            support_xyz = xyz[np.asarray(obs.point_indices, dtype=np.int32)]
            centroid = support_xyz.mean(axis=0).astype(np.float32)
            bbox_min = support_xyz.min(axis=0).astype(np.float32)
            bbox_max = support_xyz.max(axis=0).astype(np.float32)

            object_id = self._resolve_object_id(obs, centroid)
            label_votes = self._votes_from_observation(obs)

            if object_id in self.objects:
                obj = self.objects[object_id]
                obs.candidate_object_id = object_id
                obj.point_support_refs.append((obs.local_cloud_id, [int(v) for v in obs.point_indices]))
                obj.observation_count += 1
                obj.last_seen_stamp = float(keyframe_stamp_sec)
                obj.best_view_keyframes = self._update_best_views(obj.best_view_keyframes, keyframe_id)
                obj.centroid_world = self._running_average(obj.centroid_world, centroid.tolist(), obj.observation_count)
                obj.bbox_world = self._merge_bbox(obj.bbox_world, bbox_min, bbox_max)
                for label, score in label_votes.items():
                    obj.label_votes[label] = obj.label_votes.get(label, 0.0) + score
                obj.stability_score = self._compute_stability(obj)
                obj.completeness_score = min(1.0, obj.observation_count / 8.0)
                obj.dirty_flag = True
                updated.append(object_id)
            else:
                object_id = self._create_object_id()
                obs.candidate_object_id = object_id
                obj = ObjectMemory(
                    object_id=object_id,
                    point_support_refs=[(obs.local_cloud_id, [int(v) for v in obs.point_indices])],
                    centroid_world=[float(v) for v in centroid.tolist()],
                    bbox_world=[float(v) for v in bbox_min.tolist() + bbox_max.tolist()],
                    label_votes=label_votes,
                    embedding_path=None,
                    best_view_keyframes=[int(keyframe_id)],
                    observation_count=1,
                    stability_score=1.0 if label_votes else 0.0,
                    completeness_score=0.1,
                    dirty_flag=True,
                    last_seen_stamp=float(keyframe_stamp_sec),
                )
                self.objects[object_id] = obj
                created.append(object_id)

        return ObjectMemoryUpdateResult(updated, created, ignored)

    def to_dict(self) -> dict[str, dict]:
        return {object_id: obj.to_dict() for object_id, obj in self.objects.items()}

    @classmethod
    def from_dict(cls, payload: dict[str, dict]) -> "ObjectMemoryStore":
        store = cls()
        raw_objects = payload.get("objects", payload) if isinstance(payload, dict) else {}
        if not isinstance(raw_objects, dict):
            raw_objects = {}
        for object_id, object_payload in raw_objects.items():
            store.objects[str(object_id)] = ObjectMemory.from_dict(object_payload)
        if store.objects:
            max_numeric = max(int(obj_id.split("_")[-1]) for obj_id in store.objects)
            store._next_object_id = max_numeric + 1
        return store

    def _resolve_object_id(self, observation: ObservationLink, centroid: np.ndarray) -> str | None:
        if observation.candidate_object_id is not None and observation.candidate_object_id in self.objects:
            return observation.candidate_object_id

        best_id = None
        best_dist = None
        for object_id, obj in self.objects.items():
            dist = float(np.linalg.norm(np.asarray(obj.centroid_world, dtype=np.float32) - centroid))
            if dist <= self.merge_centroid_radius_m and (best_dist is None or dist < best_dist):
                best_id = object_id
                best_dist = dist
        return best_id

    def _create_object_id(self) -> str:
        object_id = f"obj_{self._next_object_id:05d}"
        self._next_object_id += 1
        return object_id

    def _votes_from_observation(self, observation: ObservationLink) -> dict[str, float]:
        if observation.semantic_label_candidates and observation.semantic_scores:
            return {
                str(label): float(score)
                for label, score in zip(observation.semantic_label_candidates, observation.semantic_scores, strict=False)
            }
        if observation.semantic_label_candidates:
            weight = 1.0 / max(len(observation.semantic_label_candidates), 1)
            return {str(label): weight for label in observation.semantic_label_candidates}
        return {}

    def _load_xyz(self, local_cloud: LocalCloudPacket) -> np.ndarray:
        with np.load(Path(local_cloud.cloud_path)) as data:
            return np.asarray(data["xyz"], dtype=np.float32)

    def _running_average(self, current: list[float], new_value: list[float], count: int) -> list[float]:
        if count <= 1:
            return [float(v) for v in new_value]
        alpha = 1.0 / float(count)
        return [float((1.0 - alpha) * c + alpha * n) for c, n in zip(current, new_value, strict=False)]

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

    def _compute_stability(self, obj: ObjectMemory) -> float:
        if not obj.label_votes:
            return 0.0
        total = float(sum(obj.label_votes.values()))
        best = float(max(obj.label_votes.values()))
        if total <= 1e-6:
            return 0.0
        return best / total
