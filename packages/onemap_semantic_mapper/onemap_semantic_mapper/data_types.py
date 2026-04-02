from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SensorConfig:
    sensor_config_id: str
    calib_version: str
    camera_model: str
    intrinsics: dict[str, float]
    t_lidar_cam: list[float]
    image_size: tuple[int, int]
    distortion_model: str = "none"
    r_lidar_cam: list[float] = field(default_factory=list)
    t_imu_lidar: list[float] = field(default_factory=list)
    r_imu_lidar: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_size"] = list(self.image_size)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SensorConfig":
        image_size = payload.get("image_size", [0, 0])
        return cls(
            sensor_config_id=str(payload["sensor_config_id"]),
            calib_version=str(payload["calib_version"]),
            camera_model=str(payload["camera_model"]),
            intrinsics=dict(payload["intrinsics"]),
            t_lidar_cam=[float(v) for v in payload["t_lidar_cam"]],
            image_size=(int(image_size[0]), int(image_size[1])),
            distortion_model=str(payload.get("distortion_model", "none")),
            r_lidar_cam=[float(v) for v in payload.get("r_lidar_cam", [])],
            t_imu_lidar=[float(v) for v in payload.get("t_imu_lidar", [])],
            r_imu_lidar=[float(v) for v in payload.get("r_imu_lidar", [])],
        )


@dataclass
class KeyframePacket:
    keyframe_id: int
    stamp_sec: float
    t_world_body: list[float]
    t_world_lidar: list[float]
    t_world_cam: list[float]
    rgb_path: str
    depth_path: str | None
    local_cloud_ref: str
    local_cloud_frame: str
    sensor_config_id: str
    calib_version: str
    pose_source: str
    status: str
    source_scan_ids: list[str] = field(default_factory=list)
    selection_reasons: list[str] = field(default_factory=list)
    pose_alignment: str = "unknown"
    pose_dt_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KeyframePacket":
        return cls(
            keyframe_id=int(payload["keyframe_id"]),
            stamp_sec=float(payload["stamp_sec"]),
            t_world_body=[float(v) for v in payload["t_world_body"]],
            t_world_lidar=[float(v) for v in payload["t_world_lidar"]],
            t_world_cam=[float(v) for v in payload["t_world_cam"]],
            rgb_path=str(payload["rgb_path"]),
            depth_path=None if payload.get("depth_path") is None else str(payload["depth_path"]),
            local_cloud_ref=str(payload["local_cloud_ref"]),
            local_cloud_frame=str(payload["local_cloud_frame"]),
            sensor_config_id=str(payload["sensor_config_id"]),
            calib_version=str(payload["calib_version"]),
            pose_source=str(payload["pose_source"]),
            status=str(payload["status"]),
            source_scan_ids=[str(v) for v in payload.get("source_scan_ids", [])],
            selection_reasons=[str(v) for v in payload.get("selection_reasons", [])],
            pose_alignment=str(payload.get("pose_alignment", "unknown")),
            pose_dt_sec=None if payload.get("pose_dt_sec") is None else float(payload["pose_dt_sec"]),
        )


@dataclass
class LocalCloudPacket:
    local_cloud_id: str
    source_scan_ids: list[str]
    stamp_start: float
    stamp_end: float
    frame: str
    cloud_path: str
    point_count: int
    parent_submap_id: str | None = None
    cloud_kind: str = "generic"
    quality_fields_present: bool = False
    has_uv: bool = False
    has_normal: bool = False
    has_visibility_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalCloudPacket":
        return cls(
            local_cloud_id=str(payload["local_cloud_id"]),
            source_scan_ids=[str(v) for v in payload.get("source_scan_ids", [])],
            stamp_start=float(payload["stamp_start"]),
            stamp_end=float(payload["stamp_end"]),
            frame=str(payload["frame"]),
            cloud_path=str(payload["cloud_path"]),
            point_count=int(payload["point_count"]),
            parent_submap_id=None if payload.get("parent_submap_id") is None else str(payload["parent_submap_id"]),
            cloud_kind=str(payload.get("cloud_kind", "generic")),
            quality_fields_present=bool(payload.get("quality_fields_present", False)),
            has_uv=bool(payload.get("has_uv", False)),
            has_normal=bool(payload.get("has_normal", False)),
            has_visibility_flag=bool(payload.get("has_visibility_flag", False)),
        )


@dataclass
class ObservationLink:
    keyframe_id: int
    mask_id: int
    local_cloud_id: str
    point_indices: list[int]
    semantic_label_candidates: list[str]
    semantic_scores: list[float]
    candidate_object_id: str | None
    vote_count: int
    visibility_score: float
    bbox_xyxy: list[int]
    mask_area: int
    quality_score: float = 0.0
    foreground_depth_median: float | None = None
    foreground_depth_p10: float | None = None
    candidate_object_scores: dict[str, float] = field(default_factory=dict)
    abstained: bool = False
    abstain_reason: str = ""
    semantic_embedding: list[float] = field(default_factory=list)
    semantic_embedding_variant: str = ""
    yolo_label_candidates: list[str] = field(default_factory=list)
    yolo_scores: list[float] = field(default_factory=list)
    detection_score: float = 0.0
    observation_kind: str = "thing"
    view_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("semantic_embedding", None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationLink":
        return cls(
            keyframe_id=int(payload["keyframe_id"]),
            mask_id=int(payload["mask_id"]),
            local_cloud_id=str(payload["local_cloud_id"]),
            point_indices=[int(v) for v in payload.get("point_indices", [])],
            semantic_label_candidates=[str(v) for v in payload.get("semantic_label_candidates", [])],
            semantic_scores=[float(v) for v in payload.get("semantic_scores", [])],
            candidate_object_id=None
            if payload.get("candidate_object_id") is None
            else str(payload["candidate_object_id"]),
            vote_count=int(payload.get("vote_count", 0)),
            visibility_score=float(payload.get("visibility_score", 0.0)),
            bbox_xyxy=[int(v) for v in payload.get("bbox_xyxy", [0, 0, 0, 0])],
            mask_area=int(payload.get("mask_area", 0)),
            quality_score=float(payload.get("quality_score", 0.0)),
            foreground_depth_median=None
            if payload.get("foreground_depth_median") is None
            else float(payload.get("foreground_depth_median")),
            foreground_depth_p10=None
            if payload.get("foreground_depth_p10") is None
            else float(payload.get("foreground_depth_p10")),
            candidate_object_scores={
                str(key): float(value) for key, value in payload.get("candidate_object_scores", {}).items()
            },
            abstained=bool(payload.get("abstained", False)),
            abstain_reason=str(payload.get("abstain_reason", "")),
            semantic_embedding=[float(v) for v in payload.get("semantic_embedding", [])],
            semantic_embedding_variant=str(payload.get("semantic_embedding_variant", "")),
            yolo_label_candidates=[str(v) for v in payload.get("yolo_label_candidates", [])],
            yolo_scores=[float(v) for v in payload.get("yolo_scores", [])],
            detection_score=float(payload.get("detection_score", 0.0)),
            observation_kind=str(payload.get("observation_kind", "thing")),
            view_quality=float(payload.get("view_quality", 0.0)),
        )


@dataclass
class ObjectMemory:
    object_id: str
    point_support_refs: list[tuple[str, list[int]]]
    centroid_world: list[float]
    bbox_world: list[float]
    label_votes: dict[str, float]
    embedding_path: str | None
    best_view_keyframes: list[int]
    observation_count: int
    stability_score: float
    completeness_score: float
    dirty_flag: bool
    last_seen_stamp: float
    state: str = "active"
    pending_since_keyframe: int | None = None
    promotion_evidence: float = 0.0
    negative_evidence_score: float = 0.0
    merge_reason: str = ""
    label_compatibility_score: float = 0.0
    support_overlap_score: float = 0.0
    descriptor_views: list[dict[str, Any]] = field(default_factory=list)
    semantic_descriptor: list[float] | None = None
    semantic_descriptor_keyframe: int | None = None
    semantic_descriptor_confidence: float = 0.0
    node_type: str = "thing"
    node_status: str = "tentative"
    posterior: dict[str, float] = field(default_factory=dict)
    unknown_score: float = 1.0
    reject_score: float = 0.0
    yolo_logit_sum: dict[str, float] = field(default_factory=dict)
    descriptor_bank: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "point_support_refs": [[ref, indices] for ref, indices in self.point_support_refs],
            "centroid_world": self.centroid_world,
            "bbox_world": self.bbox_world,
            "label_votes": self.label_votes,
            "embedding_path": self.embedding_path,
            "best_view_keyframes": self.best_view_keyframes,
            "observation_count": self.observation_count,
            "stability_score": self.stability_score,
            "completeness_score": self.completeness_score,
            "dirty_flag": self.dirty_flag,
            "last_seen_stamp": self.last_seen_stamp,
            "state": self.state,
            "pending_since_keyframe": self.pending_since_keyframe,
            "promotion_evidence": self.promotion_evidence,
            "negative_evidence_score": self.negative_evidence_score,
            "merge_reason": self.merge_reason,
            "label_compatibility_score": self.label_compatibility_score,
            "support_overlap_score": self.support_overlap_score,
            "descriptor_views": self.descriptor_views,
            "semantic_descriptor": self.semantic_descriptor,
            "semantic_descriptor_keyframe": self.semantic_descriptor_keyframe,
            "semantic_descriptor_confidence": self.semantic_descriptor_confidence,
            "node_type": self.node_type,
            "node_status": self.node_status,
            "posterior": self.posterior,
            "unknown_score": self.unknown_score,
            "reject_score": self.reject_score,
            "yolo_logit_sum": self.yolo_logit_sum,
            "descriptor_bank": self.descriptor_bank,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectMemory":
        return cls(
            object_id=str(payload["object_id"]),
            point_support_refs=[
                (str(item[0]), [int(v) for v in item[1]]) for item in payload.get("point_support_refs", [])
            ],
            centroid_world=[float(v) for v in payload.get("centroid_world", [0.0, 0.0, 0.0])],
            bbox_world=[float(v) for v in payload.get("bbox_world", [0.0] * 6)],
            label_votes={str(k): float(v) for k, v in payload.get("label_votes", {}).items()},
            embedding_path=None if payload.get("embedding_path") is None else str(payload["embedding_path"]),
            best_view_keyframes=[int(v) for v in payload.get("best_view_keyframes", [])],
            observation_count=int(payload.get("observation_count", 0)),
            stability_score=float(payload.get("stability_score", 0.0)),
            completeness_score=float(payload.get("completeness_score", 0.0)),
            dirty_flag=bool(payload.get("dirty_flag", False)),
            last_seen_stamp=float(payload.get("last_seen_stamp", 0.0)),
            state=str(payload.get("state", "active")),
            pending_since_keyframe=None
            if payload.get("pending_since_keyframe") is None
            else int(payload.get("pending_since_keyframe")),
            promotion_evidence=float(payload.get("promotion_evidence", 0.0)),
            negative_evidence_score=float(payload.get("negative_evidence_score", 0.0)),
            merge_reason=str(payload.get("merge_reason", "")),
            label_compatibility_score=float(payload.get("label_compatibility_score", 0.0)),
            support_overlap_score=float(payload.get("support_overlap_score", 0.0)),
            descriptor_views=[
                {
                    "keyframe_id": int(item.get("keyframe_id", -1)),
                    "quality_score": float(item.get("quality_score", 0.0)),
                    "embedding": [float(v) for v in item.get("embedding", [])],
                    "semantic_label_candidates": [str(v) for v in item.get("semantic_label_candidates", [])],
                    "semantic_scores": [float(v) for v in item.get("semantic_scores", [])],
                    "bbox_xyxy": [int(v) for v in item.get("bbox_xyxy", [0, 0, 0, 0])],
                    "view_quality": float(item.get("view_quality", item.get("quality_score", 0.0))),
                    "observation_kind": str(item.get("observation_kind", "thing")),
                    "yolo_label_candidates": [str(v) for v in item.get("yolo_label_candidates", [])],
                    "yolo_scores": [float(v) for v in item.get("yolo_scores", [])],
                    "source": str(item.get("source", "")),
                }
                for item in payload.get("descriptor_views", [])
            ],
            semantic_descriptor=None
            if payload.get("semantic_descriptor") is None
            else [float(v) for v in payload.get("semantic_descriptor", [])],
            semantic_descriptor_keyframe=None
            if payload.get("semantic_descriptor_keyframe") is None
            else int(payload.get("semantic_descriptor_keyframe")),
            semantic_descriptor_confidence=float(payload.get("semantic_descriptor_confidence", 0.0)),
            node_type=str(payload.get("node_type", "thing")),
            node_status=str(payload.get("node_status", "tentative")),
            posterior={str(k): float(v) for k, v in payload.get("posterior", {}).items()},
            unknown_score=float(payload.get("unknown_score", 1.0)),
            reject_score=float(payload.get("reject_score", 0.0)),
            yolo_logit_sum={str(k): float(v) for k, v in payload.get("yolo_logit_sum", {}).items()},
            descriptor_bank=[
                {
                    "keyframe_id": int(item.get("keyframe_id", -1)),
                    "quality_score": float(item.get("quality_score", 0.0)),
                    "embedding": [float(v) for v in item.get("embedding", [])],
                    "semantic_label_candidates": [str(v) for v in item.get("semantic_label_candidates", [])],
                    "semantic_scores": [float(v) for v in item.get("semantic_scores", [])],
                    "bbox_xyxy": [int(v) for v in item.get("bbox_xyxy", [0, 0, 0, 0])],
                    "view_quality": float(item.get("view_quality", item.get("quality_score", 0.0))),
                    "observation_kind": str(item.get("observation_kind", "thing")),
                    "yolo_label_candidates": [str(v) for v in item.get("yolo_label_candidates", [])],
                    "yolo_scores": [float(v) for v in item.get("yolo_scores", [])],
                    "source": str(item.get("source", "")),
                }
                for item in payload.get("descriptor_bank", [])
            ],
        )
