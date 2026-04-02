#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def pack_rgb(color: Sequence[int]) -> int:
    r, g, b = [int(np.clip(v, 0, 255)) for v in color]
    return (r << 16) | (g << 8) | b


def generate_palette(labels: Sequence[str]) -> Dict[str, Tuple[int, int, int]]:
    base = [
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (231, 111, 81),
        (168, 218, 220),
        (145, 30, 180),
        (67, 170, 139),
    ]
    palette = {}
    for idx, label in enumerate(labels):
        palette[label] = base[idx % len(base)]
    return palette


@dataclass
class Detection:
    label: str
    score: float
    box: Tuple[float, float, float, float]
    color: Tuple[int, int, int]
    label_id: int
    mask: Optional[np.ndarray] = None


@dataclass
class BufferedImage:
    stamp: float
    image_rgb: np.ndarray
    detections: List[Detection] = field(default_factory=list)


@dataclass
class BufferedDepth:
    stamp: float
    depth_m: np.ndarray


@dataclass
class BufferedOdom:
    stamp: float
    pose_rotation: np.ndarray
    pose_translation: np.ndarray


@dataclass
class SemanticVoxel:
    xyz_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    point_count: int = 0
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    label_votes: Counter = field(default_factory=Counter)
    last_seen: float = 0.0
    stable_label: str = ""


@dataclass
class DetectionAssociation:
    detection: Detection
    seed_indices: np.ndarray
    pixel_mask: Optional[np.ndarray] = None


@dataclass
class SemanticInstance:
    instance_id: int
    xyz_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    point_count: int = 0
    bounds_min: np.ndarray = field(default_factory=lambda: np.full(3, np.inf, dtype=np.float64))
    bounds_max: np.ndarray = field(default_factory=lambda: np.full(3, -np.inf, dtype=np.float64))
    label_votes: Counter = field(default_factory=Counter)
    stable_label: str = ""
    last_seen: float = 0.0


@dataclass
class SemanticObject:
    label: str
    centroid: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    voxel_count: int


class UltralyticsDetector:
    def __init__(
        self,
        labels: Sequence[str],
        label_to_id: Dict[str, int],
        label_to_color: Dict[str, Tuple[int, int, int]],
        model_name: str,
        fallback_model_name: str,
        score_threshold: float,
        logger,
    ) -> None:
        self.labels = list(labels)
        self.label_to_id = label_to_id
        self.label_to_color = label_to_color
        self.score_threshold = score_threshold
        self.logger = logger
        self._world_mode = False
        self._fallback_model_name = fallback_model_name
        self._aliases = {
            "dining table": "table",
            "table": "table",
            "bench": "table",
            "tv monitor": "tv",
            "tv": "tv",
            "tvmonitor": "tv",
            "sofa": "couch",
            "couch": "couch",
            "potted plant": "potted plant",
            "plant": "potted plant",
        }

        try:
            from ultralytics import YOLO, YOLOWorld  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(
                "ultralytics is not installed for /usr/bin/python3. "
                "Install it with `/usr/bin/python3 -m pip install --user ultralytics`."
            ) from exc

        self._model = None
        init_errors: List[str] = []

        if "world" in model_name.lower():
            try:
                self._model = YOLOWorld(model_name)
                self._world_mode = True
                self._model.set_classes(self.labels)
                self.logger.info(f"Semantic detector set to world classes: {', '.join(self.labels)}")
            except Exception as exc:  # pragma: no cover - runtime environment dependent
                init_errors.append(f"YOLOWorld({model_name}) failed: {exc}")
                self._model = None
                self._world_mode = False

        if self._model is None:
            try:
                model_to_use = fallback_model_name if "world" in model_name.lower() else model_name
                self._model = YOLO(model_to_use)
                self.logger.warn(
                    "Using standard YOLO detector backend. "
                    "This is stable for common classes like table/chair but not full open-vocabulary mode."
                )
            except Exception as exc:  # pragma: no cover - runtime environment dependent
                init_errors.append(f"YOLO({fallback_model_name}) failed: {exc}")

        if self._model is None:
            joined = "; ".join(init_errors) if init_errors else "unknown error"
            raise RuntimeError(f"Failed to initialize detector backend: {joined}")

    def detect(self, image_rgb: np.ndarray) -> List[Detection]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            source=image_bgr,
            conf=self.score_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        raw_masks = None
        result_masks = getattr(result, "masks", None)
        if result_masks is not None and getattr(result_masks, "data", None) is not None:
            raw_masks = result_masks.data.detach().cpu().numpy()

        detections: List[Detection] = []
        for idx in range(len(boxes)):
            box = boxes[idx]
            score = float(box.conf[0].item())
            if score < self.score_threshold:
                continue
            cls_idx = int(box.cls[0].item())
            raw_label = str(names[cls_idx])
            label = self._aliases.get(raw_label, raw_label)
            if label not in self.label_to_id:
                continue
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32).tolist()
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue

            det_mask = None
            if raw_masks is not None and idx < raw_masks.shape[0]:
                det_mask = cv2.resize(
                    raw_masks[idx].astype(np.float32),
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5

            detections.append(
                Detection(
                    label=label,
                    score=score,
                    box=(x1, y1, x2, y2),
                    color=self.label_to_color[label],
                    label_id=self.label_to_id[label],
                    mask=det_mask,
                )
            )

        detections.sort(key=lambda det: det.score, reverse=True)
        return detections


class SemanticMapperNode(Node):
    def __init__(self) -> None:
        super().__init__("onemap_semantic_mapper")
        self.bridge = CvBridge()

        self._declare_parameters()
        self._load_parameters()

        self._image_buffer: Deque[BufferedImage] = deque(maxlen=10)
        self._depth_buffer: Deque[BufferedDepth] = deque(maxlen=10)
        self._odom_buffer: Deque[BufferedOdom] = deque(maxlen=50)
        self._semantic_voxels: Dict[Tuple[int, int, int], SemanticVoxel] = {}
        self._semantic_raw_voxels: Dict[Tuple[int, int, int], SemanticVoxel] = {}
        self._semantic_instances: Dict[int, SemanticInstance] = {}
        self._next_instance_id = 1
        self._last_status_log_time = 0.0
        self._latest_sensor_stamp: Optional[float] = None
        self._odom_time_offset: Optional[float] = None
        self._camera_info_received = False

        self.detector = UltralyticsDetector(
            labels=self.query_labels,
            label_to_id=self.label_to_id,
            label_to_color=self.label_to_color,
            model_name=self.detector_model,
            fallback_model_name=self.fallback_detector_model,
            score_threshold=self.min_detection_score,
            logger=self.get_logger(),
        )

        self.current_cloud_pub = self.create_publisher(PointCloud2, self.current_cloud_topic, 10)
        self.map_cloud_pub = self.create_publisher(PointCloud2, self.map_cloud_topic, 10)
        self.raw_map_cloud_pub = self.create_publisher(PointCloud2, self.raw_map_cloud_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.image_sub = self.create_subscription(Image, self.img_topic, self._image_callback, 20)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self._depth_callback, 20)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, 20)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 50)
        self.cloud_sub = self.create_subscription(PointCloud2, self.lid_topic, self._cloud_callback, 20)
        self.map_publish_timer = self.create_timer(self.map_publish_period_sec, self._publish_map_outputs)

        self.get_logger().info(
            "Semantic mapper ready. "
            f"Image={self.img_topic}, Depth={self.depth_topic}, LiDAR={self.lid_topic}, Odom={self.odom_topic}, "
            f"labels={self.query_labels}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "common.img_topic": "/robot_rgb",
            "common.depth_topic": "/depth",
            "common.camera_info_topic": "/camera_info",
            "common.lid_topic": "/livox/lidar",
            "semantic.odom_topic": "/aft_mapped_to_init",
            "semantic.current_cloud_topic": "/semantic_current_cloud",
            "semantic.map_cloud_topic": "/semantic_map_cloud",
            "semantic.raw_map_cloud_topic": "/semantic_map_raw_cloud",
            "semantic.marker_topic": "/semantic_labels",
            "semantic.debug_image_topic": "/semantic_debug_image",
            "semantic.sync_slop_sec": 0.25,
            "semantic.max_range_m": 8.0,
            "semantic.min_detection_score": 0.25,
            "semantic.min_points_per_detection": 8,
            "semantic.use_depth_filter": True,
            "semantic.depth_consistency_margin_m": 0.18,
            "semantic.use_cluster_filter": True,
            "semantic.object_cluster_radius_m": 0.35,
            "semantic.min_points_per_object_cluster": 8,
            "semantic.use_cluster_completion": True,
            "semantic.completion_cluster_radius_m": 0.16,
            "semantic.completion_min_seed_points": 8,
            "semantic.completion_min_seed_ratio": 0.05,
            "semantic.completion_dominance_ratio": 1.25,
            "semantic.completion_max_cluster_extent_m": 2.0,
            "semantic.use_instance_tracking": True,
            "semantic.instance_depth_stride": 4,
            "semantic.instance_min_depth_points": 48,
            "semantic.instance_bbox_padding_m": 0.25,
            "semantic.instance_match_min_iou": 0.05,
            "semantic.instance_match_max_centroid_distance_m": 1.0,
            "semantic.instance_max_staleness_sec": 0.0,
            "semantic.use_map_object_propagation": True,
            "semantic.map_object_bbox_padding_m": 0.35,
            "semantic.map_object_min_overlap_ratio": 0.25,
            "semantic.map_object_max_centroid_distance_m": 0.8,
            "semantic.map_object_max_cluster_extent_m": 2.0,
            "semantic.voxel_size": 0.15,
            "semantic.min_points_per_voxel": 3,
            "semantic.raw_voxel_size": 0.03,
            "semantic.raw_min_points_per_voxel": 1,
            "semantic.cluster_radius_m": 0.55,
            "semantic.min_voxels_per_cluster": 6,
            "semantic.box_padding_m": 0.08,
            "semantic.box_min_size_m": 0.12,
            "semantic.map_publish_period_sec": 0.5,
            "semantic.publish_debug_image": True,
            "semantic.label_switch_margin_votes": 4,
            "semantic.label_switch_ratio": 1.15,
            "semantic.detector_model": "yolov8s.pt",
            "semantic.fallback_detector_model": "yolov8s.pt",
            "semantic.query_labels": [
                "table",
                "chair",
                "couch",
                "potted plant",
                "tv",
                "bed",
                "toilet",
            ],
            "camera.cam_width": 1280,
            "camera.cam_height": 720,
            "camera.cam_fx": 640.0,
            "camera.cam_fy": 493.355477,
            "camera.cam_cx": 640.0,
            "camera.cam_cy": 360.0,
            "extrin_calib.Rcl": [
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                0.0,
            ],
            "extrin_calib.Pcl": [-0.000000329, 0.319520463, -0.001662359],
            "extrin_calib.extrinsic_T": [0.0, 0.0, 0.104],
            "extrin_calib.extrinsic_R": [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.img_topic = self.get_parameter("common.img_topic").value
        self.depth_topic = self.get_parameter("common.depth_topic").value
        self.camera_info_topic = self.get_parameter("common.camera_info_topic").value
        self.lid_topic = self.get_parameter("common.lid_topic").value
        self.odom_topic = self.get_parameter("semantic.odom_topic").value
        self.current_cloud_topic = self.get_parameter("semantic.current_cloud_topic").value
        self.map_cloud_topic = self.get_parameter("semantic.map_cloud_topic").value
        self.raw_map_cloud_topic = self.get_parameter("semantic.raw_map_cloud_topic").value
        self.marker_topic = self.get_parameter("semantic.marker_topic").value
        self.debug_image_topic = self.get_parameter("semantic.debug_image_topic").value

        self.sync_slop_sec = float(self.get_parameter("semantic.sync_slop_sec").value)
        self.max_range_m = float(self.get_parameter("semantic.max_range_m").value)
        self.min_detection_score = float(self.get_parameter("semantic.min_detection_score").value)
        self.min_points_per_detection = int(self.get_parameter("semantic.min_points_per_detection").value)
        self.use_depth_filter = bool(self.get_parameter("semantic.use_depth_filter").value)
        self.depth_consistency_margin_m = float(self.get_parameter("semantic.depth_consistency_margin_m").value)
        self.use_cluster_filter = bool(self.get_parameter("semantic.use_cluster_filter").value)
        self.object_cluster_radius_m = float(self.get_parameter("semantic.object_cluster_radius_m").value)
        self.min_points_per_object_cluster = int(self.get_parameter("semantic.min_points_per_object_cluster").value)
        self.use_cluster_completion = bool(self.get_parameter("semantic.use_cluster_completion").value)
        self.completion_cluster_radius_m = float(self.get_parameter("semantic.completion_cluster_radius_m").value)
        self.completion_min_seed_points = int(self.get_parameter("semantic.completion_min_seed_points").value)
        self.completion_min_seed_ratio = float(self.get_parameter("semantic.completion_min_seed_ratio").value)
        self.completion_dominance_ratio = float(self.get_parameter("semantic.completion_dominance_ratio").value)
        self.completion_max_cluster_extent_m = float(self.get_parameter("semantic.completion_max_cluster_extent_m").value)
        self.use_instance_tracking = bool(self.get_parameter("semantic.use_instance_tracking").value)
        self.instance_depth_stride = max(1, int(self.get_parameter("semantic.instance_depth_stride").value))
        self.instance_min_depth_points = int(self.get_parameter("semantic.instance_min_depth_points").value)
        self.instance_bbox_padding_m = float(self.get_parameter("semantic.instance_bbox_padding_m").value)
        self.instance_match_min_iou = float(self.get_parameter("semantic.instance_match_min_iou").value)
        self.instance_match_max_centroid_distance_m = float(
            self.get_parameter("semantic.instance_match_max_centroid_distance_m").value
        )
        self.instance_max_staleness_sec = float(self.get_parameter("semantic.instance_max_staleness_sec").value)
        self.use_map_object_propagation = bool(self.get_parameter("semantic.use_map_object_propagation").value)
        self.map_object_bbox_padding_m = float(self.get_parameter("semantic.map_object_bbox_padding_m").value)
        self.map_object_min_overlap_ratio = float(self.get_parameter("semantic.map_object_min_overlap_ratio").value)
        self.map_object_max_centroid_distance_m = float(self.get_parameter("semantic.map_object_max_centroid_distance_m").value)
        self.map_object_max_cluster_extent_m = float(self.get_parameter("semantic.map_object_max_cluster_extent_m").value)
        self.voxel_size = float(self.get_parameter("semantic.voxel_size").value)
        self.min_points_per_voxel = int(self.get_parameter("semantic.min_points_per_voxel").value)
        self.raw_voxel_size = float(self.get_parameter("semantic.raw_voxel_size").value)
        self.raw_min_points_per_voxel = int(self.get_parameter("semantic.raw_min_points_per_voxel").value)
        self.cluster_radius_m = float(self.get_parameter("semantic.cluster_radius_m").value)
        self.min_voxels_per_cluster = int(self.get_parameter("semantic.min_voxels_per_cluster").value)
        self.box_padding_m = float(self.get_parameter("semantic.box_padding_m").value)
        self.box_min_size_m = float(self.get_parameter("semantic.box_min_size_m").value)
        self.map_publish_period_sec = float(self.get_parameter("semantic.map_publish_period_sec").value)
        self.publish_debug_image = bool(self.get_parameter("semantic.publish_debug_image").value)
        self.label_switch_margin_votes = int(self.get_parameter("semantic.label_switch_margin_votes").value)
        self.label_switch_ratio = float(self.get_parameter("semantic.label_switch_ratio").value)
        self.detector_model = str(self.get_parameter("semantic.detector_model").value)
        self.fallback_detector_model = str(self.get_parameter("semantic.fallback_detector_model").value)

        self.cam_width = int(self.get_parameter("camera.cam_width").value)
        self.cam_height = int(self.get_parameter("camera.cam_height").value)
        self.fx = float(self.get_parameter("camera.cam_fx").value)
        self.fy = float(self.get_parameter("camera.cam_fy").value)
        self.cx = float(self.get_parameter("camera.cam_cx").value)
        self.cy = float(self.get_parameter("camera.cam_cy").value)

        self.r_cl = np.asarray(self.get_parameter("extrin_calib.Rcl").value, dtype=np.float32).reshape(3, 3)
        self.p_cl = np.asarray(self.get_parameter("extrin_calib.Pcl").value, dtype=np.float32).reshape(3)
        self.r_li = np.asarray(self.get_parameter("extrin_calib.extrinsic_R").value, dtype=np.float32).reshape(3, 3)
        self.p_li = np.asarray(self.get_parameter("extrin_calib.extrinsic_T").value, dtype=np.float32).reshape(3)

        raw_labels = list(self.get_parameter("semantic.query_labels").value)
        self.query_labels = [str(label) for label in raw_labels if str(label).strip()]
        self.label_to_id = {label: idx + 1 for idx, label in enumerate(self.query_labels)}
        self.label_to_color = generate_palette(self.query_labels)

    def _image_callback(self, msg: Image) -> None:
        stamp = stamp_to_seconds(msg.header.stamp)
        self._latest_sensor_stamp = stamp
        image_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        detections = self.detector.detect(image_rgb)
        self._image_buffer.append(BufferedImage(stamp=stamp, image_rgb=image_rgb, detections=detections))
        if self.publish_debug_image:
            self._publish_debug_image(image_rgb, detections, msg.header)

    def _depth_callback(self, msg: Image) -> None:
        stamp = stamp_to_seconds(msg.header.stamp)
        depth_m = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._depth_buffer.append(BufferedDepth(stamp=stamp, depth_m=np.asarray(depth_m, dtype=np.float32)))

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        k = list(msg.k)
        if len(k) != 9 or k[0] <= 0.0 or k[4] <= 0.0:
            return

        changed = (
            abs(self.fx - float(k[0])) > 1e-6
            or abs(self.fy - float(k[4])) > 1e-6
            or abs(self.cx - float(k[2])) > 1e-6
            or abs(self.cy - float(k[5])) > 1e-6
            or self.cam_width != int(msg.width)
            or self.cam_height != int(msg.height)
        )

        self.fx = float(k[0])
        self.fy = float(k[4])
        self.cx = float(k[2])
        self.cy = float(k[5])
        self.cam_width = int(msg.width)
        self.cam_height = int(msg.height)

        if changed or not self._camera_info_received:
            self.get_logger().info(
                "Semantic mapper updated intrinsics from /camera_info: "
                f"fx={self.fx:.3f}, fy={self.fy:.3f}, cx={self.cx:.3f}, cy={self.cy:.3f}, "
                f"size={self.cam_width}x{self.cam_height}"
            )
        self._camera_info_received = True

    def _odom_callback(self, msg: Odometry) -> None:
        odom_stamp = self._normalize_odom_stamp(stamp_to_seconds(msg.header.stamp))
        quat = msg.pose.pose.orientation
        pose_rotation = quaternion_to_rotation_matrix(quat.x, quat.y, quat.z, quat.w)
        pose_translation = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        self._odom_buffer.append(
            BufferedOdom(
                stamp=odom_stamp,
                pose_rotation=pose_rotation,
                pose_translation=pose_translation,
            )
        )

    def _cloud_callback(self, msg: PointCloud2) -> None:
        cloud_time = stamp_to_seconds(msg.header.stamp)
        self._latest_sensor_stamp = cloud_time
        image = self._find_closest(self._image_buffer, cloud_time)
        need_depth = self.use_depth_filter or self.use_instance_tracking
        depth = self._find_closest(self._depth_buffer, cloud_time) if need_depth else None
        odom = self._find_closest(self._odom_buffer, cloud_time)

        if image is None or odom is None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_status_log_time > 2.0:
                self.get_logger().warn(
                    "Semantic mapper skipped a cloud because synchronized image/odom was not available yet."
                )
                self._last_status_log_time = now
            return

        points_lidar = self._read_pointcloud_xyz(msg)
        if points_lidar.size == 0:
            return

        ranges = np.linalg.norm(points_lidar, axis=1)
        points_lidar = points_lidar[ranges < self.max_range_m]
        if points_lidar.size == 0:
            return

        detections = image.detections
        if not detections and not (self.use_map_object_propagation and self._semantic_instances):
            return

        points_world = self._transform_points_to_world(
            points_lidar,
            odom.pose_rotation,
            odom.pose_translation,
        )
        labels, colors, assigned_mask, detection_associations = self._associate_points(
            points_lidar,
            image.image_rgb.shape,
            detections,
            depth.depth_m if depth is not None else None,
        )
        visible_instance_ids: Set[int] = set()
        if self.use_cluster_completion and np.any(assigned_mask):
            labels, colors, assigned_mask = self._complete_cluster_labels(
                points_lidar,
                labels,
                colors,
                assigned_mask,
            )

        if self.use_instance_tracking:
            visible_instance_ids = self._update_instances_from_associations(
                detection_associations,
                points_world,
                depth.depth_m if depth is not None else None,
                odom.pose_rotation,
                odom.pose_translation,
                cloud_time,
            )
            self._prune_stale_instances(cloud_time)

        if self.use_map_object_propagation and self._semantic_instances:
            labels, colors, assigned_mask = self._propagate_instance_labels(
                points_world,
                labels,
                colors,
                assigned_mask,
                visible_instance_ids,
                cloud_time,
            )
        if not np.any(assigned_mask):
            return

        semantic_points_world = points_world[assigned_mask]
        semantic_labels = labels[assigned_mask]
        semantic_colors = colors[assigned_mask]

        self._update_semantic_map(semantic_points_world, semantic_labels, semantic_colors, cloud_time)
        stable_label_ids, stable_colors = self._stabilize_current_labels(semantic_points_world, semantic_labels)
        self._publish_current_cloud(semantic_points_world, stable_label_ids, stable_colors, msg.header)

    def _find_closest(self, buffer: Deque, stamp: float):
        if not buffer:
            return None
        best = min(buffer, key=lambda item: abs(item.stamp - stamp))
        if abs(best.stamp - stamp) > self.sync_slop_sec:
            return None
        return best

    def _normalize_odom_stamp(self, raw_stamp: float) -> float:
        if self._latest_sensor_stamp is None:
            return raw_stamp

        if abs(raw_stamp - self._latest_sensor_stamp) <= self.sync_slop_sec:
            return raw_stamp

        if self._odom_time_offset is None:
            self._odom_time_offset = raw_stamp - self._latest_sensor_stamp
            self.get_logger().warn(
                "Semantic mapper detected that odom timestamps are on a different time axis. "
                f"Applying offset {self._odom_time_offset:.6f} to align odom with sensor time."
            )

        return raw_stamp - self._odom_time_offset

    def _read_pointcloud_xyz(self, msg: PointCloud2) -> np.ndarray:
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if isinstance(points, np.ndarray):
            if points.size == 0:
                return np.empty((0, 3), dtype=np.float32)
            if points.dtype.names:
                return np.column_stack((points["x"], points["y"], points["z"])).astype(np.float32, copy=False)
            return np.asarray(points, dtype=np.float32).reshape(-1, 3)

        points = list(points)
        if not points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32).reshape(-1, 3)

    def _project_lidar_points_to_camera(
        self,
        points_lidar: np.ndarray,
        image_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points_camera = points_lidar @ self.r_cl.T + self.p_cl
        z = points_camera[:, 2]
        valid = z > 1e-4

        u = self.fx * (points_camera[:, 0] / np.maximum(z, 1e-6)) + self.cx
        v = self.fy * (points_camera[:, 1] / np.maximum(z, 1e-6)) + self.cy

        height, width = image_shape[:2]
        valid &= (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        return points_camera, u, v, valid

    def _associate_points(
        self,
        points_lidar: np.ndarray,
        image_shape: Tuple[int, int, int],
        detections: Sequence[Detection],
        depth_image: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[DetectionAssociation]]:
        points_camera, u, v, valid = self._project_lidar_points_to_camera(points_lidar, image_shape)
        label_ids = np.zeros(points_lidar.shape[0], dtype=np.uint32)
        colors = np.zeros((points_lidar.shape[0], 3), dtype=np.uint8)
        assigned = np.zeros(points_lidar.shape[0], dtype=bool)
        associations: List[DetectionAssociation] = []
        u_int = np.clip(np.rint(u).astype(np.int32), 0, max(0, image_shape[1] - 1))
        v_int = np.clip(np.rint(v).astype(np.int32), 0, max(0, image_shape[0] - 1))

        for detection in detections:
            pixel_mask = self._build_detection_pixel_mask(detection, depth_image, image_shape[:2])
            candidate_mask = (
                valid
                & (~assigned)
                & (u >= detection.box[0])
                & (u <= detection.box[2])
                & (v >= detection.box[1])
                & (v <= detection.box[3])
            )
            candidate_indices = np.flatnonzero(candidate_mask)
            if pixel_mask is not None and candidate_indices.size > 0:
                pixel_hits = pixel_mask[v_int[candidate_indices], u_int[candidate_indices]]
                masked_indices = candidate_indices[pixel_hits]
                if masked_indices.size >= self.min_points_per_detection:
                    candidate_indices = masked_indices

            if candidate_indices.size >= self.min_points_per_detection:
                candidate_indices = self._filter_detection_candidates(
                    candidate_indices,
                    points_lidar,
                    points_camera[:, 2],
                    u,
                    v,
                    depth_image,
                )
            if candidate_indices.size >= self.min_points_per_detection:
                label_ids[candidate_indices] = detection.label_id
                colors[candidate_indices] = np.asarray(detection.color, dtype=np.uint8)
                assigned[candidate_indices] = True

            associations.append(
                DetectionAssociation(
                    detection=detection,
                    seed_indices=candidate_indices.astype(np.int32, copy=False),
                    pixel_mask=pixel_mask,
                )
            )

        return label_ids, colors, assigned, associations

    def _build_detection_pixel_mask(
        self,
        detection: Detection,
        depth_image: Optional[np.ndarray],
        image_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if detection.mask is not None:
            return detection.mask.astype(bool, copy=False)
        if depth_image is None:
            return None
        return self._build_depth_foreground_mask(detection.box, depth_image, image_hw)

    def _build_depth_foreground_mask(
        self,
        box: Tuple[float, float, float, float],
        depth_image: np.ndarray,
        image_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        height, width = image_hw
        if depth_image.shape[:2] != (height, width):
            depth_image = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_NEAREST)

        x1 = int(np.clip(np.floor(box[0]), 0, width - 1))
        y1 = int(np.clip(np.floor(box[1]), 0, height - 1))
        x2 = int(np.clip(np.ceil(box[2]), x1 + 1, width))
        y2 = int(np.clip(np.ceil(box[3]), y1 + 1, height))
        crop = depth_image[y1:y2, x1:x2].astype(np.float32, copy=False)
        valid = np.isfinite(crop) & (crop > 1e-4)
        if int(np.count_nonzero(valid)) < self.instance_min_depth_points:
            return None

        foreground_depth = float(np.percentile(crop[valid], 20))
        allowed_margin = max(self.depth_consistency_margin_m * 1.5, foreground_depth * 0.08)
        foreground = valid & (crop <= foreground_depth + allowed_margin)
        if int(np.count_nonzero(foreground)) < self.instance_min_depth_points:
            return None

        foreground_u8 = foreground.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_u8, connectivity=8)
        if num_labels <= 1:
            component = foreground
        else:
            center_x = int(np.clip(round(0.5 * (x2 - x1 - 1)), 0, x2 - x1 - 1))
            center_y = int(np.clip(round(0.5 * (y2 - y1 - 1)), 0, y2 - y1 - 1))
            center_label = int(labels[center_y, center_x])
            chosen_label = 0
            if center_label > 0 and stats[center_label, cv2.CC_STAT_AREA] >= self.instance_min_depth_points:
                chosen_label = center_label
            else:
                largest_area = -1
                for label_idx in range(1, num_labels):
                    area = int(stats[label_idx, cv2.CC_STAT_AREA])
                    if area > largest_area:
                        largest_area = area
                        chosen_label = label_idx
            component = labels == chosen_label

        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y1:y2, x1:x2] = component
        return full_mask

    def _complete_cluster_labels(
        self,
        points_lidar: np.ndarray,
        label_ids: np.ndarray,
        colors: np.ndarray,
        assigned: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        completed_label_ids = np.asarray(label_ids, dtype=np.uint32).copy()
        completed_colors = np.asarray(colors, dtype=np.uint8).copy()
        completed_assigned = np.asarray(assigned, dtype=bool).copy()

        for cluster in self._cluster_indices(points_lidar, self.completion_cluster_radius_m):
            if cluster.size < self.completion_min_seed_points:
                continue

            cluster_points = points_lidar[cluster]
            cluster_extent = cluster_points.max(axis=0) - cluster_points.min(axis=0)
            if float(np.max(cluster_extent)) > self.completion_max_cluster_extent_m:
                continue

            seed_mask = completed_assigned[cluster]
            seed_count = int(np.count_nonzero(seed_mask))
            if seed_count < self.completion_min_seed_points:
                continue

            seed_ratio = seed_count / float(cluster.size)
            if seed_ratio < self.completion_min_seed_ratio:
                continue

            seed_labels = completed_label_ids[cluster][seed_mask]
            label_votes = Counter(int(label_id) for label_id in seed_labels.tolist())
            dominant_label_id, dominant_votes = label_votes.most_common(1)[0]
            if dominant_votes < self.completion_min_seed_points:
                continue

            if len(label_votes) > 1:
                second_votes = label_votes.most_common(2)[1][1]
                if dominant_votes < second_votes * self.completion_dominance_ratio:
                    continue

            resolved_label = self.query_labels[int(dominant_label_id) - 1]
            resolved_color = np.asarray(self.label_to_color[resolved_label], dtype=np.uint8)
            completed_label_ids[cluster] = int(dominant_label_id)
            completed_colors[cluster] = resolved_color
            completed_assigned[cluster] = True

        return completed_label_ids, completed_colors, completed_assigned

    def _update_instances_from_associations(
        self,
        associations: Sequence[DetectionAssociation],
        points_world: np.ndarray,
        depth_image: Optional[np.ndarray],
        pose_rotation: np.ndarray,
        pose_translation: np.ndarray,
        stamp_sec: float,
    ) -> Set[int]:
        visible_instance_ids: Set[int] = set()

        for association in associations:
            observation_points = self._extract_detection_observation_points(
                association,
                points_world,
                depth_image,
                pose_rotation,
                pose_translation,
            )
            if observation_points.size == 0:
                continue

            instance_id = self._match_or_create_instance(
                association.detection.label,
                observation_points,
                stamp_sec,
            )
            visible_instance_ids.add(instance_id)

        return visible_instance_ids

    def _extract_detection_observation_points(
        self,
        association: DetectionAssociation,
        points_world: np.ndarray,
        depth_image: Optional[np.ndarray],
        pose_rotation: np.ndarray,
        pose_translation: np.ndarray,
    ) -> np.ndarray:
        point_groups: List[np.ndarray] = []

        if association.seed_indices.size > 0:
            point_groups.append(points_world[association.seed_indices])

        if depth_image is not None and association.pixel_mask is not None:
            depth_source = depth_image
            if depth_source.shape[:2] != association.pixel_mask.shape[:2]:
                depth_source = cv2.resize(
                    depth_source,
                    (association.pixel_mask.shape[1], association.pixel_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            ys, xs = np.nonzero(association.pixel_mask)
            if ys.size >= self.instance_min_depth_points:
                ys = ys[:: self.instance_depth_stride]
                xs = xs[:: self.instance_depth_stride]
                sampled_depth = depth_source[ys, xs].astype(np.float32, copy=False)
                valid = np.isfinite(sampled_depth) & (sampled_depth > 1e-4)
                if int(np.count_nonzero(valid)) >= self.instance_min_depth_points:
                    xs = xs[valid].astype(np.float32, copy=False)
                    ys = ys[valid].astype(np.float32, copy=False)
                    sampled_depth = sampled_depth[valid]
                    camera_points = np.column_stack(
                        (
                            (xs - self.cx) * sampled_depth / self.fx,
                            (ys - self.cy) * sampled_depth / self.fy,
                            sampled_depth,
                        )
                    ).astype(np.float32, copy=False)
                    point_groups.append(
                        self._transform_camera_points_to_world(
                            camera_points,
                            pose_rotation,
                            pose_translation,
                        )
                    )

        if not point_groups:
            return np.empty((0, 3), dtype=np.float32)

        observation_points = np.vstack(point_groups).astype(np.float32, copy=False)
        if observation_points.shape[0] > 4096:
            keep = np.linspace(0, observation_points.shape[0] - 1, 4096, dtype=np.int32)
            observation_points = observation_points[keep]
        return observation_points

    def _transform_camera_points_to_world(
        self,
        points_camera: np.ndarray,
        pose_rotation: np.ndarray,
        pose_translation: np.ndarray,
    ) -> np.ndarray:
        points_lidar = (points_camera - self.p_cl) @ self.r_cl
        return self._transform_points_to_world(points_lidar, pose_rotation, pose_translation)

    def _match_or_create_instance(
        self,
        label: str,
        observation_points: np.ndarray,
        stamp_sec: float,
    ) -> int:
        instance = self._select_best_instance_match(label, observation_points)
        if instance is None:
            instance = SemanticInstance(instance_id=self._next_instance_id)
            self._semantic_instances[instance.instance_id] = instance
            self._next_instance_id += 1

        self._update_instance(instance, observation_points, label, stamp_sec)
        return instance.instance_id

    def _select_best_instance_match(
        self,
        label: str,
        observation_points: np.ndarray,
    ) -> Optional[SemanticInstance]:
        if not self._semantic_instances:
            return None

        obs_min = observation_points.min(axis=0)
        obs_max = observation_points.max(axis=0)
        obs_centroid = observation_points.mean(axis=0)
        best_instance: Optional[SemanticInstance] = None
        best_score = -1e9

        for instance in self._semantic_instances.values():
            instance_label = self._resolve_instance_label(instance)
            if instance_label is not None and instance_label != label:
                continue
            if not np.all(np.isfinite(instance.bounds_min)) or not np.all(np.isfinite(instance.bounds_max)):
                continue

            padded_min = instance.bounds_min - self.instance_bbox_padding_m
            padded_max = instance.bounds_max + self.instance_bbox_padding_m
            iou = self._aabb_iou(obs_min, obs_max, padded_min, padded_max)
            centroid_delta = np.maximum(0.0, np.maximum(padded_min - obs_centroid, obs_centroid - padded_max))
            centroid_distance = float(np.linalg.norm(centroid_delta))
            if iou < self.instance_match_min_iou and centroid_distance > self.instance_match_max_centroid_distance_m:
                continue

            score = iou - 0.05 * centroid_distance + 0.01 * min(instance.point_count, 400)
            if score > best_score:
                best_score = score
                best_instance = instance

        return best_instance

    def _update_instance(
        self,
        instance: SemanticInstance,
        observation_points: np.ndarray,
        label: str,
        stamp_sec: float,
    ) -> None:
        if observation_points.size == 0:
            return

        instance.xyz_sum += observation_points.sum(axis=0, dtype=np.float64)
        instance.point_count += int(observation_points.shape[0])
        instance.bounds_min = np.minimum(instance.bounds_min, observation_points.min(axis=0))
        instance.bounds_max = np.maximum(instance.bounds_max, observation_points.max(axis=0))
        instance.label_votes[label] += int(observation_points.shape[0])
        instance.last_seen = stamp_sec
        self._refresh_instance_stable_label(instance)

    def _refresh_instance_stable_label(self, instance: SemanticInstance) -> None:
        if not instance.label_votes:
            instance.stable_label = ""
            return

        candidate_label, candidate_votes = instance.label_votes.most_common(1)[0]
        if not instance.stable_label:
            instance.stable_label = candidate_label
            return

        stable_votes = instance.label_votes.get(instance.stable_label, 0)
        if candidate_label == instance.stable_label:
            return

        required_votes = max(
            stable_votes + self.label_switch_margin_votes,
            int(np.ceil(stable_votes * self.label_switch_ratio)),
        )
        if candidate_votes >= required_votes:
            instance.stable_label = candidate_label

    def _resolve_instance_label(self, instance: SemanticInstance) -> Optional[str]:
        if instance.stable_label:
            return instance.stable_label
        if not instance.label_votes:
            return None
        return instance.label_votes.most_common(1)[0][0]

    def _prune_stale_instances(self, stamp_sec: float) -> None:
        del stamp_sec
        # Keep instances persistent for mapping. Staleness is applied only when
        # selecting propagation candidates, not by deleting map memory.
        return

    def _propagate_instance_labels(
        self,
        points_world: np.ndarray,
        label_ids: np.ndarray,
        colors: np.ndarray,
        assigned: np.ndarray,
        visible_instance_ids: Set[int],
        stamp_sec: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._semantic_instances:
            return label_ids, colors, assigned

        candidate_instances: List[SemanticInstance] = []
        if visible_instance_ids:
            for instance_id in visible_instance_ids:
                instance = self._semantic_instances.get(instance_id)
                if instance is not None:
                    candidate_instances.append(instance)
        else:
            if self.instance_max_staleness_sec <= 0.0:
                return label_ids, colors, assigned
            for instance in self._semantic_instances.values():
                if stamp_sec - instance.last_seen > self.instance_max_staleness_sec:
                    continue
                candidate_instances.append(instance)

        if not candidate_instances:
            return label_ids, colors, assigned

        propagated_label_ids = np.asarray(label_ids, dtype=np.uint32).copy()
        propagated_colors = np.asarray(colors, dtype=np.uint8).copy()
        propagated_assigned = np.asarray(assigned, dtype=bool).copy()

        for cluster in self._cluster_indices(points_world, self.completion_cluster_radius_m):
            if cluster.size < self.completion_min_seed_points:
                continue

            cluster_points = points_world[cluster]
            cluster_extent = cluster_points.max(axis=0) - cluster_points.min(axis=0)
            if float(np.max(cluster_extent)) > self.map_object_max_cluster_extent_m:
                continue

            cluster_label_votes = Counter(int(label_id) for label_id in propagated_label_ids[cluster][propagated_assigned[cluster]].tolist())
            dominant_cluster_label: Optional[str] = None
            if cluster_label_votes:
                dominant_cluster_label = self.query_labels[cluster_label_votes.most_common(1)[0][0] - 1]

            cluster_min = cluster_points.min(axis=0)
            cluster_max = cluster_points.max(axis=0)
            cluster_centroid = cluster_points.mean(axis=0)
            best_instance: Optional[SemanticInstance] = None
            best_score = -1e9

            for instance in candidate_instances:
                instance_label = self._resolve_instance_label(instance)
                if instance_label is None:
                    continue
                if dominant_cluster_label is not None and instance_label != dominant_cluster_label:
                    continue

                padded_min = instance.bounds_min - max(self.map_object_bbox_padding_m, self.instance_bbox_padding_m)
                padded_max = instance.bounds_max + max(self.map_object_bbox_padding_m, self.instance_bbox_padding_m)
                inside_mask = np.all((cluster_points >= padded_min) & (cluster_points <= padded_max), axis=1)
                overlap_ratio = float(np.count_nonzero(inside_mask)) / float(cluster.size)
                iou = self._aabb_iou(cluster_min, cluster_max, padded_min, padded_max)
                if overlap_ratio < self.map_object_min_overlap_ratio and iou < self.instance_match_min_iou:
                    continue

                centroid_delta = np.maximum(0.0, np.maximum(padded_min - cluster_centroid, cluster_centroid - padded_max))
                centroid_distance = float(np.linalg.norm(centroid_delta))
                max_centroid_distance = max(
                    self.map_object_max_centroid_distance_m,
                    self.instance_match_max_centroid_distance_m,
                )
                if centroid_distance > max_centroid_distance:
                    continue

                score = overlap_ratio + iou - 0.05 * centroid_distance + 0.01 * min(instance.point_count, 400)
                if score > best_score:
                    best_score = score
                    best_instance = instance

            if best_instance is None:
                continue

            label = self._resolve_instance_label(best_instance)
            if label is None:
                continue

            propagated_label_ids[cluster] = self.label_to_id[label]
            propagated_colors[cluster] = np.asarray(self.label_to_color[label], dtype=np.uint8)
            propagated_assigned[cluster] = True
            self._update_instance(best_instance, cluster_points, label, stamp_sec)

        return propagated_label_ids, propagated_colors, propagated_assigned

    def _aabb_iou(
        self,
        bounds_min_a: np.ndarray,
        bounds_max_a: np.ndarray,
        bounds_min_b: np.ndarray,
        bounds_max_b: np.ndarray,
    ) -> float:
        intersection_min = np.maximum(bounds_min_a, bounds_min_b)
        intersection_max = np.minimum(bounds_max_a, bounds_max_b)
        intersection_extent = np.maximum(0.0, intersection_max - intersection_min)
        intersection_volume = float(np.prod(intersection_extent))
        if intersection_volume <= 0.0:
            return 0.0

        volume_a = float(np.prod(np.maximum(0.0, bounds_max_a - bounds_min_a)))
        volume_b = float(np.prod(np.maximum(0.0, bounds_max_b - bounds_min_b)))
        union_volume = volume_a + volume_b - intersection_volume
        if union_volume <= 0.0:
            return 0.0
        return intersection_volume / union_volume

    def _filter_detection_candidates(
        self,
        candidate_indices: np.ndarray,
        points_lidar: np.ndarray,
        camera_depths: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        depth_image: Optional[np.ndarray],
    ) -> np.ndarray:
        if candidate_indices.size == 0:
            return candidate_indices

        filtered_indices = candidate_indices
        if self.use_depth_filter and depth_image is not None:
            filtered_indices = self._apply_depth_filter(filtered_indices, camera_depths, u, v, depth_image)
            if filtered_indices.size < self.min_points_per_detection:
                return filtered_indices

        if self.use_cluster_filter:
            filtered_indices = self._select_foreground_cluster(filtered_indices, points_lidar, camera_depths)

        return filtered_indices

    def _apply_depth_filter(
        self,
        candidate_indices: np.ndarray,
        camera_depths: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        depth_image: np.ndarray,
    ) -> np.ndarray:
        if depth_image.shape[:2] != (self.cam_height, self.cam_width):
            depth_image = cv2.resize(
                depth_image,
                (self.cam_width, self.cam_height),
                interpolation=cv2.INTER_NEAREST,
            )

        u_int = np.clip(np.rint(u[candidate_indices]).astype(np.int32), 0, depth_image.shape[1] - 1)
        v_int = np.clip(np.rint(v[candidate_indices]).astype(np.int32), 0, depth_image.shape[0] - 1)
        sampled_depth = depth_image[v_int, u_int].astype(np.float32, copy=False)
        valid_depth = np.isfinite(sampled_depth) & (sampled_depth > 1e-4)
        if int(np.count_nonzero(valid_depth)) < self.min_points_per_detection:
            return candidate_indices

        candidate_indices = candidate_indices[valid_depth]
        sampled_depth = sampled_depth[valid_depth]
        projected_depth = camera_depths[candidate_indices]
        allowed_margin = np.maximum(self.depth_consistency_margin_m, sampled_depth * 0.05)
        consistent = np.abs(projected_depth - sampled_depth) <= allowed_margin
        if int(np.count_nonzero(consistent)) >= self.min_points_per_detection:
            return candidate_indices[consistent]

        nearest_surface = float(np.percentile(sampled_depth, 15))
        foreground = projected_depth <= nearest_surface + self.depth_consistency_margin_m
        if int(np.count_nonzero(foreground)) >= self.min_points_per_detection:
            return candidate_indices[foreground]

        return candidate_indices

    def _select_foreground_cluster(
        self,
        candidate_indices: np.ndarray,
        points_lidar: np.ndarray,
        camera_depths: np.ndarray,
    ) -> np.ndarray:
        if candidate_indices.size <= self.min_points_per_object_cluster:
            return candidate_indices

        clusters = self._cluster_indices(points_lidar[candidate_indices], self.object_cluster_radius_m)
        large_clusters = [cluster for cluster in clusters if cluster.size >= self.min_points_per_object_cluster]
        if not large_clusters:
            return candidate_indices

        best_global_indices: Optional[np.ndarray] = None
        best_depth = None
        best_size = -1
        for local_cluster in large_clusters:
            global_cluster = candidate_indices[local_cluster]
            cluster_depth = float(np.median(camera_depths[global_cluster]))
            cluster_size = int(global_cluster.size)
            if best_global_indices is None:
                best_global_indices = global_cluster
                best_depth = cluster_depth
                best_size = cluster_size
                continue
            if cluster_depth < best_depth - self.depth_consistency_margin_m:
                best_global_indices = global_cluster
                best_depth = cluster_depth
                best_size = cluster_size
                continue
            if abs(cluster_depth - best_depth) <= self.depth_consistency_margin_m and cluster_size > best_size:
                best_global_indices = global_cluster
                best_depth = cluster_depth
                best_size = cluster_size

        return best_global_indices if best_global_indices is not None else candidate_indices

    def _transform_points_to_world(
        self,
        points_lidar: np.ndarray,
        pose_rotation: np.ndarray,
        pose_translation: np.ndarray,
    ) -> np.ndarray:
        points_imu = points_lidar @ self.r_li.T + self.p_li
        return points_imu @ pose_rotation.T + pose_translation

    def _update_semantic_map(
        self,
        points_world: np.ndarray,
        label_ids: np.ndarray,
        colors: np.ndarray,
        stamp_sec: float,
    ) -> None:
        for point, label_id, color in zip(points_world, label_ids, colors):
            label = self.query_labels[int(label_id) - 1]
            self._accumulate_voxel_observation(
                self._semantic_voxels,
                point,
                label,
                color,
                stamp_sec,
                self.voxel_size,
            )
            self._accumulate_voxel_observation(
                self._semantic_raw_voxels,
                point,
                label,
                color,
                stamp_sec,
                self.raw_voxel_size,
            )

    def _accumulate_voxel_observation(
        self,
        voxel_store: Dict[Tuple[int, int, int], SemanticVoxel],
        point: np.ndarray,
        label: str,
        color: np.ndarray,
        stamp_sec: float,
        voxel_size: float,
    ) -> None:
        voxel_key = tuple(np.floor(point / voxel_size).astype(np.int32).tolist())
        voxel = voxel_store.setdefault(voxel_key, SemanticVoxel())
        voxel.xyz_sum += point.astype(np.float64)
        voxel.point_count += 1
        voxel.color_sum += color.astype(np.float64)
        voxel.label_votes[label] += 1
        voxel.last_seen = stamp_sec
        self._refresh_voxel_stable_label(voxel)

    def _refresh_voxel_stable_label(self, voxel: SemanticVoxel) -> None:
        if not voxel.label_votes:
            voxel.stable_label = ""
            return

        candidate_label, candidate_votes = voxel.label_votes.most_common(1)[0]
        if not voxel.stable_label:
            voxel.stable_label = candidate_label
            return

        stable_votes = voxel.label_votes.get(voxel.stable_label, 0)
        if candidate_label == voxel.stable_label:
            return

        required_votes = max(
            stable_votes + self.label_switch_margin_votes,
            int(np.ceil(stable_votes * self.label_switch_ratio)),
        )
        if candidate_votes >= required_votes:
            voxel.stable_label = candidate_label

    def _resolve_voxel_label(self, voxel: SemanticVoxel) -> Optional[str]:
        if voxel.stable_label:
            return voxel.stable_label
        if not voxel.label_votes:
            return None
        return voxel.label_votes.most_common(1)[0][0]

    def _stabilize_current_labels(
        self,
        points_world: np.ndarray,
        fallback_label_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        stable_label_ids = np.asarray(fallback_label_ids, dtype=np.uint32).copy()
        stable_colors = np.zeros((points_world.shape[0], 3), dtype=np.uint8)

        for idx, point in enumerate(points_world):
            fallback_label = self.query_labels[int(stable_label_ids[idx]) - 1]
            voxel_key = tuple(np.floor(point / self.voxel_size).astype(np.int32).tolist())
            voxel = self._semantic_voxels.get(voxel_key)
            resolved_label = self._resolve_voxel_label(voxel) if voxel is not None else fallback_label
            if resolved_label is None:
                resolved_label = fallback_label
            stable_label_ids[idx] = self.label_to_id[resolved_label]
            stable_colors[idx] = np.asarray(self.label_to_color[resolved_label], dtype=np.uint8)

        return stable_label_ids, stable_colors

    def _publish_current_cloud(
        self,
        points_world: np.ndarray,
        label_ids: np.ndarray,
        colors: np.ndarray,
        source_header: Header,
    ) -> None:
        header = Header()
        header.stamp = source_header.stamp
        header.frame_id = "camera_init"
        cloud_msg = self._build_semantic_cloud(header, points_world, label_ids, colors)
        self.current_cloud_pub.publish(cloud_msg)

    def _publish_map_outputs(self) -> None:
        if not self._semantic_voxels and not self._semantic_raw_voxels:
            return

        map_points, map_label_ids, map_colors, stable_labels, voxel_weights = self._collect_voxel_cloud_entries(
            self._semantic_voxels,
            self.min_points_per_voxel,
        )
        raw_points, raw_label_ids, raw_colors, _, _ = self._collect_voxel_cloud_entries(
            self._semantic_raw_voxels,
            self.raw_min_points_per_voxel,
        )

        if not map_points and not raw_points:
            return

        now = self.get_clock().now().to_msg()
        header = Header(stamp=now, frame_id="camera_init")
        if map_points:
            map_cloud = self._build_semantic_cloud(
                header,
                np.asarray(map_points, dtype=np.float32),
                np.asarray(map_label_ids, dtype=np.uint32),
                np.asarray(map_colors, dtype=np.uint8),
            )
            self.map_cloud_pub.publish(map_cloud)
            self.marker_pub.publish(
                self._build_markers(
                    np.asarray(map_points, dtype=np.float32),
                    stable_labels,
                    voxel_weights,
                    header,
                )
            )

        if raw_points:
            raw_map_cloud = self._build_semantic_cloud(
                header,
                np.asarray(raw_points, dtype=np.float32),
                np.asarray(raw_label_ids, dtype=np.uint32),
                np.asarray(raw_colors, dtype=np.uint8),
            )
            self.raw_map_cloud_pub.publish(raw_map_cloud)

    def _collect_voxel_cloud_entries(
        self,
        voxel_store: Dict[Tuple[int, int, int], SemanticVoxel],
        min_points_per_voxel: int,
    ) -> Tuple[
        List[Tuple[float, float, float]],
        List[int],
        List[Tuple[int, int, int]],
        List[str],
        List[int],
    ]:
        points: List[Tuple[float, float, float]] = []
        label_ids: List[int] = []
        colors: List[Tuple[int, int, int]] = []
        stable_labels: List[str] = []
        voxel_weights: List[int] = []

        for voxel in voxel_store.values():
            if voxel.point_count < min_points_per_voxel or not voxel.label_votes:
                continue
            point = (voxel.xyz_sum / float(voxel.point_count)).astype(np.float32)
            label = self._resolve_voxel_label(voxel)
            if label is None:
                continue
            label_id = self.label_to_id[label]
            color = self.label_to_color[label]
            points.append((float(point[0]), float(point[1]), float(point[2])))
            label_ids.append(label_id)
            colors.append(color)
            stable_labels.append(label)
            voxel_weights.append(voxel.point_count)

        return points, label_ids, colors, stable_labels, voxel_weights

    def _build_semantic_cloud(
        self,
        header: Header,
        points: np.ndarray,
        label_ids: np.ndarray,
        colors: np.ndarray,
    ) -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="label", offset=16, datatype=PointField.UINT32, count=1),
        ]
        rows = []
        for point, label_id, color in zip(points, label_ids, colors):
            rows.append((float(point[0]), float(point[1]), float(point[2]), pack_rgb(color), int(label_id)))
        return point_cloud2.create_cloud(header, fields, rows)

    def _build_semantic_objects(self) -> List[SemanticObject]:
        label_points: Dict[str, List[np.ndarray]] = defaultdict(list)

        for voxel in self._semantic_voxels.values():
            if voxel.point_count < self.min_points_per_voxel or not voxel.label_votes:
                continue
            label = self._resolve_voxel_label(voxel)
            if label is None:
                continue
            label_points[label].append((voxel.xyz_sum / float(voxel.point_count)).astype(np.float32))

        semantic_objects: List[SemanticObject] = []
        for label, points in label_points.items():
            label_array = np.asarray(points, dtype=np.float32)
            for cluster in self._cluster_indices(label_array, self.cluster_radius_m):
                if cluster.size < self.min_voxels_per_cluster:
                    continue
                cluster_points = label_array[cluster]
                semantic_objects.append(
                    SemanticObject(
                        label=label,
                        centroid=cluster_points.mean(axis=0),
                        bounds_min=cluster_points.min(axis=0),
                        bounds_max=cluster_points.max(axis=0),
                        voxel_count=int(cluster.size),
                    )
                )

        return semantic_objects

    def _build_markers(
        self,
        points: np.ndarray,
        stable_labels: Sequence[str],
        voxel_weights: Sequence[int],
        header: Header,
    ) -> MarkerArray:
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        if points.size == 0:
            return marker_array

        marker_id = 0
        for cluster in self._cluster_indices(points, self.cluster_radius_m):
            if cluster.size < self.min_voxels_per_cluster:
                continue

            cluster_points = points[cluster]
            cluster_label_votes: Counter = Counter()
            for point_idx in cluster:
                cluster_label_votes[stable_labels[int(point_idx)]] += max(1, int(voxel_weights[int(point_idx)]))

            label, _ = cluster_label_votes.most_common(1)[0]
            color = self.label_to_color[label]
            bounds_min = cluster_points.min(axis=0)
            bounds_max = cluster_points.max(axis=0)
            centroid = 0.5 * (bounds_min + bounds_max)
            size = np.maximum(
                (bounds_max - bounds_min) + 2.0 * self.box_padding_m,
                self.box_min_size_m,
            )

            box_marker = Marker()
            box_marker.header = header
            box_marker.ns = "semantic_boxes"
            box_marker.id = marker_id
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position = Point(
                x=float(centroid[0]),
                y=float(centroid[1]),
                z=float(centroid[2]),
            )
            box_marker.pose.orientation.w = 1.0
            box_marker.scale.x = float(size[0])
            box_marker.scale.y = float(size[1])
            box_marker.scale.z = float(size[2])
            box_marker.color.r = color[0] / 255.0
            box_marker.color.g = color[1] / 255.0
            box_marker.color.b = color[2] / 255.0
            box_marker.color.a = 0.22
            box_marker.lifetime.sec = 0
            marker_array.markers.append(box_marker)
            marker_id += 1

            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "semantic_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = Point(
                x=float(centroid[0]),
                y=float(centroid[1]),
                z=float(bounds_max[2] + self.box_padding_m + 0.18),
            )
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.28
            text_marker.color.r = color[0] / 255.0
            text_marker.color.g = color[1] / 255.0
            text_marker.color.b = color[2] / 255.0
            text_marker.color.a = 0.95
            text_marker.text = label
            text_marker.lifetime.sec = 0
            marker_array.markers.append(text_marker)
            marker_id += 1

        return marker_array

    def _cluster_label_points(self, points: np.ndarray) -> List[np.ndarray]:
        return [points[cluster] for cluster in self._cluster_indices(points, self.cluster_radius_m)]

    def _cluster_indices(self, points: np.ndarray, radius_m: float) -> List[np.ndarray]:
        if points.size == 0:
            return []
        if points.shape[0] == 1:
            return [np.asarray([0], dtype=np.int32)]

        remaining = set(range(points.shape[0]))
        clusters: List[np.ndarray] = []
        radius_sq = float(radius_m * radius_m)

        while remaining:
            seed = remaining.pop()
            component = [seed]
            queue = [seed]
            while queue:
                current = queue.pop()
                if not remaining:
                    continue
                candidate_indices = np.fromiter(remaining, dtype=np.int32)
                deltas = points[candidate_indices] - points[current]
                neighbor_indices = candidate_indices[np.sum(deltas * deltas, axis=1) <= radius_sq]
                if neighbor_indices.size == 0:
                    continue
                queue.extend(int(idx) for idx in neighbor_indices)
                component.extend(int(idx) for idx in neighbor_indices)
                remaining.difference_update(int(idx) for idx in neighbor_indices)

            clusters.append(np.asarray(component, dtype=np.int32))

        return clusters

    def _publish_debug_image(self, image_rgb: np.ndarray, detections: Sequence[Detection], source_header: Header) -> None:
        annotated = image_rgb.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(v) for v in detection.box]
            color = tuple(int(v) for v in detection.color)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            caption = f"{detection.label} {detection.score:.2f}"
            cv2.putText(
                annotated,
                caption,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding="rgb8")
        msg.header = source_header
        self.debug_image_pub.publish(msg)


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node = SemanticMapperNode()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
