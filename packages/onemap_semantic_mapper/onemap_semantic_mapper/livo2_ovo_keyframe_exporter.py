from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2

from .data_types import KeyframePacket, LocalCloudPacket, SensorConfig
from .io.keyframe_manifest import append_keyframe_packet
from .io.local_cloud_manifest import append_local_cloud_packet
from .io.sensor_config import write_sensor_config
from .yoloe_helper_client import YOLOEHelperClient


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    m = rotation.astype(np.float64, copy=False)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, ratio: float) -> np.ndarray:
    q0 = q0.astype(np.float64, copy=False)
    q1 = q1.astype(np.float64, copy=False)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + ratio * (q1 - q0)
        return (result / np.linalg.norm(result)).astype(np.float32)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * ratio
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0 + s1 * q1).astype(np.float32)


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r_delta = r_a.T @ r_b
    cos_theta = (np.trace(r_delta) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def blur_score(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def coverage_novelty(depth_m: np.ndarray, prev_grid: Optional[np.ndarray], grid_shape: tuple[int, int]) -> tuple[float, np.ndarray]:
    valid = np.isfinite(depth_m) & (depth_m > 0.05)
    grid = cv2.resize(valid.astype(np.uint8), grid_shape, interpolation=cv2.INTER_AREA) > 0
    current_cells = int(grid.sum())
    if prev_grid is None or current_cells == 0:
        return (1.0 if current_cells > 0 else 0.0), grid
    new_cells = int(np.logical_and(grid, np.logical_not(prev_grid)).sum())
    return float(new_cells / max(current_cells, 1)), grid


def pointcloud_xyz_array(msg: PointCloud2) -> np.ndarray:
    raw_points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    if isinstance(raw_points, np.ndarray):
        arr = raw_points
    else:
        arr = np.asarray(list(raw_points))

    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if arr.dtype.fields:
        return np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)

    return np.asarray(arr, dtype=np.float32).reshape(-1, 3)


@dataclass
class BufferedImage:
    stamp: float
    image_rgb: np.ndarray


@dataclass
class BufferedDepth:
    stamp: float
    depth_m: np.ndarray


@dataclass
class BufferedOdom:
    stamp: float
    rotation: np.ndarray
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass
class BufferedCloud:
    stamp: float
    scan_id: str
    frame_id: str
    xyz: np.ndarray


@dataclass
class BufferedCloudSubmap:
    anchor_stamp: float
    stamp_start: float
    stamp_end: float
    frame_id: str
    xyz: np.ndarray
    source_scan_ids: list[str]
    source_point_indices: np.ndarray | None = None
    cloud_kind: str = "generic"
    quality_fields_present: bool = False
    uv: np.ndarray | None = None
    projected_depth: np.ndarray | None = None
    depth_residual: np.ndarray | None = None
    zbuffer_rank: np.ndarray | None = None
    distance_to_depth_edge: np.ndarray | None = None
    visibility_score: np.ndarray | None = None
    quality_score: np.ndarray | None = None


class Livo2OVOKeyframeExporter(Node):
    def __init__(self, node_name: str = "livo2_ovo_keyframe_exporter") -> None:
        super().__init__(node_name)
        self.bridge = CvBridge()

        self._declare_parameters()
        self._load_parameters()

        self._image_buffer: Deque[BufferedImage] = deque(maxlen=300)
        self._depth_buffer: Deque[BufferedDepth] = deque(maxlen=300)
        self._odom_buffer: Deque[BufferedOdom] = deque(maxlen=1000)
        self._camera_pose_buffer: Deque[BufferedOdom] = deque(maxlen=1000)
        self._cloud_buffer: Deque[BufferedCloud] = deque(maxlen=120)
        self._semantic_cloud_buffer: Deque[BufferedCloud] = deque(maxlen=120)

        self._camera_info_received = False
        self._camera_info_warning_emitted = False
        self._config_written = False
        self._latest_sensor_stamp: Optional[float] = None
        self._latest_odom_stamp: Optional[float] = None
        self._latest_camera_pose_stamp: Optional[float] = None
        self._odom_time_offset: Optional[float] = None
        self._stopped_for_max_frames = False
        self._last_wait_log_sec = 0.0
        self._last_processed_candidate_stamp: Optional[float] = None
        self._last_keyframe_stamp: Optional[float] = None
        self._last_keyframe_c2w: Optional[np.ndarray] = None
        self._last_keyframe_coverage: Optional[np.ndarray] = None
        self._last_semantic_trigger_stamp: Optional[float] = None
        self._frame_index = 0
        self._saved_frame_count = 0
        self._synced_frame_count = 0
        self._num_rgb_frames_seen = 0
        self._num_depth_frames_seen = 0
        self._num_poses_received = 0
        self._trajectory_dt_samples: list[float] = []
        self._rgb_dt_samples: list[float] = []
        self._depth_dt_samples: list[float] = []
        self._aligned_pose_dt_samples: list[float] = []
        self._coverage_trigger_count = 0
        self._translation_trigger_count = 0
        self._rotation_trigger_count = 0
        self._time_trigger_count = 0
        self._num_pose_missing = 0
        self._num_quality_rejected = 0
        self._num_frames_interpolated = 0
        self._num_frames_exact_match = 0
        self._num_frames_monotonic_rejected = 0
        self._num_pose_axis_wait_drops = 0
        self._num_pose_non_monotonic = 0
        self._selection_reasons: Counter[str] = Counter()
        self._direct_pose_to_odom_world: Optional[np.ndarray] = None
        self._direct_pose_to_odom_world_stamp: Optional[float] = None
        self._direct_pose_to_odom_world_pose_dt: Optional[float] = None
        self._direct_pose_to_odom_world_match_type: str = "missing"

        self.scene_dir = self.output_root / self.scene_name
        self.results_dir = self.scene_dir / "results"
        self.local_clouds_dir = self.scene_dir / "local_clouds"
        self.sensor_config_path = self.scene_dir / "sensor_config.yaml"
        self.scene_config_path = self.config_root / f"{self.scene_name}.yaml"
        self.metadata_path = self.scene_dir / "export_info.yaml"
        self.traj_path = self.scene_dir / "traj.txt"
        self.frame_index_path = self.scene_dir / "frames_index.csv"

        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.scene_name}"
        self.run_dir = self.run_root / self.run_id
        self.export_dir = self.run_dir / "export"
        self.thumbnails_dir = self.export_dir / "thumbnails"
        self.export_results_dir = self.export_dir / "results"
        self.export_local_clouds_dir = self.export_dir / "local_clouds"
        self.keyframe_packets_path = self.export_dir / "keyframe_packets.jsonl"
        self.local_cloud_packets_path = self.export_dir / "local_cloud_packets.jsonl"
        self.export_session_path = self.export_dir / "session.json"

        self._prepare_output_paths()
        self._traj_file = self.traj_path.open("a", encoding="utf-8")
        self._frame_index_file = self.frame_index_path.open("a", encoding="utf-8", newline="")
        self._frame_index_writer = csv.writer(self._frame_index_file)
        self._trajectory_file = (self.run_dir / "trajectory_raw.jsonl").open("a", encoding="utf-8")
        self._aligned_file = (self.run_dir / "aligned_frames.jsonl").open("a", encoding="utf-8")
        self._accepted_file = (self.run_dir / "accepted_frames.jsonl").open("a", encoding="utf-8")
        self._keyframes_file = (self.export_dir / "keyframes.jsonl").open("a", encoding="utf-8")

        self._ensure_frame_index_header()
        self._write_metadata()
        self._write_contract_check()
        self._write_manifest()
        self._write_export_session()

        self._dynamic_semantic_client: YOLOEHelperClient | None = None
        self._last_dynamic_target_seen_stamp: Optional[float] = None
        self._last_dynamic_persist_stamp: Optional[float] = None
        if self.dynamic_semantic_persistence_enabled and self.dynamic_semantic_target_labels:
            self._dynamic_semantic_client = YOLOEHelperClient(
                helper_script=self.dynamic_semantic_yolo_helper_script,
                repo_root=self.dynamic_semantic_yolo_repo_root,
                conda_env=self.dynamic_semantic_yolo_conda_env,
                python_bin=self.dynamic_semantic_yolo_python_bin,
                cuda_visible_devices=self.dynamic_semantic_yolo_cuda_visible_devices,
                model_path=self.dynamic_semantic_yolo_model_path,
                device=self.dynamic_semantic_yolo_device,
                conf_thresh=self.dynamic_semantic_yolo_conf_thresh,
                iou_thresh=self.dynamic_semantic_yolo_iou_thresh,
                max_det=self.dynamic_semantic_yolo_max_det,
                topk_labels=1,
            )

        self.image_sub = self.create_subscription(Image, self.image_topic, self._image_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self._depth_callback, qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 100)
        # Isaac ROS2 pose publishers commonly use sensor-data QoS (best effort).
        # Match that profile so exact image-time GT pose can be received without
        # reliability incompatibilities.
        self.camera_pose_sub = self.create_subscription(
            Odometry,
            self.camera_pose_topic,
            self._camera_pose_callback,
            qos_profile_sensor_data,
        )
        self.cloud_sub = None
        if not self.use_rgbd_local_cloud:
            self.cloud_sub = self.create_subscription(
                PointCloud2,
                self.local_cloud_topic,
                self._cloud_callback,
                qos_profile_sensor_data,
            )
        self.semantic_cloud_sub = None
        if self.semantic_trusted_cloud_enabled:
            self.semantic_cloud_sub = self.create_subscription(
                PointCloud2,
                self.semantic_trusted_cloud_topic,
                self._semantic_cloud_callback,
                qos_profile_sensor_data,
            )

        self.get_logger().info(
            "LIVO2 OVO keyframe exporter ready. "
            f"scene={self.scene_name}, output={self.scene_dir}, run={self.run_dir}, "
            f"image={self.image_topic}, depth={self.depth_topic}, odom={self.odom_topic}, "
            f"camera_pose={self.camera_pose_topic}, "
            f"local_cloud_mode={'rgbd' if self.use_rgbd_local_cloud else 'cloud_topic'}, "
            f"trusted_semantic_cloud={self.semantic_trusted_cloud_topic if self.semantic_trusted_cloud_enabled else 'disabled'}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "common.img_topic": "/robot_rgb",
            "extrin_calib.Rcl": [0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
            "extrin_calib.Pcl": [0.0, 0.0, 0.0],
            "extrin_calib.extrinsic_T": [0.0, 0.0, 0.790440008],
            "extrin_calib.extrinsic_R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "camera.cam_width": 1280,
            "camera.cam_height": 720,
            "camera.cam_fx": 640.0,
            "camera.cam_fy": 640.0,
            "camera.cam_cx": 640.0,
            "camera.cam_cy": 360.0,
            "export.depth_topic": "/depth",
            "export.camera_info_topic": "/camera_info",
            "export.override_intrinsics_from_camera_info": False,
            "export.odom_topic": "/aft_mapped_to_init",
            "export.camera_pose_topic": "/camera_pose_at_image",
            "export.require_direct_camera_pose": False,
            "export.require_exact_direct_camera_pose": False,
            "export.direct_camera_pose_exact_tolerance_sec": 1.0e-4,
            "export.direct_camera_pose_apply_isaac_optical_fix": False,
            "export.align_direct_camera_pose_to_odom_world": False,
            "export.align_direct_camera_pose_to_odom_max_dt_sec": 0.05,
            "export.require_exact_rgbd_sync": False,
            "export.rgbd_exact_tolerance_sec": 1.0e-4,
            "export.output_root": "/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica",
            "export.config_root": "/home/peng/isacc_slam/reference/OVO/data/working/configs/Replica",
            "export.run_root": "/home/peng/isacc_slam/runs/ovo_pose_keyframes",
            "export.scene_name": "isaac_turtlebot3_livo2",
            "export.local_cloud_topic": "/cloud_registered",
            "export.local_cloud_frame": "world",
            "export.use_rgbd_local_cloud": False,
            "export.rgbd_stride": 4,
            "export.rgbd_min_depth_m": 0.05,
            "export.local_cloud_sync_slop_sec": 0.20,
            "export.local_cloud_window_sec": 0.0,
            "export.local_cloud_min_scans": 1,
            "export.local_cloud_max_scans": 1,
            "export.local_cloud_voxel_size_m": 0.05,
            "export.semantic_local_cloud_enabled": True,
            "export.semantic_trusted_cloud_enabled": True,
            "export.semantic_trusted_cloud_topic": "/cloud_visual_sub_map_before",
            "export.semantic_trusted_cloud_sync_slop_sec": 0.20,
            "export.semantic_trusted_point_radius_m": 0.12,
            "export.semantic_trusted_min_support_ratio": 0.15,
            "export.semantic_local_cloud_depth_tolerance_m": 0.03,
            "export.semantic_local_cloud_min_points": 16,
            "export.semantic_local_cloud_min_edge_distance_px": 1.25,
            "export.semantic_local_cloud_min_quality_score": 0.35,
            "export.sensor_config_id": "fast_livo2_isaac_rgbd_livox",
            "export.calib_version": "camera_raised_0p61952",
            "export.sync_slop_sec": 0.12,
            "export.depth_input_scale": 1000.0,
            "export.output_depth_scale": 1000.0,
            "export.max_depth_m": 8.0,
            "export.jpeg_quality": 95,
            "export.max_frames": -1,
            "export.overwrite_scene": True,
            "export.dynamic_semantic_persistence_enabled": False,
            "export.dynamic_semantic_target_labels": "",
            "export.dynamic_semantic_min_detections": 1,
            "export.dynamic_semantic_min_score": 0.10,
            "export.dynamic_semantic_keepalive_sec": 1.5,
            "export.dynamic_semantic_fallback_sec": 0.0,
            "export.dynamic_semantic_yolo_helper_script": "/home/peng/isacc_slam/src/onemap_semantic_mapper/scripts/run_yoloe26x_region_infer.py",
            "export.dynamic_semantic_yolo_repo_root": "/home/peng/isacc_slam/reference/YOLOE_official",
            "export.dynamic_semantic_yolo_conda_env": "yoloe_env",
            "export.dynamic_semantic_yolo_python_bin": "",
            "export.dynamic_semantic_yolo_cuda_visible_devices": "1",
            "export.dynamic_semantic_yolo_model_path": "yoloe-26x-seg.pt",
            "export.dynamic_semantic_yolo_device": "cuda",
            "export.dynamic_semantic_yolo_conf_thresh": 0.10,
            "export.dynamic_semantic_yolo_iou_thresh": 0.50,
            "export.dynamic_semantic_yolo_max_det": 100,
            "selector.min_time_gap_sec": 0.25,
            "selector.max_time_gap_sec": 1.2,
            "selector.translation_thresh_m": 0.12,
            "selector.rotation_thresh_deg": 12.0,
            "selector.depth_valid_ratio_thresh": 0.55,
            "selector.blur_score_min": 30.0,
            "selector.coverage_novelty_thresh": 0.18,
            "selector.coverage_grid_width": 24,
            "selector.coverage_grid_height": 18,
            "selector.force_every_synced_frame": False,
            "selector.min_translation_for_rotation_trigger_m": 0.03,
            "selector.min_translation_for_time_trigger_m": 0.05,
            "selector.semantic_trigger_enabled": True,
            "selector.semantic_status_path": "",
            "selector.semantic_object_memory_path": "",
            "selector.semantic_trigger_min_translation_m": 0.03,
            "selector.semantic_trigger_cooldown_sec": 1.5,
            "selector.semantic_pending_object_threshold": 1,
            "selector.semantic_recent_object_window_sec": 4.0,
            "selector.semantic_under_supported_max_support_points": 48,
            "selector.semantic_under_supported_max_observations": 2,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.image_topic = str(self.get_parameter("common.img_topic").value)
        self.depth_topic = str(self.get_parameter("export.depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("export.camera_info_topic").value)
        self.override_intrinsics_from_camera_info = bool(
            self.get_parameter("export.override_intrinsics_from_camera_info").value
        )
        self.odom_topic = str(self.get_parameter("export.odom_topic").value)
        self.camera_pose_topic = str(self.get_parameter("export.camera_pose_topic").value)
        self.require_direct_camera_pose = bool(self.get_parameter("export.require_direct_camera_pose").value)
        self.require_exact_direct_camera_pose = bool(
            self.get_parameter("export.require_exact_direct_camera_pose").value
        )
        self.direct_camera_pose_exact_tolerance_sec = float(
            self.get_parameter("export.direct_camera_pose_exact_tolerance_sec").value
        )
        self.direct_camera_pose_apply_isaac_optical_fix = bool(
            self.get_parameter("export.direct_camera_pose_apply_isaac_optical_fix").value
        )
        self.align_direct_camera_pose_to_odom_world = bool(
            self.get_parameter("export.align_direct_camera_pose_to_odom_world").value
        )
        self.align_direct_camera_pose_to_odom_max_dt_sec = float(
            self.get_parameter("export.align_direct_camera_pose_to_odom_max_dt_sec").value
        )
        self.require_exact_rgbd_sync = bool(self.get_parameter("export.require_exact_rgbd_sync").value)
        self.rgbd_exact_tolerance_sec = float(self.get_parameter("export.rgbd_exact_tolerance_sec").value)
        self.output_root = Path(str(self.get_parameter("export.output_root").value)).expanduser()
        self.config_root = Path(str(self.get_parameter("export.config_root").value)).expanduser()
        self.run_root = Path(str(self.get_parameter("export.run_root").value)).expanduser()
        self.scene_name = str(self.get_parameter("export.scene_name").value)
        self.local_cloud_topic = str(self.get_parameter("export.local_cloud_topic").value)
        self.local_cloud_frame = str(self.get_parameter("export.local_cloud_frame").value)
        self.use_rgbd_local_cloud = bool(self.get_parameter("export.use_rgbd_local_cloud").value)
        self.rgbd_stride = max(1, int(self.get_parameter("export.rgbd_stride").value))
        self.rgbd_min_depth_m = max(0.0, float(self.get_parameter("export.rgbd_min_depth_m").value))
        self.local_cloud_sync_slop_sec = float(self.get_parameter("export.local_cloud_sync_slop_sec").value)
        self.local_cloud_window_sec = float(self.get_parameter("export.local_cloud_window_sec").value)
        self.local_cloud_min_scans = int(self.get_parameter("export.local_cloud_min_scans").value)
        self.local_cloud_max_scans = int(self.get_parameter("export.local_cloud_max_scans").value)
        self.local_cloud_voxel_size_m = float(self.get_parameter("export.local_cloud_voxel_size_m").value)
        self.semantic_local_cloud_enabled = bool(self.get_parameter("export.semantic_local_cloud_enabled").value)
        self.semantic_trusted_cloud_enabled = bool(self.get_parameter("export.semantic_trusted_cloud_enabled").value)
        self.semantic_trusted_cloud_topic = str(self.get_parameter("export.semantic_trusted_cloud_topic").value)
        self.semantic_trusted_cloud_sync_slop_sec = float(
            self.get_parameter("export.semantic_trusted_cloud_sync_slop_sec").value
        )
        self.semantic_trusted_point_radius_m = float(self.get_parameter("export.semantic_trusted_point_radius_m").value)
        self.semantic_trusted_min_support_ratio = float(
            self.get_parameter("export.semantic_trusted_min_support_ratio").value
        )
        self.semantic_local_cloud_depth_tolerance_m = float(
            self.get_parameter("export.semantic_local_cloud_depth_tolerance_m").value
        )
        self.semantic_local_cloud_min_points = int(self.get_parameter("export.semantic_local_cloud_min_points").value)
        self.semantic_local_cloud_min_edge_distance_px = float(
            self.get_parameter("export.semantic_local_cloud_min_edge_distance_px").value
        )
        self.semantic_local_cloud_min_quality_score = float(
            self.get_parameter("export.semantic_local_cloud_min_quality_score").value
        )
        self.sensor_config_id = str(self.get_parameter("export.sensor_config_id").value)
        self.calib_version = str(self.get_parameter("export.calib_version").value)
        self.sync_slop_sec = float(self.get_parameter("export.sync_slop_sec").value)
        self.depth_input_scale = float(self.get_parameter("export.depth_input_scale").value)
        self.output_depth_scale = float(self.get_parameter("export.output_depth_scale").value)
        self.max_depth_m = float(self.get_parameter("export.max_depth_m").value)
        self.jpeg_quality = int(self.get_parameter("export.jpeg_quality").value)
        self.max_frames = int(self.get_parameter("export.max_frames").value)
        self.overwrite_scene = bool(self.get_parameter("export.overwrite_scene").value)
        self.dynamic_semantic_persistence_enabled = bool(
            self.get_parameter("export.dynamic_semantic_persistence_enabled").value
        )
        raw_dynamic_target_labels = str(self.get_parameter("export.dynamic_semantic_target_labels").value).strip()
        self.dynamic_semantic_target_labels = [
            token.strip().lower()
            for token in raw_dynamic_target_labels.replace(";", ",").split(",")
            if token.strip()
        ]
        self.dynamic_semantic_min_detections = max(
            1, int(self.get_parameter("export.dynamic_semantic_min_detections").value)
        )
        self.dynamic_semantic_min_score = float(
            self.get_parameter("export.dynamic_semantic_min_score").value
        )
        self.dynamic_semantic_keepalive_sec = float(
            self.get_parameter("export.dynamic_semantic_keepalive_sec").value
        )
        self.dynamic_semantic_fallback_sec = float(
            self.get_parameter("export.dynamic_semantic_fallback_sec").value
        )
        self.dynamic_semantic_yolo_helper_script = Path(
            str(self.get_parameter("export.dynamic_semantic_yolo_helper_script").value)
        ).expanduser()
        self.dynamic_semantic_yolo_repo_root = Path(
            str(self.get_parameter("export.dynamic_semantic_yolo_repo_root").value)
        ).expanduser()
        self.dynamic_semantic_yolo_conda_env = str(
            self.get_parameter("export.dynamic_semantic_yolo_conda_env").value
        ).strip()
        self.dynamic_semantic_yolo_python_bin = str(
            self.get_parameter("export.dynamic_semantic_yolo_python_bin").value
        ).strip()
        self.dynamic_semantic_yolo_cuda_visible_devices = str(
            self.get_parameter("export.dynamic_semantic_yolo_cuda_visible_devices").value
        ).strip()
        self.dynamic_semantic_yolo_model_path = str(
            self.get_parameter("export.dynamic_semantic_yolo_model_path").value
        ).strip()
        self.dynamic_semantic_yolo_device = str(
            self.get_parameter("export.dynamic_semantic_yolo_device").value
        ).strip()
        self.dynamic_semantic_yolo_conf_thresh = float(
            self.get_parameter("export.dynamic_semantic_yolo_conf_thresh").value
        )
        self.dynamic_semantic_yolo_iou_thresh = float(
            self.get_parameter("export.dynamic_semantic_yolo_iou_thresh").value
        )
        self.dynamic_semantic_yolo_max_det = int(
            self.get_parameter("export.dynamic_semantic_yolo_max_det").value
        )

        self.min_time_gap_sec = float(self.get_parameter("selector.min_time_gap_sec").value)
        self.max_time_gap_sec = float(self.get_parameter("selector.max_time_gap_sec").value)
        self.translation_thresh_m = float(self.get_parameter("selector.translation_thresh_m").value)
        self.rotation_thresh_deg = float(self.get_parameter("selector.rotation_thresh_deg").value)
        self.depth_valid_ratio_thresh = float(self.get_parameter("selector.depth_valid_ratio_thresh").value)
        self.blur_score_min = float(self.get_parameter("selector.blur_score_min").value)
        self.coverage_novelty_thresh = float(self.get_parameter("selector.coverage_novelty_thresh").value)
        self.coverage_grid_shape = (
            int(self.get_parameter("selector.coverage_grid_width").value),
            int(self.get_parameter("selector.coverage_grid_height").value),
        )
        self.force_every_synced_frame = bool(self.get_parameter("selector.force_every_synced_frame").value)
        self.min_translation_for_rotation_trigger_m = float(
            self.get_parameter("selector.min_translation_for_rotation_trigger_m").value
        )
        self.min_translation_for_time_trigger_m = float(
            self.get_parameter("selector.min_translation_for_time_trigger_m").value
        )
        self.semantic_trigger_enabled = bool(self.get_parameter("selector.semantic_trigger_enabled").value)
        semantic_status_path = str(self.get_parameter("selector.semantic_status_path").value).strip()
        semantic_object_memory_path = str(self.get_parameter("selector.semantic_object_memory_path").value).strip()
        self.semantic_status_path = Path(semantic_status_path).expanduser() if semantic_status_path else None
        self.semantic_object_memory_path = (
            Path(semantic_object_memory_path).expanduser() if semantic_object_memory_path else None
        )
        self.semantic_trigger_min_translation_m = float(
            self.get_parameter("selector.semantic_trigger_min_translation_m").value
        )
        self.semantic_trigger_cooldown_sec = float(self.get_parameter("selector.semantic_trigger_cooldown_sec").value)
        self.semantic_pending_object_threshold = int(
            self.get_parameter("selector.semantic_pending_object_threshold").value
        )
        self.semantic_recent_object_window_sec = float(
            self.get_parameter("selector.semantic_recent_object_window_sec").value
        )
        self.semantic_under_supported_max_support_points = int(
            self.get_parameter("selector.semantic_under_supported_max_support_points").value
        )
        self.semantic_under_supported_max_observations = int(
            self.get_parameter("selector.semantic_under_supported_max_observations").value
        )

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

    def _prepare_output_paths(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.export_results_dir.mkdir(parents=True, exist_ok=True)
        self.export_local_clouds_dir.mkdir(parents=True, exist_ok=True)

        if self.overwrite_scene and self.scene_dir.exists():
            for path in [
                self.results_dir,
                self.local_clouds_dir,
                self.traj_path,
                self.metadata_path,
                self.frame_index_path,
                self.sensor_config_path,
            ]:
                if path.is_dir():
                    for child in path.glob("*"):
                        if child.is_file():
                            child.unlink()
                    path.rmdir()
                elif path.exists():
                    path.unlink()

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.local_clouds_dir.mkdir(parents=True, exist_ok=True)

    def _load_optional_json(self, path: Path | None) -> dict | None:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _count_support_points(self, object_payload: dict) -> int:
        fused_voxel_count = object_payload.get("fused_voxel_count")
        if fused_voxel_count is not None:
            try:
                return int(fused_voxel_count)
            except Exception:
                pass

        geometry_store_path = object_payload.get("geometry_store_path")
        if geometry_store_path:
            try:
                geometry_path = Path(str(geometry_store_path)).expanduser()
                if not geometry_path.is_absolute() and self.semantic_object_memory_path is not None:
                    geometry_path = self.semantic_object_memory_path.parent / geometry_path
                if geometry_path.exists():
                    with np.load(geometry_path, allow_pickle=False) as data:
                        if "voxel_keys" in data:
                            return int(np.asarray(data["voxel_keys"]).shape[0])
                        if "xyz_mean" in data:
                            return int(np.asarray(data["xyz_mean"]).shape[0])
            except Exception:
                pass

        total = 0
        for support_ref in object_payload.get("point_support_refs", []):
            if not isinstance(support_ref, list) or len(support_ref) != 2:
                continue
            total += len(support_ref[1])
        return int(total)

    def _iter_semantic_objects(self, object_memory_payload: dict | None) -> list[dict]:
        if object_memory_payload is None:
            return []
        if not isinstance(object_memory_payload, dict):
            return []
        raw_objects = object_memory_payload.get("objects", object_memory_payload)
        if not isinstance(raw_objects, dict):
            return []
        return [obj for obj in raw_objects.values() if isinstance(obj, dict)]

    def _semantic_object_top_label(self, object_payload: dict) -> str:
        top_label = str(object_payload.get("top_label", "")).strip().lower()
        if top_label:
            return top_label
        raw_votes = object_payload.get("label_votes", {})
        if isinstance(raw_votes, dict) and raw_votes:
            try:
                return str(max(raw_votes.items(), key=lambda item: float(item[1]))[0]).strip().lower()
            except Exception:
                return ""
        return ""

    def _recent_target_track_alive(self, image_stamp: float) -> bool:
        if not self.dynamic_semantic_target_labels or self.semantic_object_memory_path is None:
            return False
        target_labels = set(self.dynamic_semantic_target_labels)
        object_memory_payload = self._load_optional_json(self.semantic_object_memory_path)
        for obj_payload in self._iter_semantic_objects(object_memory_payload):
            last_seen = float(obj_payload.get("last_seen_stamp", 0.0))
            if (image_stamp - last_seen) > self.dynamic_semantic_keepalive_sec:
                continue
            state = str(obj_payload.get("state", "active")).strip().lower()
            if state == "dead":
                continue
            if self._semantic_object_top_label(obj_payload) in target_labels:
                return True
        return False

    def _dynamic_semantic_persist_gate(self, image_rgb: np.ndarray, image_stamp: float) -> tuple[bool, list[str], int]:
        if not self.dynamic_semantic_persistence_enabled:
            return True, ["disabled"], 0
        if not self.dynamic_semantic_target_labels:
            return True, ["no_target_labels_configured"], 0

        reasons: list[str] = []
        detection_hits = 0
        if self._dynamic_semantic_client is not None:
            try:
                records, _ = self._dynamic_semantic_client.infer(
                    image_rgb,
                    self.dynamic_semantic_target_labels,
                    include_masks=False,
                )
                for record in records:
                    labels = [str(v).strip().lower() for v in record.get("semantic_label_candidates", []) if str(v).strip()]
                    scores = [float(v) for v in record.get("semantic_scores", [])]
                    top_label = labels[0] if labels else ""
                    top_score = scores[0] if scores else 0.0
                    if top_label in self.dynamic_semantic_target_labels and top_score >= self.dynamic_semantic_min_score:
                        detection_hits += 1
                if detection_hits >= self.dynamic_semantic_min_detections:
                    self._last_dynamic_target_seen_stamp = image_stamp
                    reasons.append("target_detected")
            except Exception as exc:
                self.get_logger().warning(f"Dynamic semantic gate YOLO precheck failed: {exc}")

        if self._recent_target_track_alive(image_stamp):
            reasons.append("keepalive_track")

        if (
            self._last_dynamic_target_seen_stamp is not None
            and (image_stamp - self._last_dynamic_target_seen_stamp) <= self.dynamic_semantic_keepalive_sec
        ):
            reasons.append("recent_target_detection")

        if (
            not reasons
            and self.dynamic_semantic_fallback_sec > 0.0
            and (
                self._last_dynamic_persist_stamp is None
                or (image_stamp - self._last_dynamic_persist_stamp) >= self.dynamic_semantic_fallback_sec
            )
        ):
            reasons.append("fallback")

        should_persist = bool(reasons)
        if should_persist:
            self._last_dynamic_persist_stamp = image_stamp
        return should_persist, sorted(set(reasons)), int(detection_hits)

    def _semantic_trigger_state(self, image_stamp: float) -> tuple[bool, float, list[str]]:
        if not self.semantic_trigger_enabled:
            return False, 0.0, []
        if (
            self._last_semantic_trigger_stamp is not None
            and (image_stamp - self._last_semantic_trigger_stamp) < self.semantic_trigger_cooldown_sec
        ):
            return False, 0.0, []

        under_supported_count = 0
        reasons: list[str] = []
        status_payload = self._load_optional_json(self.semantic_status_path)
        if status_payload is not None:
            pending_count = int(status_payload.get("num_pending_objects", 0))
            if pending_count >= self.semantic_pending_object_threshold:
                under_supported_count = max(under_supported_count, pending_count)
                reasons.append("pending_objects")

        object_memory_payload = self._load_optional_json(self.semantic_object_memory_path)
        if object_memory_payload is not None:
            for obj_payload in self._iter_semantic_objects(object_memory_payload):
                last_seen = float(obj_payload.get("last_seen_stamp", 0.0))
                if (image_stamp - last_seen) > self.semantic_recent_object_window_sec:
                    continue
                support_count = self._count_support_points(obj_payload)
                observation_count = int(obj_payload.get("observation_count", 0))
                completeness_score = float(obj_payload.get("completeness_score", 0.0))
                state = str(obj_payload.get("state", "active"))
                if state == "pending":
                    under_supported_count += 1
                    reasons.append("pending_object_memory")
                    continue
                if (
                    support_count <= self.semantic_under_supported_max_support_points
                    and (completeness_score < 0.35 or observation_count <= self.semantic_under_supported_max_observations)
                ):
                    under_supported_count += 1
                    reasons.append("weak_active_objects")

        if under_supported_count <= 0:
            return False, 0.0, []
        semantic_novelty = min(1.0, float(under_supported_count) / 3.0)
        return True, semantic_novelty, sorted(set(reasons))

    def _ensure_frame_index_header(self) -> None:
        if self.frame_index_path.exists() and self.frame_index_path.stat().st_size > 0:
            return
        self._frame_index_writer.writerow(
            [
                "saved_frame_index",
                "image_stamp_sec",
                "depth_stamp_sec",
                "pose_stamp_sec",
                "sync_match_type",
                "translation_from_prev_keyframe_m",
                "rotation_from_prev_keyframe_deg",
                "reasons",
                "image_name",
                "depth_name",
            ]
        )
        self._frame_index_file.flush()

    def _write_metadata(self) -> None:
        metadata = {
            "scene_name": self.scene_name,
            "pose_source": "direct_camera_pose" if self.require_direct_camera_pose else "fast_livo2_or_direct_camera_pose",
            "pose_topic": self.odom_topic,
            "camera_pose_topic": self.camera_pose_topic,
            "require_direct_camera_pose": self.require_direct_camera_pose,
            "require_exact_direct_camera_pose": self.require_exact_direct_camera_pose,
            "direct_camera_pose_exact_tolerance_sec": self.direct_camera_pose_exact_tolerance_sec,
            "direct_camera_pose_apply_isaac_optical_fix": self.direct_camera_pose_apply_isaac_optical_fix,
            "align_direct_camera_pose_to_odom_world": self.align_direct_camera_pose_to_odom_world,
            "align_direct_camera_pose_to_odom_max_dt_sec": self.align_direct_camera_pose_to_odom_max_dt_sec,
            "require_exact_rgbd_sync": self.require_exact_rgbd_sync,
            "rgbd_exact_tolerance_sec": self.rgbd_exact_tolerance_sec,
            "local_cloud_topic": self.local_cloud_topic,
            "local_cloud_mode": "rgbd_unprojection" if self.use_rgbd_local_cloud else "pointcloud_topic",
            "rgbd_stride": self.rgbd_stride,
            "rgbd_min_depth_m": self.rgbd_min_depth_m,
            "local_cloud_window_sec": self.local_cloud_window_sec,
            "local_cloud_min_scans": self.local_cloud_min_scans,
            "local_cloud_max_scans": self.local_cloud_max_scans,
            "local_cloud_voxel_size_m": self.local_cloud_voxel_size_m,
            "semantic_local_cloud_enabled": self.semantic_local_cloud_enabled,
            "semantic_trusted_cloud_enabled": self.semantic_trusted_cloud_enabled,
            "semantic_trusted_cloud_topic": self.semantic_trusted_cloud_topic,
            "semantic_trusted_cloud_sync_slop_sec": self.semantic_trusted_cloud_sync_slop_sec,
            "semantic_trusted_point_radius_m": self.semantic_trusted_point_radius_m,
            "semantic_trusted_min_support_ratio": self.semantic_trusted_min_support_ratio,
            "semantic_local_cloud_depth_tolerance_m": self.semantic_local_cloud_depth_tolerance_m,
            "semantic_local_cloud_min_points": self.semantic_local_cloud_min_points,
            "semantic_local_cloud_min_edge_distance_px": self.semantic_local_cloud_min_edge_distance_px,
            "semantic_local_cloud_min_quality_score": self.semantic_local_cloud_min_quality_score,
            "image_topic": self.image_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "intrinsics_source": "camera_info" if self.override_intrinsics_from_camera_info else "fast_livo_config",
            "sensor_config_id": self.sensor_config_id,
            "calib_version": self.calib_version,
            "dynamic_semantic_persistence": {
                "enabled": self.dynamic_semantic_persistence_enabled,
                "target_labels": list(self.dynamic_semantic_target_labels),
                "min_detections": self.dynamic_semantic_min_detections,
                "min_score": self.dynamic_semantic_min_score,
                "keepalive_sec": self.dynamic_semantic_keepalive_sec,
                "fallback_sec": self.dynamic_semantic_fallback_sec,
            },
            "selector": {
                "min_time_gap_sec": self.min_time_gap_sec,
                "max_time_gap_sec": self.max_time_gap_sec,
                "translation_thresh_m": self.translation_thresh_m,
                "rotation_thresh_deg": self.rotation_thresh_deg,
                "depth_valid_ratio_thresh": self.depth_valid_ratio_thresh,
                "blur_score_min": self.blur_score_min,
                "coverage_novelty_thresh": self.coverage_novelty_thresh,
                "coverage_grid_shape": list(self.coverage_grid_shape),
                "force_every_synced_frame": self.force_every_synced_frame,
                "min_translation_for_rotation_trigger_m": self.min_translation_for_rotation_trigger_m,
                "min_translation_for_time_trigger_m": self.min_translation_for_time_trigger_m,
                "semantic_trigger_enabled": self.semantic_trigger_enabled,
                "semantic_status_path": None if self.semantic_status_path is None else str(self.semantic_status_path),
                "semantic_object_memory_path": None
                if self.semantic_object_memory_path is None
                else str(self.semantic_object_memory_path),
                "semantic_trigger_min_translation_m": self.semantic_trigger_min_translation_m,
                "semantic_trigger_cooldown_sec": self.semantic_trigger_cooldown_sec,
            },
            "direct_pose_world_alignment": {
                "status": "ready" if self._direct_pose_to_odom_world is not None else "waiting"
                if self.align_direct_camera_pose_to_odom_world
                else "disabled",
                "stamp_sec": self._direct_pose_to_odom_world_stamp,
                "odom_pose_dt_sec": self._direct_pose_to_odom_world_pose_dt,
                "match_type": self._direct_pose_to_odom_world_match_type,
                "matrix": None
                if self._direct_pose_to_odom_world is None
                else self._direct_pose_to_odom_world.astype(np.float32).tolist(),
            },
            "extrinsics": {
                "Rcl": self.r_cl.tolist(),
                "Pcl": self.p_cl.tolist(),
                "Rli": self.r_li.tolist(),
                "Pli": self.p_li.tolist(),
            },
        }
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False)
        self._write_sensor_config()

    def _write_json(self, path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _write_jsonl(self, handle, payload: dict) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    def _write_manifest(self) -> None:
        pose_authority = self.camera_pose_topic if self.require_direct_camera_pose else f"{self.camera_pose_topic} or {self.odom_topic}"
        local_cloud_description = (
            f"each `local_cloud_packet` is built by unprojecting RGB-D using `{self.camera_pose_topic}` "
            f"(stride={self.rgbd_stride}, min_depth_m={self.rgbd_min_depth_m}, voxel_size_m={self.local_cloud_voxel_size_m})"
            if self.use_rgbd_local_cloud
            else "each `local_cloud_packet` is a short, past-only local submap fused from recent `/cloud_registered` scans"
        )
        manifest = f"""# FAST-LIVO2 to OVO Keyframe Export

- run_id: `{self.run_id}`
- scene_name: `{self.scene_name}`
- pose authority: `{pose_authority}`
- image topic: `{self.image_topic}`
- depth topic: `{self.depth_topic}`
- intrinsics source: `{"camera_info" if self.override_intrinsics_from_camera_info else "fast_livo_config"}`
- dynamic semantic persistence: `{"enabled" if self.dynamic_semantic_persistence_enabled else "disabled"}`
- dynamic semantic target labels: `{",".join(self.dynamic_semantic_target_labels) if self.dynamic_semantic_target_labels else "none"}`
- keyframe policy: ORB-like but independent of ORB-SLAM3

This run exports Replica-style OVO inputs under:
- dataset root: `{self.scene_dir}`
- run root: `{self.run_dir}`

Observer-style additions:
- `sensor_config.yaml`
- `local_clouds/local_cloud_*.npz`
- `export/results/frame*.jpg` and `export/results/depth*.png` (stable run-local copies)
- `export/local_clouds/local_cloud_*.npz` (stable run-local copies)
- `export/keyframe_packets.jsonl`
- `export/local_cloud_packets.jsonl`

Selection logic:
- quality gates: pose present, monotonic timestamp, depth valid ratio, blur score
- triggers: translation, rotation, coverage novelty, max time gap

Downstream OVO immediate contract:
- `results/frame*.jpg`
- `results/depth*.png`
- `traj.txt`
- scene config yaml with intrinsics and `depth_scale`

Local cloud export:
- {local_cloud_description}
- submap settings: `window_sec={self.local_cloud_window_sec}`, `min_scans={self.local_cloud_min_scans}`, `max_scans={self.local_cloud_max_scans}`, `voxel_size_m={self.local_cloud_voxel_size_m}`
"""
        (self.run_dir / "manifest.md").write_text(manifest, encoding="utf-8")
        (self.export_dir / "manifest.md").write_text(manifest, encoding="utf-8")

    def _write_contract_check(self) -> None:
        contract = {
            "downstream_dataset": "OVO Replica",
            "required_fields": [
                "results/frame*.jpg",
                "results/depth*.png",
                "traj.txt",
                "scene_config(cam.H,W,fx,fy,cx,cy,depth_scale)",
                "sensor_config.yaml",
                "export/results/frame*.jpg",
                "export/results/depth*.png",
                "export/local_clouds/local_cloud_*.npz",
                "export/keyframe_packets.jsonl",
                "export/local_cloud_packets.jsonl",
            ],
            "present_now": {
                "results/frame*.jpg": True,
                "results/depth*.png": True,
                "traj.txt": True,
                "scene_config(cam.H,W,fx,fy,cx,cy,depth_scale)": True,
                "sensor_config.yaml": True,
                "export/results/frame*.jpg": True,
                "export/results/depth*.png": True,
                "export/local_clouds/local_cloud_*.npz": True,
                "export/keyframe_packets.jsonl": True,
                "export/local_cloud_packets.jsonl": True,
            },
            "pose_format": "T_world_cam as c2w rows in traj.txt",
            "status": "usable",
            "notes": [
                "OVO local Replica loaders read image/depth pairs from results/ and c2w poses from traj.txt.",
                "Additional run metadata is stored under runs/ovo_pose_keyframes and does not block OVO.",
            ],
            "sources": [
                "/home/peng/isacc_slam/reference/OVO/ovo/entities/datasets.py",
                "/home/peng/isacc_slam/reference/OVO/ovo/submodules/gaussian_slam/entities/datasets.py",
            ],
        }
        self._write_json(self.run_dir / "ovo_contract_check.json", contract)

    def _write_export_session(self) -> None:
        payload = {
            "run_id": self.run_id,
            "scene_name": self.scene_name,
            "run_dir": str(self.run_dir),
            "export_dir": str(self.export_dir),
            "created_wall_time_ns": int(datetime.now().timestamp() * 1e9),
        }
        self._write_json(self.export_session_path, payload)

    def _write_scene_config(self) -> None:
        scene_config = {
            "cam": {
                "H": int(self.cam_height),
                "W": int(self.cam_width),
                "fx": float(self.fx),
                "fy": float(self.fy),
                "cx": float(self.cx),
                "cy": float(self.cy),
                "depth_scale": float(self.output_depth_scale),
                "depth_min_m": 0.0,
                "depth_max_m": float(self.max_depth_m),
            }
        }
        with self.scene_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(scene_config, handle, sort_keys=False)
        self._config_written = True
        self._write_sensor_config()

    def _write_sensor_config(self) -> None:
        sensor_config = SensorConfig(
            sensor_config_id=self.sensor_config_id,
            calib_version=self.calib_version,
            camera_model="pinhole",
            intrinsics={
                "fx": float(self.fx),
                "fy": float(self.fy),
                "cx": float(self.cx),
                "cy": float(self.cy),
                "depth_scale": float(self.output_depth_scale),
                "depth_min_m": 0.0,
                "depth_max_m": float(self.max_depth_m),
            },
            t_lidar_cam=[float(v) for v in (-(self.r_cl.T @ self.p_cl.reshape(3, 1))).reshape(3).tolist()],
            image_size=(int(self.cam_width), int(self.cam_height)),
            distortion_model="none",
            r_lidar_cam=[float(v) for v in self.r_cl.reshape(-1).tolist()],
            t_imu_lidar=[float(v) for v in self.p_li.tolist()],
            r_imu_lidar=[float(v) for v in self.r_li.reshape(-1).tolist()],
        )
        write_sensor_config(self.sensor_config_path, sensor_config)

    def _update_trajectory_summary(self) -> None:
        mean_dt = float(np.mean(self._trajectory_dt_samples)) if self._trajectory_dt_samples else 0.0
        max_dt = float(np.max(self._trajectory_dt_samples)) if self._trajectory_dt_samples else 0.0
        payload = {
            "pose_topic": self.odom_topic,
            "camera_pose_topic": self.camera_pose_topic,
            "require_direct_camera_pose": self.require_direct_camera_pose,
            "require_exact_direct_camera_pose": self.require_exact_direct_camera_pose,
            "direct_camera_pose_apply_isaac_optical_fix": self.direct_camera_pose_apply_isaac_optical_fix,
            "align_direct_camera_pose_to_odom_world": self.align_direct_camera_pose_to_odom_world,
            "align_direct_camera_pose_to_odom_max_dt_sec": self.align_direct_camera_pose_to_odom_max_dt_sec,
            "require_exact_rgbd_sync": self.require_exact_rgbd_sync,
            "rgbd_exact_tolerance_sec": self.rgbd_exact_tolerance_sec,
            "message_type": "nav_msgs/msg/Odometry",
            "num_poses": self._num_poses_received,
            "first_timestamp": self._odom_buffer[0].stamp if self._odom_buffer else None,
            "last_timestamp": self._odom_buffer[-1].stamp if self._odom_buffer else None,
            "camera_pose_first_timestamp": self._camera_pose_buffer[0].stamp if self._camera_pose_buffer else None,
            "camera_pose_last_timestamp": self._camera_pose_buffer[-1].stamp if self._camera_pose_buffer else None,
            "mean_dt": mean_dt,
            "max_dt": max_dt,
            "num_non_monotonic_poses_dropped": self._num_pose_non_monotonic,
            "num_pre_axis_poses_dropped": self._num_pose_axis_wait_drops,
            "frame_convention": "direct camera pose is interpreted as T_world_cam; odom fallback is converted through camera extrinsics",
            "direct_camera_pose_frame_note": (
                "when direct_camera_pose_apply_isaac_optical_fix=true, direct camera pose rotation is converted "
                "from Isaac/USD camera axes (+X right,+Y up,-Z forward) to exporter optical axes "
                "(+X right,+Y down,+Z forward)"
            ),
            "camera_extrinsic_available": True,
            "preferred_pose_source": self.camera_pose_topic if self._camera_pose_buffer else self.odom_topic,
            "status": "ok" if (self._camera_pose_buffer or self._odom_buffer) else "waiting_for_pose",
            "direct_pose_world_alignment_status": "ready"
            if self._direct_pose_to_odom_world is not None
            else "waiting"
            if self.align_direct_camera_pose_to_odom_world
            else "disabled",
            "direct_pose_world_alignment_stamp_sec": self._direct_pose_to_odom_world_stamp,
            "direct_pose_world_alignment_odom_pose_dt_sec": self._direct_pose_to_odom_world_pose_dt,
            "direct_pose_world_alignment_match_type": self._direct_pose_to_odom_world_match_type,
        }
        self._write_json(self.run_dir / "trajectory_summary.json", payload)

    def _update_alignment_summary(self) -> None:
        payload = {
            "num_rgb_frames_seen": self._num_rgb_frames_seen,
            "num_depth_frames_seen": self._num_depth_frames_seen,
            "num_frames_aligned": self._synced_frame_count,
            "num_frames_pose_missing": self._num_pose_missing,
            "num_frames_interpolated": self._num_frames_interpolated,
            "num_frames_exact_match": self._num_frames_exact_match,
            "rgb_dt_mean": float(np.mean(self._rgb_dt_samples)) if self._rgb_dt_samples else 0.0,
            "depth_dt_mean": float(np.mean(self._depth_dt_samples)) if self._depth_dt_samples else 0.0,
            "aligned_pose_dt_mean": float(np.mean(self._aligned_pose_dt_samples)) if self._aligned_pose_dt_samples else 0.0,
            "pose_alignment_status": "ok" if self._synced_frame_count > 0 else "waiting_for_alignment",
            "align_direct_camera_pose_to_odom_world": self.align_direct_camera_pose_to_odom_world,
            "align_direct_camera_pose_to_odom_max_dt_sec": self.align_direct_camera_pose_to_odom_max_dt_sec,
            "direct_pose_world_alignment_status": "ready"
            if self._direct_pose_to_odom_world is not None
            else "waiting"
            if self.align_direct_camera_pose_to_odom_world
            else "disabled",
            "direct_pose_world_alignment_stamp_sec": self._direct_pose_to_odom_world_stamp,
            "direct_pose_world_alignment_odom_pose_dt_sec": self._direct_pose_to_odom_world_pose_dt,
            "direct_pose_world_alignment_match_type": self._direct_pose_to_odom_world_match_type,
            "direct_pose_world_alignment_matrix": None
            if self._direct_pose_to_odom_world is None
            else self._direct_pose_to_odom_world.astype(np.float32).tolist(),
        }
        self._write_json(self.run_dir / "alignment_summary.json", payload)

    def _update_keyframe_summary(self) -> None:
        mean_translation = 0.0
        mean_rotation = 0.0
        accepted_depth_ratios = []
        accepted_csv = self.run_dir / "accepted_frames.jsonl"
        if accepted_csv.exists() and accepted_csv.stat().st_size > 0:
            translations = []
            rotations = []
            with accepted_csv.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("is_keyframe"):
                        translations.append(float(row.get("translation_from_prev_keyframe_m", 0.0)))
                        rotations.append(float(row.get("rotation_from_prev_keyframe_deg", 0.0)))
                    accepted_depth_ratios.append(float(row.get("depth_valid_ratio", 0.0)))
            if translations:
                mean_translation = float(np.mean(translations))
            if rotations:
                mean_rotation = float(np.mean(rotations))
        payload = {
            "num_candidate_frames": self._synced_frame_count,
            "num_keyframes": self._saved_frame_count,
            "selection_ratio": float(self._saved_frame_count / max(self._synced_frame_count, 1)),
            "mean_translation_between_keyframes": mean_translation,
            "mean_rotation_between_keyframes": mean_rotation,
            "mean_depth_valid_ratio": float(np.mean(accepted_depth_ratios)) if accepted_depth_ratios else 0.0,
            "coverage_trigger_count": self._coverage_trigger_count,
            "translation_trigger_count": self._translation_trigger_count,
            "rotation_trigger_count": self._rotation_trigger_count,
            "time_trigger_count": self._time_trigger_count,
            "num_pose_missing": self._num_pose_missing,
            "num_quality_rejected": self._num_quality_rejected,
            "selection_reasons": dict(self._selection_reasons),
        }
        self._write_json(self.export_dir / "keyframes_summary.json", payload)
        self._write_json(self.run_dir / "keyframes_summary.json", payload)

    def _record_rgb_dt(self, stamp: float) -> None:
        if self._image_buffer:
            self._rgb_dt_samples.append(stamp - self._image_buffer[-1].stamp)

    def _record_depth_dt(self, stamp: float) -> None:
        if self._depth_buffer:
            self._depth_dt_samples.append(stamp - self._depth_buffer[-1].stamp)

    def _image_callback(self, msg: Image) -> None:
        if self._stopped_for_max_frames:
            return
        stamp = stamp_to_seconds(msg.header.stamp)
        self._num_rgb_frames_seen += 1
        self._record_rgb_dt(stamp)
        self._latest_sensor_stamp = stamp
        image_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self._image_buffer.append(BufferedImage(stamp=stamp, image_rgb=image_rgb))
        self._drain_pending_frames()

    def _depth_callback(self, msg: Image) -> None:
        stamp = stamp_to_seconds(msg.header.stamp)
        self._num_depth_frames_seen += 1
        self._record_depth_dt(stamp)
        self._latest_sensor_stamp = stamp
        depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth_raw)
        if np.issubdtype(depth.dtype, np.integer):
            depth_m = depth.astype(np.float32) / max(self.depth_input_scale, 1e-6)
        else:
            depth_m = depth.astype(np.float32, copy=False)
        if self.max_depth_m > 0.0:
            depth_m = depth_m.copy()
            depth_m[(~np.isfinite(depth_m)) | (depth_m > self.max_depth_m)] = 0.0
        self._depth_buffer.append(BufferedDepth(stamp=stamp, depth_m=depth_m))
        self._drain_pending_frames()

    def _cloud_callback(self, msg: PointCloud2) -> None:
        stamp = stamp_to_seconds(msg.header.stamp)
        points = pointcloud_xyz_array(msg)
        if points.size == 0:
            return
        scan_id = f"scan_{int(round(stamp * 1.0e9)):019d}"
        self._cloud_buffer.append(
            BufferedCloud(
                stamp=stamp,
                scan_id=scan_id,
                frame_id=msg.header.frame_id,
                xyz=points,
            )
        )
        self._drain_pending_frames()

    def _semantic_cloud_callback(self, msg: PointCloud2) -> None:
        stamp = stamp_to_seconds(msg.header.stamp)
        points = pointcloud_xyz_array(msg)
        if points.size == 0:
            return
        scan_id = f"semantic_scan_{int(round(stamp * 1.0e9)):019d}"
        self._semantic_cloud_buffer.append(
            BufferedCloud(
                stamp=stamp,
                scan_id=scan_id,
                frame_id=msg.header.frame_id,
                xyz=points,
            )
        )
        self._drain_pending_frames()

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        k = list(msg.k)
        if len(k) != 9 or k[0] <= 0.0 or k[4] <= 0.0:
            return

        msg_fx = float(k[0])
        msg_fy = float(k[4])
        msg_cx = float(k[2])
        msg_cy = float(k[5])
        msg_width = int(msg.width)
        msg_height = int(msg.height)

        if self.override_intrinsics_from_camera_info:
            self.fx = msg_fx
            self.fy = msg_fy
            self.cx = msg_cx
            self.cy = msg_cy
            self.cam_width = msg_width
            self.cam_height = msg_height
        elif not self._camera_info_warning_emitted:
            mismatch = (
                abs(self.fx - msg_fx) > 1e-3
                or abs(self.fy - msg_fy) > 1e-3
                or abs(self.cx - msg_cx) > 1e-3
                or abs(self.cy - msg_cy) > 1e-3
                or self.cam_width != msg_width
                or self.cam_height != msg_height
            )
            if mismatch:
                self.get_logger().warn(
                    "CameraInfo intrinsics differ from FAST-LIVO2 camera model. "
                    f"Keeping configured intrinsics fx={self.fx:.3f}, fy={self.fy:.3f}, "
                    f"cx={self.cx:.3f}, cy={self.cy:.3f}, size={self.cam_width}x{self.cam_height}; "
                    f"camera_info reported fx={msg_fx:.3f}, fy={msg_fy:.3f}, "
                    f"cx={msg_cx:.3f}, cy={msg_cy:.3f}, size={msg_width}x{msg_height}."
                )
            self._camera_info_warning_emitted = True

        self._camera_info_received = True
        self._write_scene_config()

    def _odom_callback(self, msg: Odometry) -> None:
        raw_stamp = stamp_to_seconds(msg.header.stamp)
        odom_stamp = self._normalize_odom_stamp(raw_stamp)
        if odom_stamp is None:
            self._update_trajectory_summary()
            return
        self._num_poses_received += 1
        if self._odom_buffer:
            dt = odom_stamp - self._odom_buffer[-1].stamp
            if dt <= 0.0:
                self._num_pose_non_monotonic += 1
                return
            self._trajectory_dt_samples.append(dt)

        quat_msg = msg.pose.pose.orientation
        quat = np.array([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w], dtype=np.float32)
        rotation = quaternion_to_rotation_matrix(*quat)
        translation = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=np.float32,
        )
        entry = BufferedOdom(
            stamp=odom_stamp,
            rotation=rotation,
            translation=translation,
            quaternion_xyzw=quat,
        )
        self._odom_buffer.append(entry)
        self._latest_odom_stamp = odom_stamp
        self._write_jsonl(
            self._trajectory_file,
            {
                "timestamp_sec": odom_stamp,
                "translation_xyz_m": translation.tolist(),
                "quaternion_xyzw": quat.tolist(),
                "frame_id": msg.header.frame_id,
                "child_frame_id": msg.child_frame_id,
            },
        )
        self._update_trajectory_summary()
        self._drain_pending_frames()

    def _camera_pose_callback(self, msg: Odometry) -> None:
        pose_stamp = stamp_to_seconds(msg.header.stamp)
        if self._camera_pose_buffer:
            dt = pose_stamp - self._camera_pose_buffer[-1].stamp
            if dt <= 0.0:
                return
        quat_msg = msg.pose.pose.orientation
        quat = np.array([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w], dtype=np.float32)
        rotation = quaternion_to_rotation_matrix(*quat)
        translation = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=np.float32,
        )
        self._camera_pose_buffer.append(
            BufferedOdom(
                stamp=pose_stamp,
                rotation=rotation,
                translation=translation,
                quaternion_xyzw=quat,
            )
        )
        self._latest_camera_pose_stamp = pose_stamp
        self._update_trajectory_summary()
        self._drain_pending_frames()

    def _normalize_odom_stamp(self, raw_stamp: float) -> Optional[float]:
        if self._latest_sensor_stamp is None:
            if raw_stamp > 1.0e6:
                self._num_pose_axis_wait_drops += 1
                if self._num_pose_axis_wait_drops == 1:
                    self.get_logger().warn(
                        "LIVO2 exporter saw wall-time odom before sensor sim-time was established. "
                        "Dropping early odom until camera/depth timestamps are available."
                    )
                return None
            return raw_stamp
        if abs(raw_stamp - self._latest_sensor_stamp) <= self.sync_slop_sec:
            return raw_stamp
        if self._odom_time_offset is None:
            self._odom_time_offset = raw_stamp - self._latest_sensor_stamp
            self.get_logger().warn(
                "LIVO2 exporter detected odom on a different time axis. "
                f"Applying offset {self._odom_time_offset:.6f}s."
            )
        return raw_stamp - self._odom_time_offset

    def _drain_pending_frames(self) -> None:
        for item in list(self._image_buffer):
            if self._last_processed_candidate_stamp is not None and item.stamp <= self._last_processed_candidate_stamp + 1e-6:
                continue
            processed = self._try_export_frame(item.stamp)
            if not processed:
                continue

    def _find_closest(self, buffer: Deque, stamp: float):
        if not buffer:
            return None
        best = min(buffer, key=lambda item: abs(item.stamp - stamp))
        if abs(best.stamp - stamp) > self.sync_slop_sec:
            return None
        return best

    def _find_closest_cloud(self, stamp: float) -> Optional[BufferedCloud]:
        if not self._cloud_buffer:
            return None
        best = min(self._cloud_buffer, key=lambda item: abs(item.stamp - stamp))
        if abs(best.stamp - stamp) > self.local_cloud_sync_slop_sec:
            return None
        return best

    def _find_closest_semantic_cloud(self, stamp: float) -> Optional[BufferedCloud]:
        if not self.semantic_trusted_cloud_enabled or not self._semantic_cloud_buffer:
            return None
        best = min(self._semantic_cloud_buffer, key=lambda item: abs(item.stamp - stamp))
        if abs(best.stamp - stamp) > self.semantic_trusted_cloud_sync_slop_sec:
            return None
        return best

    def _build_local_submap(self, stamp: float) -> Optional[BufferedCloudSubmap]:
        anchor = self._find_closest_cloud(stamp)
        if anchor is None:
            return None

        if self.local_cloud_window_sec <= 1e-6 or self.local_cloud_max_scans <= 1:
            xyz = self._voxel_downsample_xyz(anchor.xyz.astype(np.float32, copy=False), self.local_cloud_voxel_size_m)
            if xyz.size == 0:
                return None
            return BufferedCloudSubmap(
                anchor_stamp=float(anchor.stamp),
                stamp_start=float(anchor.stamp),
                stamp_end=float(anchor.stamp),
                frame_id=str(self.local_cloud_frame or anchor.frame_id),
                xyz=xyz,
                source_scan_ids=[anchor.scan_id],
            )

        window_start = anchor.stamp - max(self.local_cloud_window_sec, 0.0)
        source_clouds = [item for item in self._cloud_buffer if window_start <= item.stamp <= anchor.stamp + 1e-6]
        if self.local_cloud_max_scans > 0 and len(source_clouds) > self.local_cloud_max_scans:
            source_clouds = source_clouds[-self.local_cloud_max_scans :]
        if len(source_clouds) < max(self.local_cloud_min_scans, 1):
            return None

        xyz_parts = [item.xyz for item in source_clouds if item.xyz.size > 0]
        if not xyz_parts:
            return None
        xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32, copy=False)
        xyz = self._voxel_downsample_xyz(xyz, self.local_cloud_voxel_size_m)
        if xyz.size == 0:
            return None

        return BufferedCloudSubmap(
            anchor_stamp=float(anchor.stamp),
            stamp_start=float(source_clouds[0].stamp),
            stamp_end=float(source_clouds[-1].stamp),
            frame_id=str(self.local_cloud_frame or anchor.frame_id),
            xyz=xyz,
            source_scan_ids=[item.scan_id for item in source_clouds],
        )

    def _build_rgbd_local_submap(self, depth_m: np.ndarray, c2w: np.ndarray, stamp: float) -> Optional[BufferedCloudSubmap]:
        if depth_m.size == 0:
            return None

        stride = max(self.rgbd_stride, 1)
        sampled_depth = depth_m[::stride, ::stride]
        valid = np.isfinite(sampled_depth) & (sampled_depth > max(self.rgbd_min_depth_m, 1e-4))
        if self.max_depth_m > 0.0:
            valid &= sampled_depth <= self.max_depth_m
        if not np.any(valid):
            return None

        sample_v, sample_u = np.nonzero(valid)
        u = (sample_u * stride).astype(np.float32)
        v = (sample_v * stride).astype(np.float32)
        z = sampled_depth[valid].astype(np.float32, copy=False)
        x = ((u - float(self.cx)) * z) / max(float(self.fx), 1e-6)
        y = ((v - float(self.cy)) * z) / max(float(self.fy), 1e-6)
        xyz_cam = np.column_stack((x, y, z)).astype(np.float32, copy=False)
        xyz_world = (xyz_cam @ c2w[:3, :3].T.astype(np.float32, copy=False)) + c2w[:3, 3].astype(np.float32, copy=False)
        xyz_world = self._voxel_downsample_xyz(xyz_world, self.local_cloud_voxel_size_m)
        if xyz_world.size == 0:
            return None

        frame_id = str(self.local_cloud_frame or "camera_init")
        scan_id = f"rgbd_{int(round(stamp * 1.0e9)):019d}"
        return BufferedCloudSubmap(
            anchor_stamp=float(stamp),
            stamp_start=float(stamp),
            stamp_end=float(stamp),
            frame_id=frame_id,
            xyz=xyz_world.astype(np.float32, copy=False),
            source_scan_ids=[scan_id],
            cloud_kind="rgbd_local_cloud",
        )

    def _voxel_downsample_xyz(self, xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
        if xyz.size == 0:
            return xyz.reshape(-1, 3).astype(np.float32, copy=False)
        if voxel_size_m <= 1e-6:
            return xyz.astype(np.float32, copy=False)
        voxel_coords = np.floor(xyz / voxel_size_m).astype(np.int64, copy=False)
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        unique_indices.sort()
        return xyz[np.asarray(unique_indices, dtype=np.int64)].astype(np.float32, copy=False)

    def _depth_edge_distance(self, depth_m: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth_m) & (depth_m > 0.05)
        if not np.any(valid):
            return np.zeros_like(depth_m, dtype=np.float32)
        depth_filled = depth_m.astype(np.float32, copy=True)
        depth_filled[~valid] = 0.0
        grad_x = cv2.Sobel(depth_filled, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_filled, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
        edge_mask = (~valid) | (grad_mag > max(self.semantic_local_cloud_depth_tolerance_m * 1.5, 0.03))
        safe_region = (~edge_mask).astype(np.uint8)
        return cv2.distanceTransform(safe_region, cv2.DIST_L2, 3).astype(np.float32)

    def _trusted_support_mask(self, xyz_world: np.ndarray, trusted_xyz_world: np.ndarray) -> np.ndarray:
        if xyz_world.size == 0 or trusted_xyz_world.size == 0:
            return np.zeros((xyz_world.shape[0],), dtype=bool)
        radius = max(self.semantic_trusted_point_radius_m, 1e-3)
        voxel_size = max(radius, 0.05)
        radius_sq = radius * radius
        trusted_voxels: dict[tuple[int, int, int], list[np.ndarray]] = {}
        trusted_coords = np.floor(trusted_xyz_world / voxel_size).astype(np.int32)
        for coord, point in zip(trusted_coords, trusted_xyz_world):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            trusted_voxels.setdefault(key, []).append(point.astype(np.float32, copy=False))
        support = np.zeros((xyz_world.shape[0],), dtype=bool)
        for idx, point in enumerate(xyz_world):
            coord = np.floor(point / voxel_size).astype(np.int32)
            found = False
            for dx in (-1, 0, 1):
                if found:
                    break
                for dy in (-1, 0, 1):
                    if found:
                        break
                    for dz in (-1, 0, 1):
                        key = (int(coord[0] + dx), int(coord[1] + dy), int(coord[2] + dz))
                        candidates = trusted_voxels.get(key)
                        if not candidates:
                            continue
                        candidate_xyz = np.asarray(candidates, dtype=np.float32)
                        diff = candidate_xyz - point.astype(np.float32, copy=False)
                        dist_sq = np.einsum("ij,ij->i", diff, diff)
                        if np.any(dist_sq <= radius_sq):
                            support[idx] = True
                            found = True
                            break
        return support

    def _build_semantic_local_cloud(
        self,
        xyz_world: np.ndarray,
        c2w: np.ndarray,
        depth_m: np.ndarray,
        trusted_xyz_world: np.ndarray | None = None,
    ) -> BufferedCloudSubmap | None:
        if xyz_world.size == 0 or depth_m.size == 0:
            return None
        source_point_indices = np.arange(xyz_world.shape[0], dtype=np.int32)
        w2c = np.linalg.inv(c2w)
        xyz_h = np.hstack((xyz_world.astype(np.float32, copy=False), np.ones((xyz_world.shape[0], 1), dtype=np.float32)))
        xyz_cam = (w2c @ xyz_h.T).T[:, :3]
        z = xyz_cam[:, 2]
        in_front = z > 0.05
        if not np.any(in_front):
            return None

        xyz_world = xyz_world[in_front]
        source_point_indices = source_point_indices[in_front]
        xyz_cam = xyz_cam[in_front]
        z = z[in_front]
        u = np.rint((xyz_cam[:, 0] * self.fx / xyz_cam[:, 2]) + self.cx).astype(np.int32)
        v = np.rint((xyz_cam[:, 1] * self.fy / xyz_cam[:, 2]) + self.cy).astype(np.int32)
        in_image = (u >= 0) & (u < self.cam_width) & (v >= 0) & (v < self.cam_height)
        if not np.any(in_image):
            return None

        xyz_world = xyz_world[in_image]
        source_point_indices = source_point_indices[in_image]
        z = z[in_image]
        u = u[in_image]
        v = v[in_image]

        pixel_id = v.astype(np.int64) * self.cam_width + u.astype(np.int64)
        order = np.lexsort((z, pixel_id))
        pixel_id = pixel_id[order]
        xyz_world = xyz_world[order]
        source_point_indices = source_point_indices[order]
        z = z[order]
        u = u[order]
        v = v[order]
        zbuffer_rank = np.zeros_like(z, dtype=np.int32)
        for idx in range(1, pixel_id.shape[0]):
            if pixel_id[idx] == pixel_id[idx - 1]:
                zbuffer_rank[idx] = zbuffer_rank[idx - 1] + 1
        _, unique_indices = np.unique(pixel_id, return_index=True)
        xyz_world = xyz_world[unique_indices]
        source_point_indices = source_point_indices[unique_indices]
        z = z[unique_indices]
        u = u[unique_indices]
        v = v[unique_indices]
        zbuffer_rank = zbuffer_rank[unique_indices]

        sampled_depth = depth_m[v, u]
        finite = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
        if not np.any(finite):
            return None
        xyz_world = xyz_world[finite]
        source_point_indices = source_point_indices[finite]
        z = z[finite]
        u = u[finite]
        v = v[finite]
        zbuffer_rank = zbuffer_rank[finite]
        sampled_depth = sampled_depth[finite]
        depth_residual = sampled_depth - z
        keep = np.abs(depth_residual) <= self.semantic_local_cloud_depth_tolerance_m
        if not np.any(keep):
            return None

        xyz_world = xyz_world[keep]
        source_point_indices = source_point_indices[keep]
        z = z[keep]
        u = u[keep]
        v = v[keep]
        zbuffer_rank = zbuffer_rank[keep]
        depth_residual = depth_residual[keep]
        sampled_depth = sampled_depth[keep]
        depth_edge = self._depth_edge_distance(depth_m)
        distance_to_depth_edge = depth_edge[v, u]
        visibility_score = np.clip(
            1.0 - (np.abs(depth_residual) / max(self.semantic_local_cloud_depth_tolerance_m, 1e-4)),
            0.0,
            1.0,
        ).astype(np.float32)
        edge_score = np.clip(distance_to_depth_edge / 4.0, 0.0, 1.0)
        rank_score = np.clip(1.0 - (zbuffer_rank.astype(np.float32) * 0.25), 0.0, 1.0)
        quality_score = np.clip((visibility_score * 0.65) + (edge_score * 0.25) + (rank_score * 0.10), 0.0, 1.0)
        trusted_support_mask = np.ones((xyz_world.shape[0],), dtype=bool)
        if trusted_xyz_world is not None and trusted_xyz_world.size > 0:
            trusted_support_mask = self._trusted_support_mask(xyz_world, trusted_xyz_world.astype(np.float32, copy=False))
            trusted_support_ratio = float(np.count_nonzero(trusted_support_mask)) / float(max(xyz_world.shape[0], 1))
            quality_score = np.clip(
                (quality_score * 0.75) + (trusted_support_mask.astype(np.float32) * 0.25),
                0.0,
                1.0,
            )
            if (
                int(np.count_nonzero(trusted_support_mask)) >= max(self.semantic_local_cloud_min_points, 1)
                and trusted_support_ratio >= self.semantic_trusted_min_support_ratio
            ):
                xyz_world = xyz_world[trusted_support_mask]
                source_point_indices = source_point_indices[trusted_support_mask]
                z = z[trusted_support_mask]
                u = u[trusted_support_mask]
                v = v[trusted_support_mask]
                zbuffer_rank = zbuffer_rank[trusted_support_mask]
                depth_residual = depth_residual[trusted_support_mask]
                distance_to_depth_edge = distance_to_depth_edge[trusted_support_mask]
                visibility_score = visibility_score[trusted_support_mask]
                quality_score = quality_score[trusted_support_mask]
        strict_keep = (
            (distance_to_depth_edge >= self.semantic_local_cloud_min_edge_distance_px)
            & (quality_score >= self.semantic_local_cloud_min_quality_score)
        )
        if int(np.count_nonzero(strict_keep)) >= max(self.semantic_local_cloud_min_points, 1):
            xyz_world = xyz_world[strict_keep]
            source_point_indices = source_point_indices[strict_keep]
            z = z[strict_keep]
            u = u[strict_keep]
            v = v[strict_keep]
            zbuffer_rank = zbuffer_rank[strict_keep]
            depth_residual = depth_residual[strict_keep]
            distance_to_depth_edge = distance_to_depth_edge[strict_keep]
            visibility_score = visibility_score[strict_keep]
            quality_score = quality_score[strict_keep]

        if xyz_world.shape[0] < max(self.semantic_local_cloud_min_points, 1):
            return None

        return BufferedCloudSubmap(
            anchor_stamp=0.0,
            stamp_start=0.0,
            stamp_end=0.0,
            frame_id=str(self.local_cloud_frame),
            xyz=xyz_world.astype(np.float32, copy=False),
            source_scan_ids=[],
            source_point_indices=source_point_indices.astype(np.int32, copy=False),
            cloud_kind="semantic_local_cloud",
            quality_fields_present=True,
            uv=np.column_stack((u, v)).astype(np.int32),
            projected_depth=z.astype(np.float32),
            depth_residual=depth_residual.astype(np.float32),
            zbuffer_rank=zbuffer_rank.astype(np.int32),
            distance_to_depth_edge=distance_to_depth_edge.astype(np.float32),
            visibility_score=visibility_score.astype(np.float32),
            quality_score=quality_score.astype(np.float32),
        )

    def _interpolate_pose(self, stamp: float) -> tuple[Optional[np.ndarray], Optional[float], str, Optional[float], str]:
        c2w, pose_dt, match_type, aligned_pose_dt = self._interpolate_pose_buffer(
            self._camera_pose_buffer,
            stamp,
            direct_camera_pose=True,
        )
        if c2w is not None:
            if self.align_direct_camera_pose_to_odom_world:
                c2w = self._align_direct_pose_to_odom_world(stamp, c2w)
                if c2w is None:
                    return None, None, "missing", None, "waiting_for_odom_world_alignment"
                return c2w, pose_dt, match_type, aligned_pose_dt, f"{self.camera_pose_topic}+odom_world_align"
            return c2w, pose_dt, match_type, aligned_pose_dt, self.camera_pose_topic

        if self.require_direct_camera_pose:
            return None, None, "missing", None, "missing"

        c2w, pose_dt, match_type, aligned_pose_dt = self._interpolate_pose_buffer(
            self._odom_buffer,
            stamp,
            direct_camera_pose=False,
        )
        if c2w is not None:
            return c2w, pose_dt, match_type, aligned_pose_dt, self.odom_topic
        return None, None, "missing", None, "missing"

    def _align_direct_pose_to_odom_world(self, stamp: float, direct_c2w: np.ndarray) -> Optional[np.ndarray]:
        if not self.align_direct_camera_pose_to_odom_world:
            return direct_c2w

        if self._direct_pose_to_odom_world is None:
            odom_c2w, odom_pose_dt, odom_match_type, _ = self._interpolate_pose_buffer(
                self._odom_buffer,
                stamp,
                direct_camera_pose=False,
            )
            if odom_c2w is None or odom_pose_dt is None or odom_pose_dt > self.align_direct_camera_pose_to_odom_max_dt_sec:
                return None
            self._direct_pose_to_odom_world = odom_c2w @ np.linalg.inv(direct_c2w)
            self._direct_pose_to_odom_world_stamp = float(stamp)
            self._direct_pose_to_odom_world_pose_dt = float(odom_pose_dt)
            self._direct_pose_to_odom_world_match_type = str(odom_match_type)
            yaw_deg = float(
                np.degrees(
                    np.arctan2(
                        self._direct_pose_to_odom_world[1, 0],
                        self._direct_pose_to_odom_world[0, 0],
                    )
                )
            )
            self.get_logger().info(
                "Aligned direct camera pose world to FAST-LIVO2 world. "
                f"stamp={stamp:.6f}, odom_dt={odom_pose_dt:.4f}s, match_type={odom_match_type}, yaw_deg={yaw_deg:.2f}"
            )

        return self._direct_pose_to_odom_world @ direct_c2w

    def _interpolate_pose_buffer(
        self,
        buffer: Deque[BufferedOdom],
        stamp: float,
        *,
        direct_camera_pose: bool,
    ) -> tuple[Optional[np.ndarray], Optional[float], str, Optional[float]]:
        if not buffer:
            return None, None, "missing", None

        if len(buffer) == 1:
            entry = buffer[0]
            dt = abs(entry.stamp - stamp)
            max_dt = self.sync_slop_sec
            if direct_camera_pose and self.require_exact_direct_camera_pose:
                max_dt = self.direct_camera_pose_exact_tolerance_sec
            if dt <= max_dt:
                c2w = self._pose_matrix_from_entry(entry, direct_camera_pose=direct_camera_pose)
                return c2w, dt, "exact_match", dt
            return None, None, "missing", None

        prev_entry = None
        next_entry = None
        for entry in buffer:
            if entry.stamp <= stamp:
                prev_entry = entry
            if entry.stamp >= stamp:
                next_entry = entry
                break

        if prev_entry is None or next_entry is None:
            return None, None, "missing", None

        exact_tol = 1e-4
        if direct_camera_pose and self.require_exact_direct_camera_pose:
            exact_tol = self.direct_camera_pose_exact_tolerance_sec

        if abs(prev_entry.stamp - stamp) <= exact_tol:
            c2w = self._pose_matrix_from_entry(prev_entry, direct_camera_pose=direct_camera_pose)
            return c2w, 0.0, "exact_match", 0.0

        if abs(next_entry.stamp - stamp) <= exact_tol:
            c2w = self._pose_matrix_from_entry(next_entry, direct_camera_pose=direct_camera_pose)
            return c2w, 0.0, "exact_match", 0.0

        if direct_camera_pose and self.require_exact_direct_camera_pose:
            return None, None, "missing", None

        if next_entry.stamp - prev_entry.stamp > 2.0 * self.sync_slop_sec:
            return None, None, "missing", None

        ratio = float((stamp - prev_entry.stamp) / max(next_entry.stamp - prev_entry.stamp, 1e-6))
        translation = prev_entry.translation + ratio * (next_entry.translation - prev_entry.translation)
        quaternion = quaternion_slerp(prev_entry.quaternion_xyzw, next_entry.quaternion_xyzw, ratio)
        rotation = quaternion_to_rotation_matrix(*quaternion)
        interpolated_entry = BufferedOdom(
            stamp=stamp,
            rotation=rotation,
            translation=translation,
            quaternion_xyzw=quaternion,
        )
        c2w = self._pose_matrix_from_entry(interpolated_entry, direct_camera_pose=direct_camera_pose)
        pose_dt = float(min(abs(stamp - prev_entry.stamp), abs(next_entry.stamp - stamp)))
        return c2w, pose_dt, "interpolated", pose_dt

    def _pose_matrix_from_entry(self, entry: BufferedOdom, *, direct_camera_pose: bool) -> np.ndarray:
        if direct_camera_pose:
            rotation = entry.rotation
            if self.direct_camera_pose_apply_isaac_optical_fix:
                # Isaac/USD camera prim pose is expressed in camera axes (+X right, +Y up, -Z forward),
                # while exporter RGB-D unprojection expects optical axes (+X right, +Y down, +Z forward).
                rotation = rotation @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            return self._pose_matrix_from_rotation_translation(rotation, entry.translation)
        return self._camera_to_world_matrix(entry.rotation, entry.translation)

    def _pose_matrix_from_rotation_translation(self, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation.astype(np.float32, copy=False)
        c2w[:3, 3] = translation.astype(np.float32, copy=False)
        return c2w

    def _camera_to_world_matrix(self, pose_rotation: np.ndarray, pose_translation: np.ndarray) -> np.ndarray:
        sample_camera = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        sample_world = self._transform_camera_points_to_world(sample_camera, pose_rotation, pose_translation)
        origin = sample_world[0]
        rotation = np.column_stack((sample_world[1] - origin, sample_world[2] - origin, sample_world[3] - origin))
        u, _, vt = np.linalg.svd(rotation.astype(np.float64, copy=False))
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = (u @ vt).astype(np.float32)
        c2w[:3, 3] = origin.astype(np.float32)
        return c2w

    def _transform_camera_points_to_world(
        self,
        points_camera: np.ndarray,
        pose_rotation: np.ndarray,
        pose_translation: np.ndarray,
    ) -> np.ndarray:
        points_lidar = (points_camera - self.p_cl) @ self.r_cl
        points_imu = points_lidar @ self.r_li.T + self.p_li
        return points_imu @ pose_rotation.T + pose_translation

    def _try_export_frame(self, image_stamp: float) -> bool:
        if not self._camera_info_received:
            return False

        if self._last_processed_candidate_stamp is not None and image_stamp <= self._last_processed_candidate_stamp + 1e-6:
            return True

        if self.max_frames > 0 and self._saved_frame_count >= self.max_frames:
            if not self._stopped_for_max_frames:
                self.get_logger().info(f"LIVO2 exporter reached max_frames={self.max_frames}.")
                self._stopped_for_max_frames = True
            return True

        image = self._find_closest(self._image_buffer, image_stamp)
        depth = self._find_closest(self._depth_buffer, image_stamp)
        if (
            image is not None
            and depth is not None
            and self.require_exact_rgbd_sync
            and abs(depth.stamp - image.stamp) > self.rgbd_exact_tolerance_sec
        ):
            depth = None
        if image is None or depth is None:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self._last_wait_log_sec > 2.0:
                missing = []
                if image is None:
                    missing.append("image")
                if depth is None:
                    missing.append("depth")
                self.get_logger().warn(
                    "LIVO2 exporter is waiting for synchronized inputs. "
                    f"missing={','.join(missing) if missing else 'unknown'}, "
                    f"sync_slop_sec={self.sync_slop_sec:.3f}, "
                    f"rgb_buffer={len(self._image_buffer)}, depth_buffer={len(self._depth_buffer)}, "
                    f"require_exact_rgbd_sync={self.require_exact_rgbd_sync}."
                )
                self._last_wait_log_sec = now_sec
            return False

        c2w, pose_dt, match_type, aligned_pose_dt, pose_source = self._interpolate_pose(image_stamp)
        pose_ok = c2w is not None
        if not pose_ok:
            pose_stamp_candidates = (
                (self._latest_camera_pose_stamp,)
                if self.require_direct_camera_pose
                else (self._latest_camera_pose_stamp, self._latest_odom_stamp)
            )
            latest_pose_stamp = (
                max(value for value in pose_stamp_candidates if value is not None)
                if any(value is not None for value in pose_stamp_candidates)
                else None
            )
            if latest_pose_stamp is None or latest_pose_stamp < image_stamp + self.sync_slop_sec:
                return False

        if self.use_rgbd_local_cloud:
            local_submap = self._build_rgbd_local_submap(depth.depth_m, c2w, image_stamp) if pose_ok else None
        else:
            local_submap = self._build_local_submap(image_stamp)
        if local_submap is None:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self._last_wait_log_sec > 2.0:
                if self.use_rgbd_local_cloud:
                    self.get_logger().warn(
                        "LIVO2 exporter is waiting for a valid RGB-D local cloud from depth + exact camera pose."
                    )
                else:
                    self.get_logger().warn("LIVO2 exporter is waiting for a synchronized local submap window.")
                self._last_wait_log_sec = now_sec
            return False
        monotonic = self._last_processed_candidate_stamp is None or image_stamp > self._last_processed_candidate_stamp
        if not monotonic:
            self._num_frames_monotonic_rejected += 1
        self._synced_frame_count += 1

        if pose_ok and aligned_pose_dt is not None:
            self._aligned_pose_dt_samples.append(aligned_pose_dt)
            if match_type == "interpolated":
                self._num_frames_interpolated += 1
            else:
                self._num_frames_exact_match += 1
        else:
            self._num_pose_missing += 1

        depth_valid_ratio = float(np.mean(np.isfinite(depth.depth_m) & (depth.depth_m > 0.05)))
        image_blur = blur_score(image.image_rgb)
        coverage_score, coverage_grid = coverage_novelty(depth.depth_m, self._last_keyframe_coverage, self.coverage_grid_shape)

        aligned_entry = {
            "frame_id": f"candidate_{self._synced_frame_count:06d}",
            "timestamp_sec": image_stamp,
            "rgb_stamp_sec": image.stamp,
            "depth_stamp_sec": depth.stamp,
            "pose_source": pose_source,
            "pose_alignment": match_type,
            "pose_ok": pose_ok,
            "pose_dt_sec": pose_dt,
            "timestamp_monotonic": monotonic,
            "depth_valid_ratio": depth_valid_ratio,
            "blur_score": image_blur,
            "coverage_novelty": coverage_score,
            "T_world_cam": c2w.tolist() if pose_ok else None,
        }
        self._write_jsonl(self._aligned_file, aligned_entry)
        self._update_alignment_summary()

        quality_reasons: list[str] = []
        if not pose_ok:
            quality_reasons.append("pose_missing")
        if not monotonic:
            quality_reasons.append("timestamp_non_monotonic")
        if not self.force_every_synced_frame:
            if depth_valid_ratio < self.depth_valid_ratio_thresh:
                quality_reasons.append("depth_valid_ratio")
            if image_blur < self.blur_score_min:
                quality_reasons.append("blur")

        translation = 0.0
        rotation_deg = 0.0
        time_gap = float("inf") if self._last_keyframe_stamp is None else image_stamp - self._last_keyframe_stamp
        if self._last_keyframe_c2w is not None and pose_ok:
            translation = float(np.linalg.norm(c2w[:3, 3] - self._last_keyframe_c2w[:3, 3]))
            rotation_deg = rotation_angle_deg(self._last_keyframe_c2w[:3, :3], c2w[:3, :3])

        accepted_entry = {
            "frame_id": f"candidate_{self._synced_frame_count:06d}",
            "timestamp_sec": image_stamp,
            "rgb_path": None,
            "depth_path": None,
            "pose_source": pose_source,
            "T_world_cam": c2w.tolist() if pose_ok else None,
            "is_keyframe": False,
            "reason": [],
            "translation_from_prev_keyframe_m": translation,
            "rotation_from_prev_keyframe_deg": rotation_deg,
            "depth_valid_ratio": depth_valid_ratio,
            "blur_score": image_blur,
            "coverage_novelty": coverage_score,
            "semantic_novelty": 0.0,
            "pose_ok": pose_ok,
            "pose_alignment": match_type,
            "timestamp_monotonic": monotonic,
            "quality_reject_reasons": quality_reasons,
        }

        if quality_reasons:
            self._num_quality_rejected += 1
            self._write_jsonl(self._accepted_file, accepted_entry)
            self._update_keyframe_summary()
            self._last_processed_candidate_stamp = image_stamp
            return True

        if not self.force_every_synced_frame and self._last_keyframe_stamp is not None and time_gap < self.min_time_gap_sec:
            accepted_entry["quality_reject_reasons"] = ["min_time_gap"]
            self._write_jsonl(self._accepted_file, accepted_entry)
            self._update_keyframe_summary()
            self._last_processed_candidate_stamp = image_stamp
            return True

        semantic_trigger, semantic_novelty, semantic_reasons = self._semantic_trigger_state(image_stamp)
        accepted_entry["semantic_novelty"] = float(semantic_novelty)

        reasons = []
        if self.force_every_synced_frame:
            reasons.append("every_frame")
            if self._last_keyframe_stamp is None:
                reasons.append("bootstrap")
        else:
            if self._last_keyframe_stamp is None:
                reasons.append("bootstrap")
            if translation >= self.translation_thresh_m:
                reasons.append("translation")
            if rotation_deg >= self.rotation_thresh_deg and translation >= self.min_translation_for_rotation_trigger_m:
                reasons.append("rotation")
            if coverage_score >= self.coverage_novelty_thresh:
                reasons.append("coverage")
            if semantic_trigger and translation >= self.semantic_trigger_min_translation_m:
                reasons.append("semantic_under_supported")
            if (
                self._last_keyframe_stamp is not None
                and time_gap >= self.max_time_gap_sec
                and translation >= self.min_translation_for_time_trigger_m
            ):
                reasons.append("time")

        if not reasons:
            accepted_entry["quality_reject_reasons"] = ["no_trigger"]
            self._write_jsonl(self._accepted_file, accepted_entry)
            self._update_keyframe_summary()
            self._last_processed_candidate_stamp = image_stamp
            return True

        if "translation" in reasons:
            self._translation_trigger_count += 1
        if "rotation" in reasons:
            self._rotation_trigger_count += 1
        if "coverage" in reasons:
            self._coverage_trigger_count += 1
        if "time" in reasons:
            self._time_trigger_count += 1
        for reason in reasons:
            self._selection_reasons[reason] += 1
        if "semantic_under_supported" in reasons:
            self._last_semantic_trigger_stamp = image_stamp

        persist_allowed, persist_reasons, dynamic_detection_hits = self._dynamic_semantic_persist_gate(
            image.image_rgb,
            image_stamp,
        )
        accepted_entry["dynamic_semantic_detection_hits"] = int(dynamic_detection_hits)
        accepted_entry["dynamic_semantic_persist_reasons"] = persist_reasons
        if not persist_allowed:
            accepted_entry["quality_reject_reasons"] = ["dynamic_semantic_gate"]
            self._write_jsonl(self._accepted_file, accepted_entry)
            self._update_keyframe_summary()
            self._last_processed_candidate_stamp = image_stamp
            return True

        image_name, depth_name, local_cloud_packet = self._save_frame(
            image.image_rgb, depth.depth_m, c2w, image.stamp, depth.stamp, image_stamp, local_submap
        )
        accepted_entry["is_keyframe"] = True
        accepted_entry["reason"] = reasons
        accepted_entry["semantic_trigger_reasons"] = semantic_reasons
        accepted_entry["rgb_path"] = str((self.results_dir / image_name).relative_to(self.scene_dir))
        accepted_entry["depth_path"] = str((self.results_dir / depth_name).relative_to(self.scene_dir))
        accepted_entry["local_cloud_ref"] = local_cloud_packet.local_cloud_id
        accepted_entry["local_cloud_frame"] = local_cloud_packet.frame
        accepted_entry["sensor_config_id"] = self.sensor_config_id
        accepted_entry["calib_version"] = self.calib_version
        accepted_entry["quality_reject_reasons"] = []
        self._write_jsonl(self._accepted_file, accepted_entry)
        self._write_jsonl(self._keyframes_file, accepted_entry)
        self._write_keyframe_packet(
            c2w=c2w,
            image_stamp=image.stamp,
            image_name=image_name,
            depth_name=depth_name,
            local_cloud_packet=local_cloud_packet,
            reasons=reasons,
            pose_source=pose_source,
            pose_alignment=match_type,
            pose_dt=pose_dt,
        )
        self._update_keyframe_summary()
        self._last_processed_candidate_stamp = image_stamp

        self._last_keyframe_stamp = image_stamp
        self._last_keyframe_c2w = c2w.copy()
        self._last_keyframe_coverage = coverage_grid.copy()
        return True

    def _save_frame(
        self,
        image_rgb: np.ndarray,
        depth_m: np.ndarray,
        c2w: np.ndarray,
        image_stamp: float,
        depth_stamp: float,
        pose_stamp: float,
        local_submap: BufferedCloudSubmap,
    ) -> tuple[str, str, LocalCloudPacket]:
        if image_rgb.shape[:2] != depth_m.shape[:2]:
            depth_m = cv2.resize(depth_m, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        image_name = f"frame{self._frame_index:06d}.jpg"
        depth_name = f"depth{self._frame_index:06d}.png"
        image_path = self.results_dir / image_name
        depth_path = self.results_dir / depth_name

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(image_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
            raise RuntimeError(f"Failed to write image frame: {image_path}")
        export_image_path = self.export_results_dir / image_name
        self._mirror_file(image_path, export_image_path)

        depth_clean = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        if self.max_depth_m > 0.0:
            depth_clean = depth_clean.copy()
            depth_clean[depth_clean > self.max_depth_m] = 0.0
        depth_uint16 = np.clip(np.rint(depth_clean * self.output_depth_scale), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        if not cv2.imwrite(str(depth_path), depth_uint16):
            raise RuntimeError(f"Failed to write depth frame: {depth_path}")
        export_depth_path = self.export_results_dir / depth_name
        self._mirror_file(depth_path, export_depth_path)

        thumbnail_path = self.thumbnails_dir / f"thumb_{self._frame_index:06d}.jpg"
        thumb = cv2.resize(image_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumbnail_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        semantic_local_cloud = None
        if self.semantic_local_cloud_enabled:
            trusted_cloud = self._find_closest_semantic_cloud(image_stamp)
            trusted_xyz = None if trusted_cloud is None else trusted_cloud.xyz
            semantic_local_cloud = self._build_semantic_local_cloud(local_submap.xyz, c2w, depth_clean, trusted_xyz_world=trusted_xyz)
            if semantic_local_cloud is not None:
                semantic_local_cloud.anchor_stamp = float(local_submap.anchor_stamp)
                semantic_local_cloud.stamp_start = float(local_submap.stamp_start)
                semantic_local_cloud.stamp_end = float(local_submap.stamp_end)
                semantic_local_cloud.frame_id = str(local_submap.frame_id)
                semantic_local_cloud.source_scan_ids = list(local_submap.source_scan_ids)

        local_cloud_id = f"local_cloud_{self._frame_index:06d}"
        local_cloud_path = (self.local_clouds_dir / f"{local_cloud_id}.npz").resolve()
        save_payload: dict[str, np.ndarray] = {"xyz": local_submap.xyz.astype(np.float32)}
        if local_submap.quality_fields_present:
            if local_submap.uv is not None:
                save_payload["projected_uv"] = local_submap.uv.astype(np.int32)
            if local_submap.projected_depth is not None:
                save_payload["projected_depth"] = local_submap.projected_depth.astype(np.float32)
            if local_submap.depth_residual is not None:
                save_payload["depth_residual"] = local_submap.depth_residual.astype(np.float32)
            if local_submap.zbuffer_rank is not None:
                save_payload["zbuffer_rank"] = local_submap.zbuffer_rank.astype(np.int32)
            if local_submap.distance_to_depth_edge is not None:
                save_payload["distance_to_depth_edge"] = local_submap.distance_to_depth_edge.astype(np.float32)
            if local_submap.visibility_score is not None:
                save_payload["visibility_score"] = local_submap.visibility_score.astype(np.float32)
                save_payload["visibility_flag"] = (local_submap.visibility_score > 0.0).astype(np.uint8)
            if local_submap.quality_score is not None:
                save_payload["quality_score"] = local_submap.quality_score.astype(np.float32)
        if semantic_local_cloud is not None and semantic_local_cloud.source_point_indices is not None:
            save_payload["semantic_point_indices"] = semantic_local_cloud.source_point_indices.astype(np.int32)
            if semantic_local_cloud.uv is not None:
                save_payload["semantic_projected_uv"] = semantic_local_cloud.uv.astype(np.int32)
            if semantic_local_cloud.projected_depth is not None:
                save_payload["semantic_projected_depth"] = semantic_local_cloud.projected_depth.astype(np.float32)
            if semantic_local_cloud.depth_residual is not None:
                save_payload["semantic_depth_residual"] = semantic_local_cloud.depth_residual.astype(np.float32)
            if semantic_local_cloud.zbuffer_rank is not None:
                save_payload["semantic_zbuffer_rank"] = semantic_local_cloud.zbuffer_rank.astype(np.int32)
            if semantic_local_cloud.distance_to_depth_edge is not None:
                save_payload["semantic_distance_to_depth_edge"] = semantic_local_cloud.distance_to_depth_edge.astype(np.float32)
            if semantic_local_cloud.visibility_score is not None:
                save_payload["semantic_visibility_score"] = semantic_local_cloud.visibility_score.astype(np.float32)
                save_payload["semantic_visibility_flag"] = (semantic_local_cloud.visibility_score > 0.0).astype(np.uint8)
            if semantic_local_cloud.quality_score is not None:
                save_payload["semantic_quality_score"] = semantic_local_cloud.quality_score.astype(np.float32)
        np.savez_compressed(local_cloud_path, **save_payload)
        export_local_cloud_path = (self.export_local_clouds_dir / f"{local_cloud_id}.npz").resolve()
        self._mirror_file(local_cloud_path, export_local_cloud_path)
        local_cloud_packet = LocalCloudPacket(
            local_cloud_id=local_cloud_id,
            source_scan_ids=list(local_submap.source_scan_ids),
            stamp_start=float(local_submap.stamp_start),
            stamp_end=float(local_submap.stamp_end),
            frame=local_submap.frame_id,
            cloud_path=str(export_local_cloud_path),
            point_count=int(local_submap.xyz.shape[0]),
            parent_submap_id=None,
            cloud_kind=str(local_submap.cloud_kind),
            quality_fields_present=bool(local_submap.quality_fields_present or semantic_local_cloud is not None),
            has_uv=(local_submap.uv is not None) or (semantic_local_cloud is not None and semantic_local_cloud.uv is not None),
            has_normal=False,
            has_visibility_flag=(local_submap.visibility_score is not None)
            or (semantic_local_cloud is not None and semantic_local_cloud.visibility_score is not None),
        )
        append_local_cloud_packet(self.local_cloud_packets_path, local_cloud_packet)

        flattened = " ".join(f"{value:.9f}" for value in c2w.reshape(-1))
        self._traj_file.write(flattened + "\n")
        self._traj_file.flush()

        translation = 0.0
        rotation_deg = 0.0
        if self._last_keyframe_c2w is not None:
            translation = float(np.linalg.norm(c2w[:3, 3] - self._last_keyframe_c2w[:3, 3]))
            rotation_deg = rotation_angle_deg(self._last_keyframe_c2w[:3, :3], c2w[:3, :3])

        self._frame_index_writer.writerow(
            [
                self._frame_index,
                f"{image_stamp:.9f}",
                f"{depth_stamp:.9f}",
                f"{pose_stamp:.9f}",
                "selected",
                f"{translation:.6f}",
                f"{rotation_deg:.6f}",
                "",
                image_name,
                depth_name,
            ]
        )
        self._frame_index_file.flush()

        self._frame_index += 1
        self._saved_frame_count += 1
        if self._saved_frame_count == 1 or self._saved_frame_count % 20 == 0:
            self.get_logger().info(
                f"LIVO2 keyframe exporter saved {self._saved_frame_count} keyframes to {self.scene_dir}."
            )
        return image_name, depth_name, local_cloud_packet

    def _mirror_file(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _write_keyframe_packet(
        self,
        c2w: np.ndarray,
        image_stamp: float,
        image_name: str,
        depth_name: str,
        local_cloud_packet: LocalCloudPacket,
        reasons: list[str],
        pose_source: str,
        pose_alignment: str,
        pose_dt: Optional[float],
    ) -> None:
        t_world_cam = [float(v) for v in c2w.reshape(-1).tolist()]
        t_world_body = self._body_pose_matrix(c2w).reshape(-1).tolist()
        t_world_lidar = self._lidar_pose_matrix(c2w).reshape(-1).tolist()
        packet = KeyframePacket(
            keyframe_id=int(self._frame_index - 1),
            stamp_sec=float(image_stamp),
            t_world_body=[float(v) for v in t_world_body],
            t_world_lidar=[float(v) for v in t_world_lidar],
            t_world_cam=[float(v) for v in t_world_cam],
            rgb_path=str((self.export_results_dir / image_name).resolve()),
            depth_path=str((self.export_results_dir / depth_name).resolve()),
            local_cloud_ref=local_cloud_packet.local_cloud_id,
            local_cloud_frame=local_cloud_packet.frame,
            sensor_config_id=self.sensor_config_id,
            calib_version=self.calib_version,
            pose_source=pose_source,
            status="raw",
            source_scan_ids=list(local_cloud_packet.source_scan_ids),
            selection_reasons=[str(v) for v in reasons],
            pose_alignment=pose_alignment,
            pose_dt_sec=None if pose_dt is None else float(pose_dt),
        )
        append_keyframe_packet(self.keyframe_packets_path, packet)

    def _body_pose_matrix(self, c2w: np.ndarray) -> np.ndarray:
        t_cam_lidar = -(self.r_cl.T @ self.p_cl.reshape(3, 1)).reshape(3)
        t_lidar_imu = self.p_li.astype(np.float32)
        t_cam_imu = t_cam_lidar + t_lidar_imu
        out = c2w.copy()
        out[:3, 3] = (c2w[:3, :3] @ (-t_cam_imu.reshape(3, 1))).reshape(3) + c2w[:3, 3]
        return out

    def _lidar_pose_matrix(self, c2w: np.ndarray) -> np.ndarray:
        t_cam_lidar = -(self.r_cl.T @ self.p_cl.reshape(3, 1)).reshape(3)
        out = c2w.copy()
        out[:3, 3] = (c2w[:3, :3] @ (-t_cam_lidar.reshape(3, 1))).reshape(3) + c2w[:3, 3]
        out[:3, :3] = c2w[:3, :3] @ self.r_cl.T
        return out

    def destroy_node(self) -> bool:
        try:
            if self._dynamic_semantic_client is not None:
                self._dynamic_semantic_client.close()
            self._update_trajectory_summary()
            self._update_alignment_summary()
            self._update_keyframe_summary()
            self._write_manifest()
            for handle_name in [
                "_traj_file",
                "_frame_index_file",
                "_trajectory_file",
                "_aligned_file",
                "_accepted_file",
                "_keyframes_file",
            ]:
                handle = getattr(self, handle_name, None)
                if handle is not None and not handle.closed:
                    handle.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Livo2OVOKeyframeExporter()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pragma: no cover - defensive cleanup path for forced shutdown
        if "context is not valid" not in str(exc):
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
