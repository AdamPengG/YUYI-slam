from __future__ import annotations

import csv
import json
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
from sensor_msgs.msg import CameraInfo, Image


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


class Livo2OVOKeyframeExporter(Node):
    def __init__(self) -> None:
        super().__init__("livo2_ovo_keyframe_exporter")
        self.bridge = CvBridge()

        self._declare_parameters()
        self._load_parameters()

        self._image_buffer: Deque[BufferedImage] = deque(maxlen=300)
        self._depth_buffer: Deque[BufferedDepth] = deque(maxlen=300)
        self._odom_buffer: Deque[BufferedOdom] = deque(maxlen=1000)

        self._camera_info_received = False
        self._camera_info_warning_emitted = False
        self._config_written = False
        self._latest_sensor_stamp: Optional[float] = None
        self._latest_odom_stamp: Optional[float] = None
        self._odom_time_offset: Optional[float] = None
        self._stopped_for_max_frames = False
        self._last_wait_log_sec = 0.0
        self._last_processed_candidate_stamp: Optional[float] = None
        self._last_keyframe_stamp: Optional[float] = None
        self._last_keyframe_c2w: Optional[np.ndarray] = None
        self._last_keyframe_coverage: Optional[np.ndarray] = None
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

        self.scene_dir = self.output_root / self.scene_name
        self.results_dir = self.scene_dir / "results"
        self.scene_config_path = self.config_root / f"{self.scene_name}.yaml"
        self.metadata_path = self.scene_dir / "export_info.yaml"
        self.traj_path = self.scene_dir / "traj.txt"
        self.frame_index_path = self.scene_dir / "frames_index.csv"

        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.scene_name}"
        self.run_dir = self.run_root / self.run_id
        self.export_dir = self.run_dir / "export"
        self.thumbnails_dir = self.export_dir / "thumbnails"

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

        self.image_sub = self.create_subscription(Image, self.image_topic, self._image_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self._depth_callback, qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 100)

        self.get_logger().info(
            "LIVO2 OVO keyframe exporter ready. "
            f"scene={self.scene_name}, output={self.scene_dir}, run={self.run_dir}, "
            f"image={self.image_topic}, depth={self.depth_topic}, odom={self.odom_topic}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "common.img_topic": "/robot_rgb",
            "extrin_calib.Rcl": [0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
            "extrin_calib.Pcl": [-0.000000329, 0.619520000, -0.001662359],
            "extrin_calib.extrinsic_T": [0.0, 0.0, 0.104],
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
            "export.output_root": "/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica",
            "export.config_root": "/home/peng/isacc_slam/reference/OVO/data/working/configs/Replica",
            "export.run_root": "/home/peng/isacc_slam/runs/ovo_pose_keyframes",
            "export.scene_name": "isaac_turtlebot3_livo2",
            "export.sync_slop_sec": 0.08,
            "export.depth_input_scale": 1000.0,
            "export.output_depth_scale": 1000.0,
            "export.jpeg_quality": 95,
            "export.max_frames": -1,
            "export.overwrite_scene": True,
            "selector.min_time_gap_sec": 0.4,
            "selector.max_time_gap_sec": 2.0,
            "selector.translation_thresh_m": 0.35,
            "selector.rotation_thresh_deg": 10.0,
            "selector.depth_valid_ratio_thresh": 0.60,
            "selector.blur_score_min": 30.0,
            "selector.coverage_novelty_thresh": 0.20,
            "selector.coverage_grid_width": 24,
            "selector.coverage_grid_height": 18,
            "selector.min_translation_for_rotation_trigger_m": 0.05,
            "selector.min_translation_for_time_trigger_m": 0.05,
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
        self.output_root = Path(str(self.get_parameter("export.output_root").value)).expanduser()
        self.config_root = Path(str(self.get_parameter("export.config_root").value)).expanduser()
        self.run_root = Path(str(self.get_parameter("export.run_root").value)).expanduser()
        self.scene_name = str(self.get_parameter("export.scene_name").value)
        self.sync_slop_sec = float(self.get_parameter("export.sync_slop_sec").value)
        self.depth_input_scale = float(self.get_parameter("export.depth_input_scale").value)
        self.output_depth_scale = float(self.get_parameter("export.output_depth_scale").value)
        self.jpeg_quality = int(self.get_parameter("export.jpeg_quality").value)
        self.max_frames = int(self.get_parameter("export.max_frames").value)
        self.overwrite_scene = bool(self.get_parameter("export.overwrite_scene").value)

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
        self.min_translation_for_rotation_trigger_m = float(
            self.get_parameter("selector.min_translation_for_rotation_trigger_m").value
        )
        self.min_translation_for_time_trigger_m = float(
            self.get_parameter("selector.min_translation_for_time_trigger_m").value
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

        if self.overwrite_scene and self.scene_dir.exists():
            for path in [self.results_dir, self.traj_path, self.metadata_path, self.frame_index_path]:
                if path.is_dir():
                    for child in path.glob("*"):
                        if child.is_file():
                            child.unlink()
                    path.rmdir()
                elif path.exists():
                    path.unlink()

        self.results_dir.mkdir(parents=True, exist_ok=True)

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
            "pose_source": "fast_livo2",
            "pose_topic": self.odom_topic,
            "image_topic": self.image_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "intrinsics_source": "camera_info" if self.override_intrinsics_from_camera_info else "fast_livo_config",
            "selector": {
                "min_time_gap_sec": self.min_time_gap_sec,
                "max_time_gap_sec": self.max_time_gap_sec,
                "translation_thresh_m": self.translation_thresh_m,
                "rotation_thresh_deg": self.rotation_thresh_deg,
                "depth_valid_ratio_thresh": self.depth_valid_ratio_thresh,
                "blur_score_min": self.blur_score_min,
                "coverage_novelty_thresh": self.coverage_novelty_thresh,
                "coverage_grid_shape": list(self.coverage_grid_shape),
                "min_translation_for_rotation_trigger_m": self.min_translation_for_rotation_trigger_m,
                "min_translation_for_time_trigger_m": self.min_translation_for_time_trigger_m,
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

    def _write_json(self, path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _write_jsonl(self, handle, payload: dict) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    def _write_manifest(self) -> None:
        manifest = f"""# FAST-LIVO2 to OVO Keyframe Export

- run_id: `{self.run_id}`
- scene_name: `{self.scene_name}`
- pose authority: `FAST-LIVO2` via `{self.odom_topic}`
- image topic: `{self.image_topic}`
- depth topic: `{self.depth_topic}`
- intrinsics source: `{"camera_info" if self.override_intrinsics_from_camera_info else "fast_livo_config"}`
- keyframe policy: ORB-like but independent of ORB-SLAM3

This run exports Replica-style OVO inputs under:
- dataset root: `{self.scene_dir}`
- run root: `{self.run_dir}`

Selection logic:
- quality gates: pose present, monotonic timestamp, depth valid ratio, blur score
- triggers: translation, rotation, coverage novelty, max time gap

Downstream OVO immediate contract:
- `results/frame*.jpg`
- `results/depth*.png`
- `traj.txt`
- scene config yaml with intrinsics and `depth_scale`
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
            ],
            "present_now": {
                "results/frame*.jpg": True,
                "results/depth*.png": True,
                "traj.txt": True,
                "scene_config(cam.H,W,fx,fy,cx,cy,depth_scale)": True,
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
            }
        }
        with self.scene_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(scene_config, handle, sort_keys=False)
        self._config_written = True

    def _update_trajectory_summary(self) -> None:
        mean_dt = float(np.mean(self._trajectory_dt_samples)) if self._trajectory_dt_samples else 0.0
        max_dt = float(np.max(self._trajectory_dt_samples)) if self._trajectory_dt_samples else 0.0
        payload = {
            "pose_topic": self.odom_topic,
            "message_type": "nav_msgs/msg/Odometry",
            "num_poses": self._num_poses_received,
            "first_timestamp": self._odom_buffer[0].stamp if self._odom_buffer else None,
            "last_timestamp": self._odom_buffer[-1].stamp if self._odom_buffer else None,
            "mean_dt": mean_dt,
            "max_dt": max_dt,
            "num_non_monotonic_poses_dropped": self._num_pose_non_monotonic,
            "num_pre_axis_poses_dropped": self._num_pose_axis_wait_drops,
            "frame_convention": "FAST-LIVO2 /aft_mapped_to_init published in camera_init->aft_mapped tree",
            "camera_extrinsic_available": True,
            "status": "ok" if self._odom_buffer else "waiting_for_pose",
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
        self._depth_buffer.append(BufferedDepth(stamp=stamp, depth_m=depth_m))
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

    def _interpolate_pose(self, stamp: float) -> tuple[Optional[np.ndarray], Optional[float], str, Optional[float]]:
        if not self._odom_buffer:
            return None, None, "missing", None

        if len(self._odom_buffer) == 1:
            entry = self._odom_buffer[0]
            dt = abs(entry.stamp - stamp)
            if dt <= self.sync_slop_sec:
                c2w = self._camera_to_world_matrix(entry.rotation, entry.translation)
                return c2w, dt, "exact_match", dt
            return None, None, "missing", None

        prev_entry = None
        next_entry = None
        for entry in self._odom_buffer:
            if entry.stamp <= stamp:
                prev_entry = entry
            if entry.stamp >= stamp:
                next_entry = entry
                break

        if prev_entry is None or next_entry is None:
            return None, None, "missing", None

        if abs(prev_entry.stamp - stamp) <= 1e-4:
            c2w = self._camera_to_world_matrix(prev_entry.rotation, prev_entry.translation)
            return c2w, 0.0, "exact_match", 0.0

        if abs(next_entry.stamp - stamp) <= 1e-4:
            c2w = self._camera_to_world_matrix(next_entry.rotation, next_entry.translation)
            return c2w, 0.0, "exact_match", 0.0

        if next_entry.stamp - prev_entry.stamp > 2.0 * self.sync_slop_sec:
            return None, None, "missing", None

        ratio = float((stamp - prev_entry.stamp) / max(next_entry.stamp - prev_entry.stamp, 1e-6))
        translation = prev_entry.translation + ratio * (next_entry.translation - prev_entry.translation)
        quaternion = quaternion_slerp(prev_entry.quaternion_xyzw, next_entry.quaternion_xyzw, ratio)
        rotation = quaternion_to_rotation_matrix(*quaternion)
        c2w = self._camera_to_world_matrix(rotation, translation)
        pose_dt = float(min(abs(stamp - prev_entry.stamp), abs(next_entry.stamp - stamp)))
        return c2w, pose_dt, "interpolated", pose_dt

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
        if image is None or depth is None:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self._last_wait_log_sec > 2.0:
                self.get_logger().warn("LIVO2 exporter is waiting for synchronized image/depth/camera_info.")
                self._last_wait_log_sec = now_sec
            return False

        c2w, pose_dt, match_type, aligned_pose_dt = self._interpolate_pose(image_stamp)
        pose_ok = c2w is not None
        if not pose_ok:
            if self._latest_odom_stamp is None or self._latest_odom_stamp < image_stamp + self.sync_slop_sec:
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
            "pose_source": "fast_livo2",
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
            "pose_source": "fast_livo2",
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

        if self._last_keyframe_stamp is not None and time_gap < self.min_time_gap_sec:
            accepted_entry["quality_reject_reasons"] = ["min_time_gap"]
            self._write_jsonl(self._accepted_file, accepted_entry)
            self._update_keyframe_summary()
            self._last_processed_candidate_stamp = image_stamp
            return True

        reasons = []
        if self._last_keyframe_stamp is None:
            reasons.append("bootstrap")
        if translation >= self.translation_thresh_m:
            reasons.append("translation")
        if rotation_deg >= self.rotation_thresh_deg and translation >= self.min_translation_for_rotation_trigger_m:
            reasons.append("rotation")
        if coverage_score >= self.coverage_novelty_thresh:
            reasons.append("coverage")
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

        image_name, depth_name = self._save_frame(image.image_rgb, depth.depth_m, c2w, image.stamp, depth.stamp, image_stamp)
        accepted_entry["is_keyframe"] = True
        accepted_entry["reason"] = reasons
        accepted_entry["rgb_path"] = str((self.results_dir / image_name).relative_to(self.scene_dir))
        accepted_entry["depth_path"] = str((self.results_dir / depth_name).relative_to(self.scene_dir))
        accepted_entry["quality_reject_reasons"] = []
        self._write_jsonl(self._accepted_file, accepted_entry)
        self._write_jsonl(self._keyframes_file, accepted_entry)
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
    ) -> tuple[str, str]:
        if image_rgb.shape[:2] != depth_m.shape[:2]:
            depth_m = cv2.resize(depth_m, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        image_name = f"frame{self._frame_index:06d}.jpg"
        depth_name = f"depth{self._frame_index:06d}.png"
        image_path = self.results_dir / image_name
        depth_path = self.results_dir / depth_name

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(image_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
            raise RuntimeError(f"Failed to write image frame: {image_path}")

        depth_clean = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_uint16 = np.clip(np.rint(depth_clean * self.output_depth_scale), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        if not cv2.imwrite(str(depth_path), depth_uint16):
            raise RuntimeError(f"Failed to write depth frame: {depth_path}")

        thumbnail_path = self.thumbnails_dir / f"thumb_{self._frame_index:06d}.jpg"
        thumb = cv2.resize(image_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumbnail_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

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
        return image_name, depth_name

    def destroy_node(self) -> bool:
        try:
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
