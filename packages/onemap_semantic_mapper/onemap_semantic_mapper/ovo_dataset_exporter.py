from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
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
    pose_rotation: np.ndarray
    pose_translation: np.ndarray


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r_delta = r_a.T @ r_b
    cos_theta = (np.trace(r_delta) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


class OVOReplicaExporter(Node):
    def __init__(self) -> None:
        super().__init__("ovo_dataset_exporter")
        self.bridge = CvBridge()

        self._declare_parameters()
        self._load_parameters()

        self._image_buffer: Deque[BufferedImage] = deque(maxlen=200)
        self._depth_buffer: Deque[BufferedDepth] = deque(maxlen=200)
        self._odom_buffer: Deque[BufferedOdom] = deque(maxlen=400)

        self._latest_sensor_stamp: Optional[float] = None
        self._odom_time_offset: Optional[float] = None
        self._camera_info_received = False
        self._config_written = False
        self._synced_frame_count = 0
        self._saved_frame_count = 0
        self._frame_index = 0
        self._stopped_for_max_frames = False
        self._last_wait_log_sec = 0.0
        self._last_saved_c2w: Optional[np.ndarray] = None
        self._last_saved_synced_index = -1

        self.scene_dir = self.output_root / self.scene_name
        self.results_dir = self.scene_dir / "results"
        self.scene_config_path = self.config_root / f"{self.scene_name}.yaml"
        self.metadata_path = self.scene_dir / "export_info.yaml"
        self.traj_path = self.scene_dir / "traj.txt"
        self.frame_index_path = self.scene_dir / "frames_index.csv"

        self._prepare_output_paths()
        self._traj_file = self.traj_path.open("a", encoding="utf-8")
        self._frame_index_file = self.frame_index_path.open("a", encoding="utf-8", newline="")
        self._frame_index_writer = csv.writer(self._frame_index_file)
        self._load_existing_frame_index()
        self._ensure_frame_index_header()
        self._write_metadata()

        self.image_sub = self.create_subscription(Image, self.image_topic, self._image_callback, 50)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self._depth_callback, 50)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, 20)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 100)

        self.get_logger().info(
            "OVO Replica exporter ready. "
            f"scene={self.scene_name}, image={self.image_topic}, depth={self.depth_topic}, odom={self.odom_topic}, "
            f"output={self.scene_dir}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "common.img_topic": "/robot_rgb",
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
            "camera.cam_width": 1280,
            "camera.cam_height": 720,
            "camera.cam_fx": 640.0,
            "camera.cam_fy": 493.355477,
            "camera.cam_cx": 640.0,
            "camera.cam_cy": 360.0,
            "export.depth_topic": "/depth",
            "export.camera_info_topic": "/camera_info",
            "export.odom_topic": "/aft_mapped_to_init",
            "export.output_root": "/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica",
            "export.config_root": "/home/peng/isacc_slam/reference/OVO/data/working/configs/Replica",
            "export.scene_name": "isaac_turtlebot3",
            "export.sync_slop_sec": 0.08,
            "export.depth_input_scale": 1000.0,
            "export.output_depth_scale": 1000.0,
            "export.jpeg_quality": 95,
            "export.frame_stride": 1,
            "export.max_frames": -1,
            "export.overwrite_scene": True,
            "export.use_keyframe_filter": True,
            "export.keyframe_min_translation_m": 0.25,
            "export.keyframe_min_rotation_deg": 12.0,
            "export.keyframe_min_frame_gap": 5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.image_topic = str(self.get_parameter("common.img_topic").value)
        self.depth_topic = str(self.get_parameter("export.depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("export.camera_info_topic").value)
        self.odom_topic = str(self.get_parameter("export.odom_topic").value)

        self.output_root = Path(str(self.get_parameter("export.output_root").value)).expanduser()
        self.config_root = Path(str(self.get_parameter("export.config_root").value)).expanduser()
        self.scene_name = str(self.get_parameter("export.scene_name").value)

        self.sync_slop_sec = float(self.get_parameter("export.sync_slop_sec").value)
        self.depth_input_scale = float(self.get_parameter("export.depth_input_scale").value)
        self.output_depth_scale = float(self.get_parameter("export.output_depth_scale").value)
        self.jpeg_quality = int(self.get_parameter("export.jpeg_quality").value)
        self.frame_stride = max(1, int(self.get_parameter("export.frame_stride").value))
        self.max_frames = int(self.get_parameter("export.max_frames").value)
        self.overwrite_scene = bool(self.get_parameter("export.overwrite_scene").value)
        self.use_keyframe_filter = bool(self.get_parameter("export.use_keyframe_filter").value)
        self.keyframe_min_translation_m = float(self.get_parameter("export.keyframe_min_translation_m").value)
        self.keyframe_min_rotation_deg = float(self.get_parameter("export.keyframe_min_rotation_deg").value)
        self.keyframe_min_frame_gap = max(1, int(self.get_parameter("export.keyframe_min_frame_gap").value))

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

    def _load_existing_frame_index(self) -> None:
        existing_frames = sorted(self.results_dir.glob("frame*.jpg"))
        if existing_frames:
            last_name = existing_frames[-1].stem
            self._frame_index = int(last_name.replace("frame", "")) + 1
            self._saved_frame_count = len(existing_frames)
            self.get_logger().info(f"Appending to existing OVO scene at frame {self._frame_index}.")

    def _write_metadata(self) -> None:
        metadata = {
            "scene_name": self.scene_name,
            "image_topic": self.image_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "odom_topic": self.odom_topic,
            "output_depth_scale": self.output_depth_scale,
            "frame_index_path": str(self.frame_index_path),
            "keyframe_filter": {
                "enabled": self.use_keyframe_filter,
                "min_translation_m": self.keyframe_min_translation_m,
                "min_rotation_deg": self.keyframe_min_rotation_deg,
                "min_frame_gap": self.keyframe_min_frame_gap,
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

    def _ensure_frame_index_header(self) -> None:
        if self.frame_index_path.stat().st_size > 0:
            return
        self._frame_index_writer.writerow(
            [
                "saved_frame_index",
                "image_stamp_sec",
                "depth_stamp_sec",
                "odom_stamp_sec",
                "synced_frame_index",
                "image_name",
                "depth_name",
            ]
        )
        self._frame_index_file.flush()

    def _image_callback(self, msg: Image) -> None:
        if self._stopped_for_max_frames:
            return

        stamp = stamp_to_seconds(msg.header.stamp)
        self._latest_sensor_stamp = stamp
        image_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self._image_buffer.append(BufferedImage(stamp=stamp, image_rgb=image_rgb))
        self._try_export_frame(stamp)

    def _depth_callback(self, msg: Image) -> None:
        stamp = stamp_to_seconds(msg.header.stamp)
        self._latest_sensor_stamp = stamp
        depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth_raw)
        if np.issubdtype(depth.dtype, np.integer):
            depth_m = depth.astype(np.float32) / max(self.depth_input_scale, 1e-6)
        else:
            depth_m = depth.astype(np.float32, copy=False)
        self._depth_buffer.append(BufferedDepth(stamp=stamp, depth_m=depth_m))

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        k = list(msg.k)
        if len(k) != 9 or k[0] <= 0.0 or k[4] <= 0.0:
            return

        self.fx = float(k[0])
        self.fy = float(k[4])
        self.cx = float(k[2])
        self.cy = float(k[5])
        self.cam_width = int(msg.width)
        self.cam_height = int(msg.height)
        self._camera_info_received = True
        self._write_scene_config()

    def _odom_callback(self, msg: Odometry) -> None:
        raw_stamp = stamp_to_seconds(msg.header.stamp)
        odom_stamp = self._normalize_odom_stamp(raw_stamp)
        quat = msg.pose.pose.orientation
        pose_rotation = quaternion_to_rotation_matrix(quat.x, quat.y, quat.z, quat.w)
        pose_translation = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=np.float32,
        )
        self._odom_buffer.append(
            BufferedOdom(
                stamp=odom_stamp,
                pose_rotation=pose_rotation,
                pose_translation=pose_translation,
            )
        )

    def _normalize_odom_stamp(self, raw_stamp: float) -> float:
        if self._latest_sensor_stamp is None:
            return raw_stamp

        if abs(raw_stamp - self._latest_sensor_stamp) <= self.sync_slop_sec:
            return raw_stamp

        if self._odom_time_offset is None:
            self._odom_time_offset = raw_stamp - self._latest_sensor_stamp
            self.get_logger().warn(
                "OVO exporter detected odom on a different time axis. "
                f"Applying offset {self._odom_time_offset:.6f}s."
            )

        return raw_stamp - self._odom_time_offset

    def _find_closest(self, buffer: Deque, stamp: float):
        if not buffer:
            return None
        best = min(buffer, key=lambda item: abs(item.stamp - stamp))
        if abs(best.stamp - stamp) > self.sync_slop_sec:
            return None
        return best

    def _try_export_frame(self, image_stamp: float) -> None:
        if not self._camera_info_received:
            return

        if self.max_frames > 0 and self._saved_frame_count >= self.max_frames:
            if not self._stopped_for_max_frames:
                self.get_logger().info(f"OVO exporter reached max_frames={self.max_frames}.")
                self._stopped_for_max_frames = True
            return

        image = self._find_closest(self._image_buffer, image_stamp)
        depth = self._find_closest(self._depth_buffer, image_stamp)
        odom = self._find_closest(self._odom_buffer, image_stamp)
        if image is None or depth is None or odom is None:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self._last_wait_log_sec > 2.0:
                self.get_logger().warn("OVO exporter is waiting for synchronized image/depth/odom/camera_info.")
                self._last_wait_log_sec = now_sec
            return

        self._synced_frame_count += 1
        if (self._synced_frame_count - 1) % self.frame_stride != 0:
            return

        c2w = self._camera_to_world_matrix(odom.pose_rotation, odom.pose_translation)
        if self._should_skip_keyframe(c2w):
            return
        self._save_frame(
            image.image_rgb,
            depth.depth_m,
            c2w,
            image.stamp,
            depth.stamp,
            odom.stamp,
        )

    def _should_skip_keyframe(self, c2w: np.ndarray) -> bool:
        if not self.use_keyframe_filter:
            return False
        if self._last_saved_c2w is None:
            return False

        synced_gap = self._synced_frame_count - self._last_saved_synced_index
        translation = float(np.linalg.norm(c2w[:3, 3] - self._last_saved_c2w[:3, 3]))
        rotation_deg = rotation_angle_deg(self._last_saved_c2w[:3, :3], c2w[:3, :3])

        if synced_gap < self.keyframe_min_frame_gap:
            if translation < self.keyframe_min_translation_m and rotation_deg < self.keyframe_min_rotation_deg:
                return True

        if translation < self.keyframe_min_translation_m and rotation_deg < self.keyframe_min_rotation_deg:
            return True
        return False

    def _camera_to_world_matrix(self, pose_rotation: np.ndarray, pose_translation: np.ndarray) -> np.ndarray:
        sample_camera = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        sample_world = self._transform_camera_points_to_world(sample_camera, pose_rotation, pose_translation)
        origin = sample_world[0]
        rotation = np.column_stack(
            (
                sample_world[1] - origin,
                sample_world[2] - origin,
                sample_world[3] - origin,
            )
        )
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

    def _save_frame(
        self,
        image_rgb: np.ndarray,
        depth_m: np.ndarray,
        c2w: np.ndarray,
        image_stamp: float,
        depth_stamp: float,
        odom_stamp: float,
    ) -> None:
        if image_rgb.shape[:2] != depth_m.shape[:2]:
            depth_m = cv2.resize(depth_m, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        image_path = self.results_dir / f"frame{self._frame_index:06d}.jpg"
        depth_path = self.results_dir / f"depth{self._frame_index:06d}.png"

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(image_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
            raise RuntimeError(f"Failed to write image frame: {image_path}")

        depth_clean = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_uint16 = np.clip(np.rint(depth_clean * self.output_depth_scale), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        if not cv2.imwrite(str(depth_path), depth_uint16):
            raise RuntimeError(f"Failed to write depth frame: {depth_path}")

        flattened = " ".join(f"{value:.9f}" for value in c2w.reshape(-1))
        self._traj_file.write(flattened + "\n")
        self._traj_file.flush()
        self._frame_index_writer.writerow(
            [
                self._frame_index,
                f"{image_stamp:.9f}",
                f"{depth_stamp:.9f}",
                f"{odom_stamp:.9f}",
                self._synced_frame_count,
                image_path.name,
                depth_path.name,
            ]
        )
        self._frame_index_file.flush()

        self._frame_index += 1
        self._saved_frame_count += 1
        self._last_saved_c2w = c2w.copy()
        self._last_saved_synced_index = self._synced_frame_count
        if self._saved_frame_count == 1 or self._saved_frame_count % 20 == 0:
            self.get_logger().info(
                f"OVO exporter saved {self._saved_frame_count} frames to {self.scene_dir}."
            )

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
        if not self._config_written:
            self.get_logger().info(f"Wrote OVO scene config to {self.scene_config_path}.")
        self._config_written = True

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, "_traj_file") and not self._traj_file.closed:
                self._traj_file.close()
            if hasattr(self, "_frame_index_file") and not self._frame_index_file.closed:
                self._frame_index_file.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OVOReplicaExporter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
