from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def _prepare_numpy_pickle_compat() -> None:
    """Allow old snapshots pickled with newer NumPy module paths to load on ROS Python."""
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def pack_rgb_float32(rgb: np.ndarray) -> np.ndarray:
    rgb_u32 = (
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32)
    )
    return rgb_u32.view(np.float32)


class OVOSemanticMapPublisher(Node):
    def __init__(self) -> None:
        super().__init__("ovo_semantic_map_publisher")
        self._declare_parameters()
        self._load_parameters()

        self.cloud_pub = self.create_publisher(PointCloud2, self.topic_name, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.timer = self.create_timer(self.publish_period_sec, self._timer_callback)

        self._startup_wall_time_ns = time.time_ns()
        self._last_mtime_ns: int | None = None
        self._cached_cloud: PointCloud2 | None = None
        self._cached_markers: MarkerArray | None = None
        self._clear_needed = True

        self.get_logger().info(
            f"OVO semantic map publisher ready. artifact={self.artifact_path}, topic={self.topic_name}, frame_id={self.frame_id}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "artifact_path": "/home/peng/isacc_slam/reference/OVO/data/output/Replica/isaac_livo2_online_vanilla/isaac_turtlebot3_livo2/semantic_snapshot.npz",
            "topic_name": "/ovo_semantic_map",
            "marker_topic": "/ovo_instance_labels",
            "frame_id": "camera_init",
            "publish_period_sec": 2.0,
            "max_points": 3000000,
            "semantic_only": True,
            "min_instance_points_for_label": 200,
            "min_instance_views_for_label": 3,
            "load_existing_artifact_on_start": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.artifact_path = Path(str(self.get_parameter("artifact_path").value)).expanduser()
        self.topic_name = str(self.get_parameter("topic_name").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_period_sec = float(self.get_parameter("publish_period_sec").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.semantic_only = bool(self.get_parameter("semantic_only").value)
        self.min_instance_points_for_label = int(self.get_parameter("min_instance_points_for_label").value)
        self.min_instance_views_for_label = int(self.get_parameter("min_instance_views_for_label").value)
        self.load_existing_artifact_on_start = bool(self.get_parameter("load_existing_artifact_on_start").value)

    def _fields(self) -> list[PointField]:
        return [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="instance_id", offset=16, datatype=PointField.INT32, count=1),
            PointField(name="class_id", offset=20, datatype=PointField.INT32, count=1),
        ]

    def _empty_cloud(self) -> PointCloud2:
        return point_cloud2.create_cloud(Header(frame_id=self.frame_id), self._fields(), [])

    def _clear_markers(self) -> MarkerArray:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = "ovo_instances"
        marker.id = 0
        marker.action = Marker.DELETEALL
        return MarkerArray(markers=[marker])

    def _timer_callback(self) -> None:
        if self.artifact_path.exists():
            mtime_ns = self.artifact_path.stat().st_mtime_ns
            if self._last_mtime_ns != mtime_ns:
                should_load = self.load_existing_artifact_on_start or mtime_ns >= self._startup_wall_time_ns
                if should_load and self._load_artifact():
                    self._last_mtime_ns = mtime_ns
                    self._clear_needed = False
                else:
                    self._last_mtime_ns = mtime_ns
        elif self._clear_needed or self._cached_cloud is not None or self._cached_markers is not None:
            self._cached_cloud = self._empty_cloud()
            self._cached_markers = self._clear_markers()
            self._clear_needed = False

        if self._cached_cloud is not None:
            self._cached_cloud.header.stamp = self.get_clock().now().to_msg()
            self.cloud_pub.publish(self._cached_cloud)
        if self._cached_markers is not None:
            stamp = self.get_clock().now().to_msg()
            for marker in self._cached_markers.markers:
                marker.header.stamp = stamp
            self.marker_pub.publish(self._cached_markers)

    def _load_artifact(self) -> bool:
        try:
            _prepare_numpy_pickle_compat()
            with closing(np.load(self.artifact_path, allow_pickle=True)) as data:
                xyz = data["xyz"].astype(np.float32, copy=False)
                rgb = data["rgb"].astype(np.uint8, copy=False)
                instance_id = data["instance_id"].astype(np.int32, copy=False)
                class_id = data["class_id"].astype(np.int32, copy=False)
                instance_centers = data.get("instance_centers", np.zeros((0, 3), dtype=np.float32)).astype(np.float32, copy=False)
                instance_labels = np.asarray(data.get("instance_labels", np.asarray([], dtype=np.str_)), dtype=np.str_)
                instance_ids = data.get("instance_ids", np.asarray([], dtype=np.int32)).astype(np.int32, copy=False)
                instance_point_counts = data.get("instance_point_counts", np.zeros((0,), dtype=np.int32)).astype(np.int32, copy=False)
                instance_view_counts = data.get("instance_view_counts", np.zeros((0,), dtype=np.int32)).astype(np.int32, copy=False)
        except (EOFError, OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError, ImportError) as exc:
            self.get_logger().warning(
                f"Skipped semantic snapshot reload for {self.artifact_path}: {exc}"
            )
            return False

        if xyz.shape[0] > self.max_points:
            step = int(np.ceil(xyz.shape[0] / self.max_points))
            keep = np.arange(0, xyz.shape[0], step, dtype=np.int64)
            xyz = xyz[keep]
            rgb = rgb[keep]
            instance_id = instance_id[keep]
            class_id = class_id[keep]

        if self.semantic_only:
            keep = class_id >= 0
            xyz = xyz[keep]
            rgb = rgb[keep]
            instance_id = instance_id[keep]
            class_id = class_id[keep]

        rgb_f32 = pack_rgb_float32(rgb)

        points = np.zeros(
            xyz.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.float32),
                ("instance_id", np.int32),
                ("class_id", np.int32),
            ],
        )
        points["x"] = xyz[:, 0]
        points["y"] = xyz[:, 1]
        points["z"] = xyz[:, 2]
        points["rgb"] = rgb_f32
        points["instance_id"] = instance_id
        points["class_id"] = class_id

        header = Header(frame_id=self.frame_id)
        self._cached_cloud = point_cloud2.create_cloud(header, self._fields(), points)

        markers = MarkerArray()
        for idx, (center, label, ins_id, point_count, view_count) in enumerate(
            zip(instance_centers, instance_labels, instance_ids, instance_point_counts, instance_view_counts)
        ):
            if int(point_count) < self.min_instance_points_for_label:
                continue
            if int(view_count) < self.min_instance_views_for_label:
                continue
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.ns = "ovo_instances"
            marker.id = int(idx)
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2] + 0.2)
            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.18
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.9
            marker.lifetime = Duration(sec=0, nanosec=0)
            marker.text = f"{label}#{int(ins_id)}"
            markers.markers.append(marker)
        self._cached_markers = markers

        self.get_logger().info(
            f"Loaded semantic snapshot: points={xyz.shape[0]}, labels={len(markers.markers)} from {self.artifact_path}"
        )
        return True


def main() -> None:
    rclpy.init()
    node = OVOSemanticMapPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
