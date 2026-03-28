from __future__ import annotations

from contextlib import closing
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class OVOSemanticLidarMapPublisher(Node):
    def __init__(self) -> None:
        super().__init__("ovo_semantic_lidar_map_publisher")
        self._declare_parameters()
        self._load_parameters()

        self.cloud_pub = self.create_publisher(PointCloud2, self.topic_name, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.cloud_sub = self.create_subscription(PointCloud2, self.input_cloud_topic, self._cloud_callback, 1)
        self.timer = self.create_timer(self.publish_period_sec, self._timer_callback)

        self._last_mtime_ns: int | None = None
        self._snapshot_tree: cKDTree | None = None
        self._snapshot_xyz: np.ndarray | None = None
        self._snapshot_rgb: np.ndarray | None = None
        self._snapshot_instance_id: np.ndarray | None = None
        self._snapshot_class_id: np.ndarray | None = None
        self._cached_markers: MarkerArray | None = None
        self._latest_cloud_msg: PointCloud2 | None = None
        self._latest_projected_cloud: PointCloud2 | None = None
        self._latest_projected_from_stamp: tuple[int, int] | None = None
        self._last_cloud_stamp: tuple[int, int] | None = None

        self.get_logger().info(
            "OVO semantic lidar map publisher ready. "
            f"artifact={self.artifact_path}, input={self.input_cloud_topic}, output={self.topic_name}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "artifact_path": "/home/peng/isacc_slam/reference/OVO/data/output/Replica/isaac_livo2_online_vanilla/isaac_turtlebot3_livo2_online/semantic_snapshot.npz",
            "input_cloud_topic": "/cloud_registered",
            "topic_name": "/ovo_semantic_lidar_map",
            "marker_topic": "/ovo_semantic_lidar_labels",
            "frame_id": "camera_init",
            "publish_period_sec": 2.0,
            "max_points": 250000,
            "knn_k": 5,
            "match_radius_m": 0.25,
            "default_gray": 160,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.artifact_path = Path(str(self.get_parameter("artifact_path").value)).expanduser()
        self.input_cloud_topic = str(self.get_parameter("input_cloud_topic").value)
        self.topic_name = str(self.get_parameter("topic_name").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_period_sec = float(self.get_parameter("publish_period_sec").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.knn_k = int(self.get_parameter("knn_k").value)
        self.match_radius_m = float(self.get_parameter("match_radius_m").value)
        self.default_gray = int(self.get_parameter("default_gray").value)

    def _cloud_callback(self, msg: PointCloud2) -> None:
        self._latest_cloud_msg = msg
        self._last_cloud_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def _timer_callback(self) -> None:
        if self.artifact_path.exists():
            mtime_ns = self.artifact_path.stat().st_mtime_ns
            if self._last_mtime_ns != mtime_ns:
                if self._load_artifact():
                    self._last_mtime_ns = mtime_ns
                    self._latest_projected_cloud = None
                    self._latest_projected_from_stamp = None

        if self._latest_cloud_msg is not None and self._snapshot_tree is not None:
            cloud_stamp = (self._latest_cloud_msg.header.stamp.sec, self._latest_cloud_msg.header.stamp.nanosec)
            if self._latest_projected_cloud is None or self._latest_projected_from_stamp != cloud_stamp:
                self._latest_projected_cloud = self._project_lidar_cloud(self._latest_cloud_msg)
                self._latest_projected_from_stamp = cloud_stamp

        if self._latest_projected_cloud is not None:
            self._latest_projected_cloud.header.stamp = self.get_clock().now().to_msg()
            self.cloud_pub.publish(self._latest_projected_cloud)
        if self._cached_markers is not None:
            stamp = self.get_clock().now().to_msg()
            for marker in self._cached_markers.markers:
                marker.header.stamp = stamp
            self.marker_pub.publish(self._cached_markers)

    def _load_artifact(self) -> bool:
        try:
            with closing(np.load(self.artifact_path, allow_pickle=True)) as data:
                xyz = data["xyz"].astype(np.float32, copy=False)
                rgb = data["rgb"].astype(np.uint8, copy=False)
                instance_id = data["instance_id"].astype(np.int32, copy=False)
                class_id = data["class_id"].astype(np.int32, copy=False)
                instance_centers = data.get("instance_centers", np.zeros((0, 3), dtype=np.float32)).astype(
                    np.float32, copy=False
                )
                instance_labels = data.get("instance_labels", np.asarray([], dtype=object))
                instance_ids = data.get("instance_ids", np.asarray([], dtype=np.int32)).astype(np.int32, copy=False)
        except (EOFError, OSError, ValueError, KeyError) as exc:
            self.get_logger().warning(f"Skipped semantic snapshot reload for {self.artifact_path}: {exc}")
            return False

        self._snapshot_xyz = xyz
        self._snapshot_rgb = rgb
        self._snapshot_instance_id = instance_id
        self._snapshot_class_id = class_id
        self._snapshot_tree = cKDTree(xyz)

        markers = MarkerArray()
        for idx, (center, label, ins_id) in enumerate(zip(instance_centers, instance_labels, instance_ids)):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.ns = "ovo_lidar_instances"
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
            f"Loaded semantic snapshot for lidar projection: points={xyz.shape[0]}, labels={len(markers.markers)}"
        )
        return True

    def _project_lidar_cloud(self, msg: PointCloud2) -> PointCloud2:
        assert self._snapshot_tree is not None
        assert self._snapshot_xyz is not None
        assert self._snapshot_rgb is not None
        assert self._snapshot_instance_id is not None
        assert self._snapshot_class_id is not None

        pts = np.asarray(
            list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )
        if pts.size == 0:
            return point_cloud2.create_cloud(Header(frame_id=self.frame_id), [], [])

        original_count = pts.shape[0]
        if original_count > self.max_points:
            step = int(np.ceil(original_count / self.max_points))
            pts = pts[::step]

        distances, indices = self._snapshot_tree.query(
            pts,
            k=min(self.knn_k, self._snapshot_xyz.shape[0]),
            distance_upper_bound=self.match_radius_m,
            workers=-1,
        )

        if np.isscalar(distances):
            distances = np.asarray([[distances]], dtype=np.float32)
            indices = np.asarray([[indices]], dtype=np.int64)
        elif distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        default_rgb = np.full((pts.shape[0], 3), self.default_gray, dtype=np.uint8)
        default_instance = np.full((pts.shape[0],), -1, dtype=np.int32)
        default_class = np.full((pts.shape[0],), -1, dtype=np.int32)

        rgb = default_rgb.copy()
        instance_id = default_instance.copy()
        class_id = default_class.copy()

        valid = np.isfinite(distances) & (indices < self._snapshot_xyz.shape[0])
        matched_rows = np.nonzero(valid.any(axis=1))[0]

        for row in matched_rows:
            nbr_ids = indices[row][valid[row]]
            if nbr_ids.size == 0:
                continue
            nbr_instance = self._snapshot_instance_id[nbr_ids]
            nbr_class = self._snapshot_class_id[nbr_ids]
            nbr_rgb = self._snapshot_rgb[nbr_ids]

            unique_instances, counts = np.unique(nbr_instance, return_counts=True)
            chosen_instance = int(unique_instances[np.argmax(counts)])
            chosen_mask = nbr_instance == chosen_instance
            chosen_classes = nbr_class[chosen_mask]
            chosen_rgbs = nbr_rgb[chosen_mask]

            if chosen_classes.size == 0:
                chosen_classes = nbr_class
                chosen_rgbs = nbr_rgb
            unique_classes, class_counts = np.unique(chosen_classes, return_counts=True)
            chosen_class = int(unique_classes[np.argmax(class_counts)])
            chosen_rgb = np.mean(chosen_rgbs.astype(np.float32), axis=0).astype(np.uint8)

            instance_id[row] = chosen_instance
            class_id[row] = chosen_class
            rgb[row] = chosen_rgb

        rgb_u32 = (
            (rgb[:, 0].astype(np.uint32) << 16)
            | (rgb[:, 1].astype(np.uint32) << 8)
            | rgb[:, 2].astype(np.uint32)
        )
        points = np.zeros(
            pts.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
                ("instance_id", np.int32),
                ("class_id", np.int32),
            ],
        )
        points["x"] = pts[:, 0]
        points["y"] = pts[:, 1]
        points["z"] = pts[:, 2]
        points["rgb"] = rgb_u32
        points["instance_id"] = instance_id
        points["class_id"] = class_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="instance_id", offset=16, datatype=PointField.INT32, count=1),
            PointField(name="class_id", offset=20, datatype=PointField.INT32, count=1),
        ]
        header = Header(frame_id=self.frame_id)
        matched_ratio = float((instance_id >= 0).sum() / max(instance_id.shape[0], 1))
        self.get_logger().info(
            f"Projected semantic labels onto lidar map: matched={matched_ratio:.3f}, "
            f"lidar_points={pts.shape[0]}, source_points={self._snapshot_xyz.shape[0]}"
        )
        return point_cloud2.create_cloud(header, fields, points)


def main() -> None:
    rclpy.init()
    node = OVOSemanticLidarMapPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
