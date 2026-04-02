from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def _copy_header(src: Header, *, frame_id: str) -> Header:
    header = Header()
    header.stamp = src.stamp
    header.frame_id = frame_id
    return header


def _copy_cloud(src: PointCloud2, *, frame_id: str) -> PointCloud2:
    msg = PointCloud2()
    msg.header = _copy_header(src.header, frame_id=frame_id)
    msg.height = src.height
    msg.width = src.width
    msg.fields = list(src.fields)
    msg.is_bigendian = src.is_bigendian
    msg.point_step = src.point_step
    msg.row_step = src.row_step
    msg.data = bytes(src.data)
    msg.is_dense = src.is_dense
    return msg


def _copy_odom(src: Odometry, *, map_frame: str, base_frame: str) -> Odometry:
    msg = Odometry()
    msg.header = _copy_header(src.header, frame_id=map_frame)
    msg.child_frame_id = base_frame
    msg.pose = src.pose
    msg.twist = src.twist
    return msg


_XYZI_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]


def _read_xyz_array(msg: PointCloud2) -> np.ndarray:
    points = [
        (float(p[0]), float(p[1]), float(p[2]))
        for p in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
    ]
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def _make_xyzi_cloud(*, header: Header, xyzi: np.ndarray) -> PointCloud2:
    if xyzi.size == 0:
        return point_cloud2.create_cloud(header, _XYZI_FIELDS, [])
    rows = [tuple(map(float, row)) for row in xyzi]
    return point_cloud2.create_cloud(header, _XYZI_FIELDS, rows)


class FastLivoFarBridge(Node):
    def __init__(self) -> None:
        super().__init__("fast_livo_far_bridge")
        self._declare_parameters()
        self._load_parameters()
        self._terrain_ext_voxels: Dict[Tuple[int, int, int], np.ndarray] = {}

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        far_cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.odom_pub = self.create_publisher(Odometry, self.odom_out_topic, odom_qos)
        self.registered_scan_pub = self.create_publisher(
            PointCloud2, self.registered_scan_out_topic, far_cloud_qos
        )
        self.terrain_map_pub = self.create_publisher(
            PointCloud2, self.terrain_map_out_topic, far_cloud_qos
        )
        self.terrain_map_ext_pub = self.create_publisher(
            PointCloud2, self.terrain_map_ext_out_topic, far_cloud_qos
        )

        self.create_subscription(
            Odometry, self.odom_in_topic, self._odom_callback, odom_qos
        )
        self.create_subscription(
            PointCloud2, self.cloud_in_topic, self._cloud_callback, qos_profile_sensor_data
        )

        self.get_logger().info(
            "FAST-LIVO2 -> FAR bridge ready. "
            f"odom: {self.odom_in_topic} -> {self.odom_out_topic}, "
            f"cloud: {self.cloud_in_topic} -> {self.registered_scan_out_topic}, "
            f"map_frame={self.map_frame}, base_frame={self.base_frame}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "odom_in_topic": "/aft_mapped_to_init",
            "cloud_in_topic": "/cloud_registered",
            "odom_out_topic": "/state_estimation",
            "registered_scan_out_topic": "/registered_scan",
            "terrain_map_out_topic": "/terrain_map",
            "terrain_map_ext_out_topic": "/terrain_map_ext",
            "map_frame": "map",
            "base_frame": "base_link",
            "publish_terrain_map": True,
            "publish_terrain_map_ext": True,
            "terrain_ground_percentile": 2.0,
            "terrain_obstacle_height_m": 0.18,
            "terrain_local_voxel_size_m": 0.10,
            "terrain_ext_voxel_size_m": 0.18,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.odom_in_topic = str(self.get_parameter("odom_in_topic").value)
        self.cloud_in_topic = str(self.get_parameter("cloud_in_topic").value)
        self.odom_out_topic = str(self.get_parameter("odom_out_topic").value)
        self.registered_scan_out_topic = str(
            self.get_parameter("registered_scan_out_topic").value
        )
        self.terrain_map_out_topic = str(self.get_parameter("terrain_map_out_topic").value)
        self.terrain_map_ext_out_topic = str(
            self.get_parameter("terrain_map_ext_out_topic").value
        )
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.publish_terrain_map = bool(self.get_parameter("publish_terrain_map").value)
        self.publish_terrain_map_ext = bool(
            self.get_parameter("publish_terrain_map_ext").value
        )
        self.terrain_ground_percentile = float(
            self.get_parameter("terrain_ground_percentile").value
        )
        self.terrain_obstacle_height_m = float(
            self.get_parameter("terrain_obstacle_height_m").value
        )
        self.terrain_local_voxel_size_m = float(
            self.get_parameter("terrain_local_voxel_size_m").value
        )
        self.terrain_ext_voxel_size_m = float(
            self.get_parameter("terrain_ext_voxel_size_m").value
        )

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom_pub.publish(
            _copy_odom(msg, map_frame=self.map_frame, base_frame=self.base_frame)
        )

    def _cloud_callback(self, msg: PointCloud2) -> None:
        self.registered_scan_pub.publish(_copy_cloud(msg, frame_id=self.map_frame))
        xyz = _read_xyz_array(msg)
        terrain_xyzi = self._build_terrain_xyzi(xyz)
        header = _copy_header(msg.header, frame_id=self.map_frame)

        if self.publish_terrain_map:
            local_xyzi = self._voxel_downsample(terrain_xyzi, self.terrain_local_voxel_size_m)
            self.terrain_map_pub.publish(_make_xyzi_cloud(header=header, xyzi=local_xyzi))

        if self.publish_terrain_map_ext:
            self._accumulate_ext_map(terrain_xyzi)
            ext_xyzi = self._ext_voxels_to_array()
            self.terrain_map_ext_pub.publish(_make_xyzi_cloud(header=header, xyzi=ext_xyzi))

    def _build_terrain_xyzi(self, xyz: np.ndarray) -> np.ndarray:
        if xyz.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        z_floor = float(np.percentile(xyz[:, 2], self.terrain_ground_percentile))
        intensity = np.where(
            xyz[:, 2] > z_floor + self.terrain_obstacle_height_m, 1.0, 0.0
        ).astype(np.float32)
        return np.column_stack((xyz.astype(np.float32), intensity))

    def _voxel_downsample(self, xyzi: np.ndarray, voxel_size: float) -> np.ndarray:
        if xyzi.size == 0 or voxel_size <= 1e-6:
            return xyzi
        voxels: Dict[Tuple[int, int, int], np.ndarray] = {}
        for row in xyzi:
            key = tuple(np.floor(row[:3] / voxel_size).astype(np.int64).tolist())
            prev = voxels.get(key)
            if prev is None or row[3] > prev[3]:
                voxels[key] = row.copy()
        if not voxels:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack(list(voxels.values()), axis=0).astype(np.float32)

    def _accumulate_ext_map(self, xyzi: np.ndarray) -> None:
        if xyzi.size == 0:
            return
        for row in self._voxel_downsample(xyzi, self.terrain_ext_voxel_size_m):
            key = tuple(
                np.floor(row[:3] / max(self.terrain_ext_voxel_size_m, 1e-6))
                .astype(np.int64)
                .tolist()
            )
            prev = self._terrain_ext_voxels.get(key)
            if prev is None:
                self._terrain_ext_voxels[key] = row.copy()
                continue
            prev[:3] = 0.5 * (prev[:3] + row[:3])
            prev[3] = max(prev[3], row[3])

    def _ext_voxels_to_array(self) -> np.ndarray:
        if not self._terrain_ext_voxels:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack(list(self._terrain_ext_voxels.values()), axis=0).astype(np.float32)


def main() -> None:
    rclpy.init()
    node = FastLivoFarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
