from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, PointStamped, PolygonStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool


class FarAutoGoalManager(Node):
    def __init__(self) -> None:
        super().__init__("far_auto_goal_manager")
        self._declare_parameters()
        self._load_parameters()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        finish_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.goal_pub = self.create_publisher(PointStamped, self.goal_topic, qos)
        self.finish_pub = self.create_publisher(Bool, self.finish_topic, finish_qos)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)
        self.create_subscription(PointCloud2, self.terrain_map_ext_topic, self._terrain_cb, qos)
        self.create_subscription(
            PolygonStamped, self.boundary_topic, self._boundary_cb, qos
        )
        self.create_subscription(
            Bool, self.reach_goal_topic, self._reach_goal_cb, qos
        )

        self.timer = self.create_timer(1.0 / self.update_rate_hz, self._timer_cb)

        self._robot_xy: Optional[np.ndarray] = None
        self._robot_yaw: Optional[float] = None
        self._terrain_xy: Optional[np.ndarray] = None
        self._boundary_xy: Optional[np.ndarray] = None
        self._goal_xy: Optional[np.ndarray] = None
        self._last_publish_sec: float = 0.0
        self._goal_reached: bool = True
        self._finished: bool = False
        self._finish_published: bool = False
        self._sweep_direction: int = 1

        self.get_logger().info(
            "FAR auto-goal manager ready. "
            f"goal_topic={self.goal_topic}, odom_topic={self.odom_topic}, "
            f"terrain_map_ext_topic={self.terrain_map_ext_topic}, "
            f"finish_topic={self.finish_topic}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "goal_topic": "/goal_point",
            "odom_topic": "/state_estimation",
            "terrain_map_ext_topic": "/terrain_map_ext",
            "boundary_topic": "/navigation_boundary",
            "reach_goal_topic": "/far_reach_goal_status",
            "finish_topic": "/far_auto_exploration_finish",
            "world_frame": "map",
            "update_rate_hz": 2.0,
            "goal_republish_sec": 2.0,
            "lane_spacing_m": 2.5,
            "lane_forward_step_m": 4.0,
            "edge_inset_m": 1.5,
            "finish_margin_m": 1.0,
            "min_progress_extent_m": 3.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.terrain_map_ext_topic = str(
            self.get_parameter("terrain_map_ext_topic").value
        )
        self.boundary_topic = str(self.get_parameter("boundary_topic").value)
        self.reach_goal_topic = str(self.get_parameter("reach_goal_topic").value)
        self.finish_topic = str(self.get_parameter("finish_topic").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.update_rate_hz = float(self.get_parameter("update_rate_hz").value)
        self.goal_republish_sec = float(self.get_parameter("goal_republish_sec").value)
        self.lane_spacing_m = float(self.get_parameter("lane_spacing_m").value)
        self.lane_forward_step_m = float(
            self.get_parameter("lane_forward_step_m").value
        )
        self.edge_inset_m = float(self.get_parameter("edge_inset_m").value)
        self.finish_margin_m = float(self.get_parameter("finish_margin_m").value)
        self.min_progress_extent_m = float(
            self.get_parameter("min_progress_extent_m").value
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_xy = np.asarray(
            [float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)],
            dtype=np.float32,
        )
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _terrain_cb(self, msg: PointCloud2) -> None:
        pts = [
            (float(p[0]), float(p[1]))
            for p in point_cloud2.read_points(
                msg, field_names=("x", "y"), skip_nans=True
            )
        ]
        if pts:
            self._terrain_xy = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        else:
            self._terrain_xy = None

    def _boundary_cb(self, msg: PolygonStamped) -> None:
        pts = [(float(p.x), float(p.y)) for p in msg.polygon.points]
        if pts:
            self._boundary_xy = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        else:
            self._boundary_xy = None

    def _reach_goal_cb(self, msg: Bool) -> None:
        self._goal_reached = bool(msg.data)

    def _timer_cb(self) -> None:
        if self._finished:
            if not self._finish_published:
                self._publish_finish(True)
                self._finish_published = True
            return
        if self._robot_xy is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        need_new_goal = self._goal_xy is None or self._goal_reached
        need_republish = (
            self._goal_xy is not None
            and now_sec - self._last_publish_sec >= self.goal_republish_sec
        )

        if need_new_goal:
            next_goal = self._select_next_goal()
            if next_goal is None:
                return
            if next_goal is False:
                self._finished = True
                self._publish_finish(True)
                self._finish_published = True
                self.get_logger().info("FAR auto-goal manager marked exploration finished.")
                return
            self._goal_xy = next_goal
            self._goal_reached = False
            self._publish_goal(self._goal_xy)
            self._last_publish_sec = now_sec
            return

        if need_republish:
            self._publish_goal(self._goal_xy)
            self._last_publish_sec = now_sec

    def _select_next_goal(self) -> Optional[np.ndarray | bool]:
        if self._terrain_xy is None or self._terrain_xy.shape[0] < 10:
            return None

        pts = self._terrain_xy
        if self._boundary_xy is not None and self._boundary_xy.shape[0] >= 4:
            xmin = float(np.min(self._boundary_xy[:, 0]))
            xmax = float(np.max(self._boundary_xy[:, 0]))
            ymin = float(np.min(self._boundary_xy[:, 1]))
            ymax = float(np.max(self._boundary_xy[:, 1]))
        else:
            xmin = float(np.min(pts[:, 0]))
            xmax = float(np.max(pts[:, 0]))
            ymin = float(np.min(pts[:, 1]))
            ymax = float(np.max(pts[:, 1]))

        width = xmax - xmin
        height = ymax - ymin
        if width < self.min_progress_extent_m or height < self.min_progress_extent_m:
            return None

        xmin += self.edge_inset_m
        xmax -= self.edge_inset_m
        ymin += self.edge_inset_m
        ymax -= self.edge_inset_m
        if xmax <= xmin or ymax <= ymin:
            return None

        robot_x, robot_y = float(self._robot_xy[0]), float(self._robot_xy[1])
        lane_y = np.clip(robot_y, ymin, ymax)
        lane_y = round(lane_y / max(self.lane_spacing_m, 1e-3)) * self.lane_spacing_m
        lane_y = float(np.clip(lane_y, ymin, ymax))

        lane_end_x = xmax if self._sweep_direction > 0 else xmin
        if self._goal_xy is not None:
            self._sweep_direction *= -1
            lane_end_x = xmax if self._sweep_direction > 0 else xmin

        near_end = abs(robot_x - lane_end_x) <= self.finish_margin_m
        if near_end:
            next_lane_y = lane_y + self.lane_spacing_m
            if next_lane_y > ymax:
                return False
            lane_y = next_lane_y
            lane_end_x = xmax if self._sweep_direction > 0 else xmin

        if self._sweep_direction > 0:
            goal_x = min(robot_x + self.lane_forward_step_m, lane_end_x)
        else:
            goal_x = max(robot_x - self.lane_forward_step_m, lane_end_x)

        forward_goal = np.asarray([goal_x, lane_y], dtype=np.float32)

        return forward_goal

    def _publish_goal(self, goal_xy: np.ndarray) -> None:
        msg = PointStamped()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(goal_xy[0])
        msg.point.y = float(goal_xy[1])
        msg.point.z = 0.0
        self.goal_pub.publish(msg)

    def _publish_finish(self, finished: bool) -> None:
        msg = Bool()
        msg.data = finished
        self.finish_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = FarAutoGoalManager()
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
