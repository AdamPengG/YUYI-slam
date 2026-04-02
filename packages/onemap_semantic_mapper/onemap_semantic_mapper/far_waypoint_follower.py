from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _slew_limit(current: float, target: float, accel_limit: float, dt: float) -> float:
    if accel_limit <= 0.0 or dt <= 0.0:
        return target
    max_delta = accel_limit * dt
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


class FarWaypointFollower(Node):
    def __init__(self) -> None:
        super().__init__("far_waypoint_follower")
        self._declare_parameters()
        self._load_parameters()

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        waypoint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        finish_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(PointStamped, self.waypoint_topic, self._waypoint_cb, waypoint_qos)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, odom_qos)
        self.create_subscription(Bool, self.finish_topic, self._finish_cb, finish_qos)
        self.create_subscription(
            PointCloud2,
            self.registered_scan_topic,
            self._registered_scan_cb,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._control_loop)

        self._goal_x: float | None = None
        self._goal_y: float | None = None
        self._goal_stamp_sec: float | None = None
        self._exploration_finished: bool = False
        self._x: float | None = None
        self._y: float | None = None
        self._z: float | None = None
        self._yaw: float | None = None
        self._last_cmd_linear = 0.0
        self._last_cmd_angular = 0.0
        self._last_control_sec: float | None = None
        self._registered_scan_xyz = np.empty((0, 3), dtype=np.float32)
        self._last_scan_stamp_sec: float | None = None
        self._last_safety_log_sec: float = 0.0

        self.get_logger().info(
            "FAR waypoint follower ready. "
            f"waypoint={self.waypoint_topic}, odom={self.odom_topic}, cmd_vel={self.cmd_vel_topic}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "waypoint_topic": "/way_point",
            "odom_topic": "/state_estimation",
            "registered_scan_topic": "/registered_scan",
            "cmd_vel_topic": "/cmd_vel",
            "finish_topic": "/far_auto_exploration_finish",
            "control_rate_hz": 20.0,
            "goal_tolerance_m": 0.30,
            "heading_tolerance_rad": 0.18,
            "rotate_in_place_heading_rad": 0.55,
            "waypoint_timeout_sec": 8.0,
            "waypoint_stop_timeout_sec": 60.0,
            "stale_waypoint_slowdown_factor": 0.45,
            "max_linear_speed": 0.18,
            "max_angular_speed": 0.9,
            "max_linear_accel": 0.20,
            "max_linear_decel": 0.45,
            "max_angular_accel": 1.50,
            "linear_gain": 0.7,
            "angular_gain": 1.4,
            "slow_down_radius_m": 1.2,
            "use_obstacle_safety": True,
            "obstacle_stop_distance_m": 0.75,
            "obstacle_slow_distance_m": 1.40,
            "obstacle_corridor_half_width_m": 0.50,
            "obstacle_min_height_rel_m": -0.15,
            "obstacle_max_height_rel_m": 1.20,
            "scan_stale_timeout_sec": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.waypoint_topic = str(self.get_parameter("waypoint_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.registered_scan_topic = str(self.get_parameter("registered_scan_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.finish_topic = str(self.get_parameter("finish_topic").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)
        self.heading_tolerance_rad = float(self.get_parameter("heading_tolerance_rad").value)
        self.rotate_in_place_heading_rad = float(
            self.get_parameter("rotate_in_place_heading_rad").value
        )
        self.waypoint_timeout_sec = float(self.get_parameter("waypoint_timeout_sec").value)
        self.waypoint_stop_timeout_sec = float(
            self.get_parameter("waypoint_stop_timeout_sec").value
        )
        self.stale_waypoint_slowdown_factor = float(
            self.get_parameter("stale_waypoint_slowdown_factor").value
        )
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.max_linear_accel = float(self.get_parameter("max_linear_accel").value)
        self.max_linear_decel = float(self.get_parameter("max_linear_decel").value)
        self.max_angular_accel = float(self.get_parameter("max_angular_accel").value)
        self.linear_gain = float(self.get_parameter("linear_gain").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.slow_down_radius_m = float(self.get_parameter("slow_down_radius_m").value)
        self.use_obstacle_safety = bool(self.get_parameter("use_obstacle_safety").value)
        self.obstacle_stop_distance_m = float(
            self.get_parameter("obstacle_stop_distance_m").value
        )
        self.obstacle_slow_distance_m = float(
            self.get_parameter("obstacle_slow_distance_m").value
        )
        self.obstacle_corridor_half_width_m = float(
            self.get_parameter("obstacle_corridor_half_width_m").value
        )
        self.obstacle_min_height_rel_m = float(
            self.get_parameter("obstacle_min_height_rel_m").value
        )
        self.obstacle_max_height_rel_m = float(
            self.get_parameter("obstacle_max_height_rel_m").value
        )
        self.scan_stale_timeout_sec = float(
            self.get_parameter("scan_stale_timeout_sec").value
        )

    def _waypoint_cb(self, msg: PointStamped) -> None:
        if self._exploration_finished:
            return
        self._goal_x = float(msg.point.x)
        self._goal_y = float(msg.point.y)
        self._goal_stamp_sec = self.get_clock().now().nanoseconds * 1e-9

    def _finish_cb(self, msg: Bool) -> None:
        finished = bool(msg.data)
        if finished and not self._exploration_finished:
            self.get_logger().info("Received FAR exploration-finished signal, stopping robot.")
        self._exploration_finished = finished
        if finished:
            self._goal_x = None
            self._goal_y = None
            self._goal_stamp_sec = None
            self._publish_command(0.0, 0.0, immediate=True)

    def _odom_cb(self, msg: Odometry) -> None:
        self._x = float(msg.pose.pose.position.x)
        self._y = float(msg.pose.pose.position.y)
        self._z = float(msg.pose.pose.position.z)
        q = msg.pose.pose.orientation
        self._yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def _registered_scan_cb(self, msg: PointCloud2) -> None:
        points = [
            (float(p[0]), float(p[1]), float(p[2]))
            for p in point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        ]
        if points:
            self._registered_scan_xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        else:
            self._registered_scan_xyz = np.empty((0, 3), dtype=np.float32)
        self._last_scan_stamp_sec = (
            float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        )

    def _publish_command(
        self, target_linear: float, target_angular: float, *, immediate: bool = False
    ) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self._last_control_sec is None:
            dt = 1.0 / max(self.control_rate_hz, 1e-6)
        else:
            dt = max(1e-3, now_sec - self._last_control_sec)
        self._last_control_sec = now_sec

        if immediate:
            linear = target_linear
            angular = target_angular
        else:
            linear_accel = (
                self.max_linear_accel
                if target_linear >= self._last_cmd_linear
                else self.max_linear_decel
            )
            linear = _slew_limit(self._last_cmd_linear, target_linear, linear_accel, dt)
            angular = _slew_limit(
                self._last_cmd_angular, target_angular, self.max_angular_accel, dt
            )

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)
        self._last_cmd_linear = linear
        self._last_cmd_angular = angular

    def _compute_forward_clearance(self, now_sec: float) -> tuple[float | None, float]:
        if (
            not self.use_obstacle_safety
            or self._registered_scan_xyz.size == 0
            or self._x is None
            or self._y is None
            or self._z is None
            or self._yaw is None
        ):
            return None, 0.0
        if (
            self.scan_stale_timeout_sec > 0.0
            and self._last_scan_stamp_sec is not None
            and now_sec - self._last_scan_stamp_sec > self.scan_stale_timeout_sec
        ):
            return None, 0.0

        rel = self._registered_scan_xyz.copy()
        rel[:, 0] -= self._x
        rel[:, 1] -= self._y
        rel[:, 2] -= self._z

        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)
        x_body = cos_yaw * rel[:, 0] + sin_yaw * rel[:, 1]
        y_body = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]
        z_body = rel[:, 2]

        mask = (
            (x_body > 0.0)
            & (x_body < self.obstacle_slow_distance_m)
            & (np.abs(y_body) < self.obstacle_corridor_half_width_m)
            & (z_body > self.obstacle_min_height_rel_m)
            & (z_body < self.obstacle_max_height_rel_m)
        )
        if not np.any(mask):
            return None, 0.0

        forward = x_body[mask]
        lateral = y_body[mask]
        clearance = float(np.min(forward))
        left_hits = int(np.count_nonzero(lateral > 0.0))
        right_hits = int(np.count_nonzero(lateral < 0.0))
        turn_bias = float(right_hits - left_hits)
        return clearance, turn_bias

    def _control_loop(self) -> None:
        if self._exploration_finished:
            self._publish_command(0.0, 0.0)
            return
        if self._goal_x is None or self._x is None or self._yaw is None:
            self._publish_command(0.0, 0.0)
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        stale_scale = 1.0
        if (
            self.waypoint_timeout_sec > 0.0
            and self._goal_stamp_sec is not None
            and now_sec - self._goal_stamp_sec > self.waypoint_timeout_sec
        ):
            if (
                self.waypoint_stop_timeout_sec > 0.0
                and now_sec - self._goal_stamp_sec > self.waypoint_stop_timeout_sec
            ):
                self._publish_command(0.0, 0.0)
                return
            stale_scale = max(0.0, min(1.0, self.stale_waypoint_slowdown_factor))

        dx = self._goal_x - self._x
        dy = self._goal_y - self._y
        dist = math.hypot(dx, dy)
        if dist < self.goal_tolerance_m:
            self._publish_command(0.0, 0.0)
            return

        target_yaw = math.atan2(dy, dx)
        heading_error = _wrap_angle(target_yaw - self._yaw)
        angular = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.angular_gain * heading_error),
        )
        angular *= stale_scale

        if abs(heading_error) > self.rotate_in_place_heading_rad:
            self._publish_command(0.0, angular)
            return

        linear_scale = min(1.0, dist / max(self.slow_down_radius_m, 1e-6))
        linear = self.linear_gain * dist * linear_scale
        linear = min(self.max_linear_speed, linear)
        if abs(heading_error) > self.heading_tolerance_rad:
            linear *= max(0.0, 1.0 - abs(heading_error) / math.pi)
        linear *= stale_scale

        clearance, turn_bias = self._compute_forward_clearance(now_sec)
        if clearance is not None:
            if clearance <= self.obstacle_stop_distance_m:
                if abs(angular) < 0.25:
                    angular = 0.6 if turn_bias >= 0.0 else -0.6
                angular = max(-self.max_angular_speed, min(self.max_angular_speed, angular))
                if now_sec - self._last_safety_log_sec > 1.0:
                    self.get_logger().warn(
                        f"Obstacle ahead at {clearance:.2f} m, stopping linear motion."
                    )
                    self._last_safety_log_sec = now_sec
                self._publish_command(0.0, angular)
                return
            if clearance < self.obstacle_slow_distance_m:
                span = max(
                    self.obstacle_slow_distance_m - self.obstacle_stop_distance_m, 1e-6
                )
                factor = (clearance - self.obstacle_stop_distance_m) / span
                factor = max(0.15, min(1.0, factor))
                linear *= factor

        self._publish_command(linear, angular)


def main() -> None:
    rclpy.init()
    node = FarWaypointFollower()
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
