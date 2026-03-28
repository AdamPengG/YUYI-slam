#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

CAMERA_GRAPH_PATH = "/World/turtlebot3_burger_ROS/ROS_Camera"
CAMERA_PRIM_PATH = "/World/turtlebot3_burger_ROS/base_scan/Camera1"
CAMERA_FRAME_ID = "sim_camera"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_QOS_PROFILE = "Sensor Data"
CAMERA_QUEUE_SIZE = 1
CAMERA_FRAME_SKIP_COUNT = 3
IMU_GRAPH_PATH = "/World/turtlebot3_burger_ROS/ROS_IMU"
ORB_IMU_GRAPH_PATH = "/World/turtlebot3_burger_ROS/ROS_ORB_IMU"
IMU_FRAME_ID = "sim_imu"
IMU_FILTER_WIDTH = 10
PHYSICS_FREQUENCY = 120
IMU_SENSOR_PERIOD = 1.0 / PHYSICS_FREQUENCY
IMU_PUBLISH_STEP = 1
PHYSICS_SCENE_PATH = "/World/physicsScene"
LIDAR_DISABLE_KEYWORDS = ("lidar", "pointcloud")
LIVO2_GRAPH_PATH = "/World/turtlebot3_burger_ROS/ROS__Lidar"

PROFILE_CONFIG = {
    "livo2": {
        "imu_topic": "/livox/imu",
        "disable_lidar": False,
    },
    "orb": {
        "imu_topic": "/orb_slam3/imu",
        "disable_lidar": True,
    },
}


def _attr_has_connection(prim, attr_name: str) -> bool:
    attr = prim.GetAttribute(attr_name)
    return attr.IsValid() and bool(attr.GetConnections())


def _graph_needs_rebuild(stage) -> bool:
    graph = stage.GetPrimAtPath(CAMERA_GRAPH_PATH)
    if not graph.IsValid():
        return True

    render_product = stage.GetPrimAtPath(f"{CAMERA_GRAPH_PATH}/RenderProduct")
    rgb_publish = stage.GetPrimAtPath(f"{CAMERA_GRAPH_PATH}/RGBPublish")
    depth_publish = stage.GetPrimAtPath(f"{CAMERA_GRAPH_PATH}/DepthPublish")
    camera_info = stage.GetPrimAtPath(f"{CAMERA_GRAPH_PATH}/CameraInfoPublish")
    ros2_context = stage.GetPrimAtPath(f"{CAMERA_GRAPH_PATH}/Ros2Context")
    ros2_qos = stage.GetPrimAtPath(f"{CAMERA_GRAPH_PATH}/Ros2Qos")

    if not all(
        prim.IsValid()
        for prim in [render_product, rgb_publish, depth_publish, camera_info, ros2_context, ros2_qos]
    ):
        return True

    camera_prim_attr = render_product.GetAttribute("inputs:cameraPrim")
    width_attr = render_product.GetAttribute("inputs:width")
    height_attr = render_product.GetAttribute("inputs:height")
    rgb_queue_attr = rgb_publish.GetAttribute("inputs:queueSize")
    depth_queue_attr = depth_publish.GetAttribute("inputs:queueSize")
    camera_info_queue_attr = camera_info.GetAttribute("inputs:queueSize")
    rgb_skip_attr = rgb_publish.GetAttribute("inputs:frameSkipCount")
    depth_skip_attr = depth_publish.GetAttribute("inputs:frameSkipCount")
    camera_info_skip_attr = camera_info.GetAttribute("inputs:frameSkipCount")

    if not camera_prim_attr.IsValid():
        return True
    if not width_attr.IsValid() or width_attr.Get() != CAMERA_WIDTH:
        return True
    if not height_attr.IsValid() or height_attr.Get() != CAMERA_HEIGHT:
        return True

    try:
        camera_targets = [str(path) for path in camera_prim_attr.GetTargets()]
    except Exception:
        camera_targets = []
    if camera_targets != [CAMERA_PRIM_PATH]:
        return True

    if not all(
        attr.IsValid()
        for attr in [rgb_queue_attr, depth_queue_attr, camera_info_queue_attr, rgb_skip_attr, depth_skip_attr, camera_info_skip_attr]
    ):
        return True
    if rgb_queue_attr.Get() != CAMERA_QUEUE_SIZE or depth_queue_attr.Get() != CAMERA_QUEUE_SIZE:
        return True
    if camera_info_queue_attr.Get() != CAMERA_QUEUE_SIZE:
        return True
    if (
        rgb_skip_attr.Get() != CAMERA_FRAME_SKIP_COUNT
        or depth_skip_attr.Get() != CAMERA_FRAME_SKIP_COUNT
        or camera_info_skip_attr.Get() != CAMERA_FRAME_SKIP_COUNT
    ):
        return True

    expected_connections = [
        (render_product, "inputs:execIn", f"{CAMERA_GRAPH_PATH}/OnPlaybackTick.outputs:tick"),
        (rgb_publish, "inputs:renderProductPath", f"{CAMERA_GRAPH_PATH}/RenderProduct.outputs:renderProductPath"),
        (depth_publish, "inputs:renderProductPath", f"{CAMERA_GRAPH_PATH}/RenderProduct.outputs:renderProductPath"),
        (camera_info, "inputs:renderProductPath", f"{CAMERA_GRAPH_PATH}/RenderProduct.outputs:renderProductPath"),
        (rgb_publish, "inputs:context", f"{CAMERA_GRAPH_PATH}/Ros2Context.outputs:context"),
        (depth_publish, "inputs:context", f"{CAMERA_GRAPH_PATH}/Ros2Context.outputs:context"),
        (camera_info, "inputs:context", f"{CAMERA_GRAPH_PATH}/Ros2Context.outputs:context"),
        (rgb_publish, "inputs:qosProfile", f"{CAMERA_GRAPH_PATH}/Ros2Qos.outputs:qosProfile"),
        (depth_publish, "inputs:qosProfile", f"{CAMERA_GRAPH_PATH}/Ros2Qos.outputs:qosProfile"),
        (camera_info, "inputs:qosProfile", f"{CAMERA_GRAPH_PATH}/Ros2Qos.outputs:qosProfile"),
    ]
    for prim, attr_name, expected in expected_connections:
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            return True
        connections = [str(path) for path in attr.GetConnections()]
        if expected not in connections:
            return True

    return False


def _rebuild_ros_camera_graph(stage) -> bool:
    import omni.graph.core as og
    from pxr import Sdf

    if stage.GetPrimAtPath(CAMERA_GRAPH_PATH).IsValid():
        stage.RemovePrim(CAMERA_GRAPH_PATH)

    og.Controller.edit(
        {"graph_path": CAMERA_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("Ros2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("Ros2Qos", "isaacsim.ros2.bridge.ROS2QoSProfile"),
                ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", [Sdf.Path(CAMERA_PRIM_PATH)]),
                ("RenderProduct.inputs:width", CAMERA_WIDTH),
                ("RenderProduct.inputs:height", CAMERA_HEIGHT),
                ("Ros2Qos.inputs:createProfile", CAMERA_QOS_PROFILE),
                ("Ros2Qos.inputs:depth", CAMERA_QUEUE_SIZE),
                ("RGBPublish.inputs:topicName", "/robot_rgb"),
                ("RGBPublish.inputs:type", "rgb"),
                ("RGBPublish.inputs:frameId", CAMERA_FRAME_ID),
                ("RGBPublish.inputs:queueSize", CAMERA_QUEUE_SIZE),
                ("RGBPublish.inputs:frameSkipCount", CAMERA_FRAME_SKIP_COUNT),
                ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                ("DepthPublish.inputs:topicName", "/depth"),
                ("DepthPublish.inputs:type", "depth"),
                ("DepthPublish.inputs:frameId", CAMERA_FRAME_ID),
                ("DepthPublish.inputs:queueSize", CAMERA_QUEUE_SIZE),
                ("DepthPublish.inputs:frameSkipCount", CAMERA_FRAME_SKIP_COUNT),
                ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
                ("CameraInfoPublish.inputs:topicName", "camera_info"),
                ("CameraInfoPublish.inputs:frameId", CAMERA_FRAME_ID),
                ("CameraInfoPublish.inputs:queueSize", CAMERA_QUEUE_SIZE),
                ("CameraInfoPublish.inputs:frameSkipCount", CAMERA_FRAME_SKIP_COUNT),
                ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "DepthPublish.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                ("RenderProduct.outputs:renderProductPath", "DepthPublish.inputs:renderProductPath"),
                ("RenderProduct.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
                ("Ros2Context.outputs:context", "RGBPublish.inputs:context"),
                ("Ros2Context.outputs:context", "DepthPublish.inputs:context"),
                ("Ros2Context.outputs:context", "CameraInfoPublish.inputs:context"),
                ("Ros2Qos.outputs:qosProfile", "RGBPublish.inputs:qosProfile"),
                ("Ros2Qos.outputs:qosProfile", "DepthPublish.inputs:qosProfile"),
                ("Ros2Qos.outputs:qosProfile", "CameraInfoPublish.inputs:qosProfile"),
            ],
        },
    )
    return True


def _find_imu_prim_path(stage) -> str:
    preferred_prefixes = [
        "/World/turtlebot3_burger_ROS",
        "/World/turtlebot3_burger",
    ]

    for prefix in preferred_prefixes:
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(prefix) and prim.GetName() == "Imu_Sensor":
                return path

    for prim in stage.Traverse():
        if prim.GetName() == "Imu_Sensor":
            return str(prim.GetPath())

    raise RuntimeError("Could not locate an Imu_Sensor prim in the current TurtleBot3 stage.")


def _imu_graph_needs_rebuild(stage, imu_prim_path: str, imu_topic_name: str) -> bool:
    graph = stage.GetPrimAtPath(IMU_GRAPH_PATH)
    if not graph.IsValid():
        return True

    read_imu = stage.GetPrimAtPath(f"{IMU_GRAPH_PATH}/ReadIMU")
    publish_imu = stage.GetPrimAtPath(f"{IMU_GRAPH_PATH}/PublishIMU")
    playback_tick = stage.GetPrimAtPath(f"{IMU_GRAPH_PATH}/OnPlaybackTick")
    simulation_gate = stage.GetPrimAtPath(f"{IMU_GRAPH_PATH}/SimulationGate")
    read_sim_time = stage.GetPrimAtPath(f"{IMU_GRAPH_PATH}/ReadSimulationTime")
    ros2_context = stage.GetPrimAtPath(f"{IMU_GRAPH_PATH}/Ros2Context")
    ros2_qos = stage.GetPrimAtPath(f"{IMU_GRAPH_PATH}/Ros2Qos")

    if not all(
        prim.IsValid()
        for prim in [read_imu, publish_imu, playback_tick, simulation_gate, read_sim_time, ros2_context, ros2_qos]
    ):
        return True

    imu_prim_attr = read_imu.GetAttribute("inputs:imuPrim")
    latest_attr = read_imu.GetAttribute("inputs:useLatestData")
    gate_step_attr = simulation_gate.GetAttribute("inputs:step")
    topic_attr = publish_imu.GetAttribute("inputs:topicName")
    frame_attr = publish_imu.GetAttribute("inputs:frameId")

    if not all(attr.IsValid() for attr in [imu_prim_attr, latest_attr, gate_step_attr, topic_attr, frame_attr]):
        return True

    try:
        imu_targets = [str(path) for path in imu_prim_attr.GetTargets()]
    except Exception:
        imu_targets = []
    if imu_targets != [imu_prim_path]:
        return True

    if topic_attr.Get() != imu_topic_name or frame_attr.Get() != IMU_FRAME_ID:
        return True
    if latest_attr.Get() is not False:
        return True
    if gate_step_attr.Get() != IMU_PUBLISH_STEP:
        return True

    expected_connections = [
        (simulation_gate, "inputs:execIn", f"{IMU_GRAPH_PATH}/OnPlaybackTick.outputs:tick"),
        (read_imu, "inputs:execIn", f"{IMU_GRAPH_PATH}/SimulationGate.outputs:execOut"),
        (publish_imu, "inputs:execIn", f"{IMU_GRAPH_PATH}/ReadIMU.outputs:execOut"),
        (publish_imu, "inputs:orientation", f"{IMU_GRAPH_PATH}/ReadIMU.outputs:orientation"),
        (publish_imu, "inputs:linearAcceleration", f"{IMU_GRAPH_PATH}/ReadIMU.outputs:linAcc"),
        (publish_imu, "inputs:angularVelocity", f"{IMU_GRAPH_PATH}/ReadIMU.outputs:angVel"),
        (publish_imu, "inputs:timeStamp", f"{IMU_GRAPH_PATH}/ReadSimulationTime.outputs:simulationTime"),
        (publish_imu, "inputs:context", f"{IMU_GRAPH_PATH}/Ros2Context.outputs:context"),
        (publish_imu, "inputs:qosProfile", f"{IMU_GRAPH_PATH}/Ros2Qos.outputs:qosProfile"),
    ]
    for prim, attr_name, expected in expected_connections:
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            return True
        if expected is None:
            continue
        connections = [str(path) for path in attr.GetConnections()]
        if expected not in connections:
            return True

    return False


def _rebuild_ros_imu_graph(stage, imu_prim_path: str, imu_topic_name: str) -> bool:
    import omni.graph.core as og
    from pxr import Sdf

    if stage.GetPrimAtPath(IMU_GRAPH_PATH).IsValid():
        stage.RemovePrim(IMU_GRAPH_PATH)

    og.Controller.edit(
        {"graph_path": IMU_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("SimulationGate", "isaacsim.core.nodes.IsaacSimulationGate"),
                ("ReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
                ("ReadSimulationTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Ros2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("Ros2Qos", "isaacsim.ros2.bridge.ROS2QoSProfile"),
                ("PublishIMU", "isaacsim.ros2.bridge.ROS2PublishImu"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("SimulationGate.inputs:step", IMU_PUBLISH_STEP),
                ("ReadIMU.inputs:imuPrim", [Sdf.Path(imu_prim_path)]),
                ("ReadIMU.inputs:useLatestData", False),
                ("ReadIMU.inputs:readGravity", True),
                ("ReadSimulationTime.inputs:resetOnStop", True),
                ("Ros2Qos.inputs:createProfile", "Sensor Data"),
                ("Ros2Qos.inputs:depth", 5),
                ("Ros2Qos.inputs:reliability", "bestEffort"),
                ("PublishIMU.inputs:topicName", imu_topic_name),
                ("PublishIMU.inputs:frameId", IMU_FRAME_ID),
                ("PublishIMU.inputs:publishOrientation", True),
                ("PublishIMU.inputs:publishLinearAcceleration", True),
                ("PublishIMU.inputs:publishAngularVelocity", True),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SimulationGate.inputs:execIn"),
                ("SimulationGate.outputs:execOut", "ReadIMU.inputs:execIn"),
                ("ReadIMU.outputs:execOut", "PublishIMU.inputs:execIn"),
                ("ReadIMU.outputs:orientation", "PublishIMU.inputs:orientation"),
                ("ReadIMU.outputs:linAcc", "PublishIMU.inputs:linearAcceleration"),
                ("ReadIMU.outputs:angVel", "PublishIMU.inputs:angularVelocity"),
                ("ReadSimulationTime.outputs:simulationTime", "PublishIMU.inputs:timeStamp"),
                ("Ros2Context.outputs:context", "PublishIMU.inputs:context"),
                ("Ros2Qos.outputs:qosProfile", "PublishIMU.inputs:qosProfile"),
            ],
        },
    )
    return True


def _ensure_imu_sensor_defaults(stage, imu_prim_path: str) -> bool:
    imu_prim = stage.GetPrimAtPath(imu_prim_path)
    if not imu_prim.IsValid():
        raise RuntimeError(f"IMU prim not found at {imu_prim_path}")

    changed = False
    desired_values = {
        "angularVelocityFilterWidth": IMU_FILTER_WIDTH,
        "linearAccelerationFilterWidth": IMU_FILTER_WIDTH,
        "orientationFilterWidth": IMU_FILTER_WIDTH,
        "sensorPeriod": IMU_SENSOR_PERIOD,
        "visualize": False,
        "isaac:nameOverride": IMU_FRAME_ID,
    }

    for attr_name, desired_value in desired_values.items():
        attr = imu_prim.GetAttribute(attr_name)
        if not attr.IsValid():
            continue
        current_value = attr.Get()
        if current_value != desired_value:
            attr.Set(desired_value)
            changed = True

    return changed


def _set_physics_frequency(stage, timeline) -> bool:
    import carb
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    changed = False
    settings = carb.settings.get_settings()
    desired_int_settings = {
        "/app/runLoops/main/rateLimitFrequency": PHYSICS_FREQUENCY,
        "/persistent/simulation/minFrameRate": PHYSICS_FREQUENCY,
    }
    desired_bool_settings = {
        "/app/player/useFixedTimeStepping": True,
        "/app/runLoops/main/rateLimitEnabled": True,
    }

    for key, desired in desired_int_settings.items():
        current = settings.get_as_int(key)
        if current != desired:
            settings.set_int(key, desired)
            changed = True

    for key, desired in desired_bool_settings.items():
        current = settings.get_as_bool(key)
        if current != desired:
            settings.set_bool(key, desired)
            changed = True

    if int(timeline.get_target_framerate()) != PHYSICS_FREQUENCY:
        timeline.set_target_framerate(PHYSICS_FREQUENCY)
        changed = True

    physics_scene_prim = None
    candidate_paths = ["/World/physicsScene", "/physicsScene", "/PhysicsScene"]
    for path in candidate_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            physics_scene_prim = prim
            break

    for prim in stage.Traverse():
        if physics_scene_prim is not None:
            break
        if prim.GetTypeName() == "PhysicsScene" or prim.IsA(UsdPhysics.Scene):
            physics_scene_prim = prim
            break

    if physics_scene_prim is None:
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path(PHYSICS_SCENE_PATH))
        physics_scene_prim = scene.GetPrim()
        changed = True
        gravity_direction_attr = scene.GetGravityDirectionAttr()
        gravity_magnitude_attr = scene.GetGravityMagnitudeAttr()
        if not gravity_direction_attr.IsValid() or gravity_direction_attr.Get() != Gf.Vec3f(0.0, 0.0, -1.0):
            gravity_direction_attr.Set(Gf.Vec3f(0.0, 0.0, -1.0))
            changed = True
        if not gravity_magnitude_attr.IsValid() or gravity_magnitude_attr.Get() != 9.81:
            gravity_magnitude_attr.Set(9.81)
            changed = True
        print(f"[PHYSICS_SCENE_FOUND] created={PHYSICS_SCENE_PATH}", flush=True)
    else:
        print(f"[PHYSICS_SCENE_FOUND] existing={physics_scene_prim.GetPath()}", flush=True)

    physics_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    time_steps_attr = physics_scene_api.GetTimeStepsPerSecondAttr()
    if not time_steps_attr.IsValid() or time_steps_attr.Get() != PHYSICS_FREQUENCY:
        time_steps_attr.Set(PHYSICS_FREQUENCY)
        changed = True
    print(f"[PHYSICS_RATE_TARGET] hz={PHYSICS_FREQUENCY}", flush=True)
    print(f"[PHYSICS_RATE_APPLIED] hz={time_steps_attr.Get()}", flush=True)

    return changed


def _runtime_disable_lidar_prims(stage) -> list[str]:
    candidate_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        lower_path = path.lower()
        if any(keyword in lower_path for keyword in LIDAR_DISABLE_KEYWORDS):
            candidate_paths.append(path)

    candidate_paths.sort(key=len)
    disabled_paths: list[str] = []
    for path in candidate_paths:
        if any(path == disabled or path.startswith(f"{disabled}/") for disabled in disabled_paths):
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        lower_path = path.lower()
        if prim.GetTypeName() == "OmniGraph" or "ros__lidar" in lower_path or "pointcloud" in lower_path:
            stage.RemovePrim(path)
        elif prim.IsActive():
            prim.SetActive(False)
        else:
            continue
        disabled_paths.append(path)

    return disabled_paths


def _prune_legacy_livo2_camera_publishers(stage) -> list[str]:
    ros_lidar_graph = stage.GetPrimAtPath(LIVO2_GRAPH_PATH)
    if not ros_lidar_graph.IsValid():
        return []

    removed_paths: list[str] = []
    removable_node_names = {
        "isaac_get_viewport_render_product",
        "ros2_context_01",
        "ros2_camera_info_helper",
        "ros2_camera_helper",
        "isaac_run_one_simulation_frame",
        "isaac_create_render_product",
        "isaac_create_viewport",
    }

    for child in list(ros_lidar_graph.GetChildren()):
        remove_child = child.GetName() in removable_node_names
        if not remove_child:
            topic_attr = child.GetAttribute("inputs:topicName")
            if topic_attr.IsValid():
                topic_name = topic_attr.Get()
                if topic_name in {"/rgb", "rgb", "/camera_info", "camera_info"}:
                    remove_child = True
        if remove_child:
            path = str(child.GetPath())
            stage.RemovePrim(path)
            removed_paths.append(path)

    return removed_paths


def _remove_graph_if_exists(stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    stage.RemovePrim(prim_path)
    return True


def _sanitize_livo2_lidar_graph(stage) -> bool:
    ros_lidar_graph = stage.GetPrimAtPath(LIVO2_GRAPH_PATH)
    if not ros_lidar_graph.IsValid():
        return False

    changed = False
    if not ros_lidar_graph.IsActive():
        ros_lidar_graph.SetActive(True)
        changed = True

    desired_topics = {
        f"{LIVO2_GRAPH_PATH}/ros2_publish_point_cloud": "/livox/lidar",
        f"{LIVO2_GRAPH_PATH}/ros2_publish_imu": "/livox/imu",
    }
    desired_frames = {
        f"{LIVO2_GRAPH_PATH}/ros2_publish_point_cloud": "sim_lidar",
        f"{LIVO2_GRAPH_PATH}/ros2_publish_imu": IMU_FRAME_ID,
    }

    for prim_path, topic_name in desired_topics.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        if not prim.IsActive():
            prim.SetActive(True)
            changed = True
        topic_attr = prim.GetAttribute("inputs:topicName")
        if topic_attr.IsValid() and topic_attr.Get() != topic_name:
            topic_attr.Set(topic_name)
            changed = True
        frame_attr = prim.GetAttribute("inputs:frameId")
        desired_frame = desired_frames[prim_path]
        if frame_attr.IsValid() and frame_attr.Get() != desired_frame:
            frame_attr.Set(desired_frame)
            changed = True

    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Isaac Sim with the prepared TurtleBot3 semantic-mapping scene.")
    parser.add_argument(
        "--usd_path",
        default="/home/peng/isacc learned/tutle/turtle.usd",
        help="Absolute path to the USD stage to open.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIG.keys()),
        default="livo2",
        help="Sensor graph profile to apply. Use 'livo2' for FAST-LIVO2 and 'orb' for ORB-SLAM3.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without opening a GUI window.")
    parser.add_argument(
        "--headless_duration_sec",
        type=float,
        default=120.0,
        help="When --headless is set, keep the simulation alive for this many wall-clock seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usd_path = Path(args.usd_path).expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD stage not found: {usd_path}")
    profile = PROFILE_CONFIG[args.profile]
    imu_topic_name = profile["imu_topic"]
    disable_lidar_for_profile = profile["disable_lidar"]

    app = SimulationApp(
        {
            "headless": args.headless,
            "sync_loads": True,
            "width": 1280,
            "height": 720,
            "active_gpu": 0,
            "physics_gpu": 0,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "renderer": "RaytracedLighting",
        }
    )

    try:
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import is_stage_loading
        import omni.usd
        import omni.timeline

        enable_extension("isaacsim.core.nodes")
        enable_extension("isaacsim.robot.wheeled_robots")
        enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.sensors.physics")

        app.update()
        app.update()

        context = omni.usd.get_context()
        if not context.open_stage(str(usd_path)):
            raise RuntimeError(f"Failed to open stage: {usd_path}")

        while is_stage_loading():
            app.update()

        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("USD context returned no stage after open_stage().")

        timeline = omni.timeline.get_timeline_interface()
        physics_updated = _set_physics_frequency(stage, timeline)

        imu_prim_path = _find_imu_prim_path(stage)

        imu_sensor_updated = _ensure_imu_sensor_defaults(stage, imu_prim_path)

        camera_graph_rebuilt = False
        if _graph_needs_rebuild(stage):
            camera_graph_rebuilt = _rebuild_ros_camera_graph(stage)

        imu_graph_rebuilt = False
        removed_aux_imu_graphs: list[str] = []
        livo2_lidar_graph_updated = False

        if args.profile == "livo2":
            for graph_path in (IMU_GRAPH_PATH, ORB_IMU_GRAPH_PATH):
                if _remove_graph_if_exists(stage, graph_path):
                    removed_aux_imu_graphs.append(graph_path)
            livo2_lidar_graph_updated = _sanitize_livo2_lidar_graph(stage)
        else:
            if _imu_graph_needs_rebuild(stage, imu_prim_path, imu_topic_name):
                imu_graph_rebuilt = _rebuild_ros_imu_graph(stage, imu_prim_path, imu_topic_name)

        if (
            camera_graph_rebuilt
            or imu_graph_rebuilt
            or imu_sensor_updated
            or physics_updated
            or livo2_lidar_graph_updated
            or removed_aux_imu_graphs
        ):
            app.update()
            stage.GetRootLayer().Save()

        pruned_livo2_camera_paths: list[str] = []
        if args.profile == "livo2":
            pruned_livo2_camera_paths = _prune_legacy_livo2_camera_publishers(stage)
            if pruned_livo2_camera_paths:
                app.update()
                stage.GetRootLayer().Save()

        disabled_lidar_paths: list[str] = []
        if disable_lidar_for_profile:
            disabled_lidar_paths = _runtime_disable_lidar_prims(stage)
            if disabled_lidar_paths:
                app.update()
                stage.GetRootLayer().Save()

        root_layer = stage.GetRootLayer()
        print(f"opened_stage={root_layer.realPath or root_layer.identifier}", flush=True)

        has_robot = stage.GetPrimAtPath("/World/turtlebot3_burger_ROS").IsValid() or stage.GetPrimAtPath(
            "/World/turtlebot3_burger"
        ).IsValid()
        print(f"robot_present={has_robot}", flush=True)
        print(f"camera_graph_rebuilt={camera_graph_rebuilt}", flush=True)
        print(f"profile={args.profile}", flush=True)
        print(f"imu_topic={imu_topic_name}", flush=True)
        print(f"imu_prim_path={imu_prim_path}", flush=True)
        print(f"imu_sensor_updated={imu_sensor_updated}", flush=True)
        print(f"imu_graph_rebuilt={imu_graph_rebuilt}", flush=True)
        print(f"livo2_lidar_graph_updated={livo2_lidar_graph_updated}", flush=True)
        print(f"removed_aux_imu_graphs={removed_aux_imu_graphs}", flush=True)
        print(f"physics_updated={physics_updated}", flush=True)
        print(f"physics_frequency={PHYSICS_FREQUENCY}", flush=True)
        print(f"imu_sensor_period={IMU_SENSOR_PERIOD:.9f}", flush=True)
        print(f"livo2_camera_publishers_pruned={bool(pruned_livo2_camera_paths)}", flush=True)
        if pruned_livo2_camera_paths:
            print(f"livo2_camera_publisher_paths={pruned_livo2_camera_paths}", flush=True)
        print(f"lidar_disabled={bool(disabled_lidar_paths)}", flush=True)
        if disabled_lidar_paths:
            print(f"lidar_disabled_paths={disabled_lidar_paths}", flush=True)

        timeline.play()

        if args.headless:
            end_time = time.monotonic() + max(args.headless_duration_sec, 1.0)
            while time.monotonic() < end_time:
                app.update()
        else:
            while app.is_running():
                app.update()
    except Exception as exc:
        print(f"Isaac stage runner failed: {exc}", file=sys.stderr)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
