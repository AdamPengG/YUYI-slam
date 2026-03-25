#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

CAMERA_GRAPH_PATH = "/World/turtlebot3_burger_ROS/ROS_Camera"
CAMERA_PRIM_PATH = "/World/turtlebot3_burger_ROS/base_scan/Camera1"
CAMERA_FRAME_ID = "sim_camera"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


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

    if not all(prim.IsValid() for prim in [render_product, rgb_publish, depth_publish, camera_info]):
        return True

    camera_prim_attr = render_product.GetAttribute("inputs:cameraPrim")
    width_attr = render_product.GetAttribute("inputs:width")
    height_attr = render_product.GetAttribute("inputs:height")

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

    expected_connections = [
        (rgb_publish, "inputs:renderProductPath", f"{CAMERA_GRAPH_PATH}/RenderProduct.outputs:renderProductPath"),
        (depth_publish, "inputs:renderProductPath", f"{CAMERA_GRAPH_PATH}/RenderProduct.outputs:renderProductPath"),
        (camera_info, "inputs:renderProductPath", f"{CAMERA_GRAPH_PATH}/RenderProduct.outputs:renderProductPath"),
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
                ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", [Sdf.Path(CAMERA_PRIM_PATH)]),
                ("RenderProduct.inputs:width", CAMERA_WIDTH),
                ("RenderProduct.inputs:height", CAMERA_HEIGHT),
                ("RGBPublish.inputs:topicName", "/robot_rgb"),
                ("RGBPublish.inputs:type", "rgb"),
                ("RGBPublish.inputs:frameId", CAMERA_FRAME_ID),
                ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                ("DepthPublish.inputs:topicName", "/depth"),
                ("DepthPublish.inputs:type", "depth"),
                ("DepthPublish.inputs:frameId", CAMERA_FRAME_ID),
                ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
                ("CameraInfoPublish.inputs:topicName", "camera_info"),
                ("CameraInfoPublish.inputs:frameId", CAMERA_FRAME_ID),
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
            ],
        },
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Isaac Sim with the prepared TurtleBot3 semantic-mapping scene.")
    parser.add_argument(
        "--usd_path",
        default="/home/peng/isacc learned/tutle/turtle.usd",
        help="Absolute path to the USD stage to open.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without opening a GUI window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usd_path = Path(args.usd_path).expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD stage not found: {usd_path}")

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

        camera_graph_rebuilt = False
        if _graph_needs_rebuild(stage):
            camera_graph_rebuilt = _rebuild_ros_camera_graph(stage)
            app.update()
            stage.GetRootLayer().Save()

        root_layer = stage.GetRootLayer()
        print(f"opened_stage={root_layer.realPath or root_layer.identifier}", flush=True)

        has_robot = stage.GetPrimAtPath("/World/turtlebot3_burger_ROS").IsValid() or stage.GetPrimAtPath(
            "/World/turtlebot3_burger"
        ).IsValid()
        print(f"robot_present={has_robot}", flush=True)
        print(f"camera_graph_rebuilt={camera_graph_rebuilt}", flush=True)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        while app.is_running():
            app.update()
    except Exception as exc:
        print(f"Isaac stage runner failed: {exc}", file=sys.stderr)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
