#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.core.nodes")
enable_extension("isaacsim.robot.wheeled_robots")
enable_extension("isaacsim.ros2.bridge")

import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
import usdrt.Sdf  # noqa: E402


def _wait_updates(count: int = 20) -> None:
    for _ in range(count):
        APP.update()


def _configure_action_graph(graph_path: str, robot_path: str, topic_name: str) -> None:
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(graph_path).IsValid():
        stage.RemovePrim(graph_path)
        _wait_updates(2)

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ROS2SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("BreakLinear", "omni.graph.nodes.BreakVector3"),
                ("BreakAngular", "omni.graph.nodes.BreakVector3"),
                ("DifferentialController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ROS2SubscribeTwist.inputs:execIn"),
                ("ROS2Context.outputs:context", "ROS2SubscribeTwist.inputs:context"),
                ("ROS2SubscribeTwist.outputs:execOut", "DifferentialController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("ROS2SubscribeTwist.outputs:linearVelocity", "BreakLinear.inputs:tuple"),
                ("BreakLinear.outputs:x", "DifferentialController.inputs:linearVelocity"),
                ("ROS2SubscribeTwist.outputs:angularVelocity", "BreakAngular.inputs:tuple"),
                ("BreakAngular.outputs:z", "DifferentialController.inputs:angularVelocity"),
                ("DifferentialController.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ],
            keys.SET_VALUES: [
                ("ROS2SubscribeTwist.inputs:topicName", topic_name),
                ("DifferentialController.inputs:wheelRadius", 0.025),
                ("DifferentialController.inputs:wheelDistance", 0.16),
                ("DifferentialController.inputs:maxLinearSpeed", 0.22),
                ("DifferentialController.inputs:maxAngularSpeed", 1.0),
                ("ArticulationController.inputs:jointNames", ["wheel_left_joint", "wheel_right_joint"]),
                ("ArticulationController.inputs:targetPrim", [usdrt.Sdf.Path(robot_path)]),
            ],
        },
    )

    node_positions = {
        "OnPlaybackTick": (260.0, 650.0),
        "ROS2Context": (260.0, 430.0),
        "ROS2SubscribeTwist": (570.0, 500.0),
        "BreakLinear": (860.0, 590.0),
        "BreakAngular": (860.0, 360.0),
        "DifferentialController": (1160.0, 470.0),
        "ArticulationController": (1490.0, 470.0),
    }
    for node_name, pos in node_positions.items():
        attr = og.Controller.attribute(f"{graph_path}/{node_name}.ui:nodegraph:node:pos")
        if attr.is_valid():
            attr.set(pos)


def _verify_stage(stage_path: Path) -> None:
    ctx = omni.usd.get_context()
    ctx.open_stage(str(stage_path))
    _wait_updates(20)
    stage = ctx.get_stage()
    robot = stage.GetPrimAtPath("/World/turtlebot3_burger")
    graph = stage.GetPrimAtPath("/World/ActionGraph")
    print(f"robot_valid={robot.IsValid()}")
    print(f"robot_apis={list(robot.GetAppliedSchemas()) if robot.IsValid() else []}")
    print(f"graph_valid={graph.IsValid()}")
    for node_name in [
        "OnPlaybackTick",
        "ROS2Context",
        "ROS2SubscribeTwist",
        "BreakLinear",
        "BreakAngular",
        "DifferentialController",
        "ArticulationController",
    ]:
        prim = stage.GetPrimAtPath(f"/World/ActionGraph/{node_name}")
        print(f"node:{node_name}={prim.IsValid()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure a TurtleBot3 Burger stage and ActionGraph for ROS2 cmd_vel."
    )
    parser.add_argument(
        "--input",
        default="/home/peng/isacc learned/tutle/turtle.usd",
        help="Input USD stage path.",
    )
    parser.add_argument(
        "--output",
        default="/home/peng/isacc learned/tutle/turtle_actiongraph_ready.usd",
        help="Output USD stage path.",
    )
    parser.add_argument(
        "--topic",
        default="cmd_vel",
        help="ROS2 Twist topic consumed by the ActionGraph.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input stage not found: {input_path}")

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(input_path)):
        raise RuntimeError(f"Failed to open stage: {input_path}")

    _wait_updates(20)

    stage = ctx.get_stage()
    robot = stage.GetPrimAtPath("/World/turtlebot3_burger")
    if not robot.IsValid():
        raise RuntimeError("Expected /World/turtlebot3_burger in the stage, but it was not found.")

    _configure_action_graph("/World/ActionGraph", "/World/turtlebot3_burger", args.topic)
    _wait_updates(5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not ctx.save_as_stage(str(output_path)):
        raise RuntimeError(f"Failed to save configured stage to: {output_path}")

    print(f"saved={output_path}")
    _verify_stage(output_path)


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
