#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    semantic_share = get_package_share_directory("onemap_semantic_mapper")
    fast_livo_share = get_package_share_directory("fast_livo")

    default_semantic_config = os.path.join(semantic_share, "config", "semantic_mapper.yaml")
    default_rviz_config = os.path.join(semantic_share, "config", "semantic_fast_livo2.rviz")
    default_avia_config = os.path.join(fast_livo_share, "config", "avia_isaac.yaml")
    default_camera_config = os.path.join(fast_livo_share, "config", "camera_pinhole.yaml")
    default_stage_path = "/home/peng/isacc learned/tutle/turtle.usd"
    stage_runner = os.path.join(semantic_share, "scripts", "isaac_turtle_stage_runner.py")
    isaac_python = "/home/peng/IsaacSim/_build/linux-x86_64/release/python.sh"

    use_rviz = LaunchConfiguration("use_rviz")
    launch_isaac = LaunchConfiguration("launch_isaac")
    semantic_use_rviz = LaunchConfiguration("semantic_use_rviz")
    semantic_params = LaunchConfiguration("semantic_params_file")
    avia_params = LaunchConfiguration("avia_params_file")
    camera_params = LaunchConfiguration("camera_params_file")
    isaac_stage = LaunchConfiguration("isaac_stage")
    rviz_config = LaunchConfiguration("rviz_config")

    fast_livo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(fast_livo_share, "launch", "mapping_isaac.launch.py")),
        launch_arguments={
            "use_rviz": "False",
            "avia_params_file": avia_params,
            "camera_params_file": camera_params,
        }.items(),
    )

    semantic_mapper = Node(
        package="onemap_semantic_mapper",
        executable="semantic_mapper",
        name="semantic_mapper",
        output="screen",
        parameters=[
            avia_params,
            camera_params,
            semantic_params,
        ],
    )

    isaac_process = ExecuteProcess(
        condition=IfCondition(launch_isaac),
        cmd=[isaac_python, stage_runner, "--usd_path", isaac_stage],
        additional_env={
            "PYTHONPATH": "",
            "AMENT_PREFIX_PATH": "",
            "COLCON_PREFIX_PATH": "",
            "CMAKE_PREFIX_PATH": "",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "ROS_DISTRO": "humble",
        },
        output="screen",
    )

    rviz_node = Node(
        condition=IfCondition(semantic_use_rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="True"),
            DeclareLaunchArgument("launch_isaac", default_value="False"),
            SetLaunchConfiguration("semantic_use_rviz", use_rviz),
            DeclareLaunchArgument("semantic_params_file", default_value=default_semantic_config),
            DeclareLaunchArgument("avia_params_file", default_value=default_avia_config),
            DeclareLaunchArgument("camera_params_file", default_value=default_camera_config),
            DeclareLaunchArgument("isaac_stage", default_value=default_stage_path),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            fast_livo_launch,
            semantic_mapper,
            isaac_process,
            rviz_node,
        ]
    )
