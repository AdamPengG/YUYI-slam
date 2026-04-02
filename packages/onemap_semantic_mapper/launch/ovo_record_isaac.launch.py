#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    semantic_share = get_package_share_directory("onemap_semantic_mapper")
    fast_livo_share = get_package_share_directory("fast_livo")

    default_avia_config = os.path.join(fast_livo_share, "config", "avia_isaac.yaml")
    default_camera_config = os.path.join(fast_livo_share, "config", "camera_pinhole.yaml")
    default_stage_path = "/home/peng/isacc learned/tutle/turtle.usd"
    default_output_root = "/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica"
    default_config_root = "/home/peng/isacc_slam/reference/OVO/data/working/configs/Replica"
    stage_runner = os.path.join(semantic_share, "scripts", "isaac_turtle_stage_runner.py")
    isaac_python = "/home/peng/IsaacSim/_build/linux-x86_64/release/python.sh"

    use_rviz = LaunchConfiguration("use_rviz")
    launch_isaac = LaunchConfiguration("launch_isaac")
    avia_params = LaunchConfiguration("avia_params_file")
    camera_params = LaunchConfiguration("camera_params_file")
    isaac_stage = LaunchConfiguration("isaac_stage")
    scene_name = LaunchConfiguration("scene_name")
    output_root = LaunchConfiguration("output_root")
    config_root = LaunchConfiguration("config_root")
    frame_stride = LaunchConfiguration("frame_stride")
    max_frames = LaunchConfiguration("max_frames")
    overwrite_scene = LaunchConfiguration("overwrite_scene")
    use_keyframe_filter = LaunchConfiguration("use_keyframe_filter")
    keyframe_min_translation_m = LaunchConfiguration("keyframe_min_translation_m")
    keyframe_min_rotation_deg = LaunchConfiguration("keyframe_min_rotation_deg")
    keyframe_min_frame_gap = LaunchConfiguration("keyframe_min_frame_gap")

    fast_livo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(fast_livo_share, "launch", "mapping_isaac.launch.py")),
        launch_arguments={
            "use_rviz": use_rviz,
            "avia_params_file": avia_params,
            "camera_params_file": camera_params,
            "use_image_republish": "False",
        }.items(),
    )

    exporter = Node(
        package="onemap_semantic_mapper",
        executable="ovo_dataset_exporter",
        name="ovo_dataset_exporter",
        output="screen",
        parameters=[
            avia_params,
            camera_params,
            {
                "export.scene_name": scene_name,
                "export.output_root": output_root,
                "export.config_root": config_root,
                "export.frame_stride": frame_stride,
                "export.max_frames": max_frames,
                "export.overwrite_scene": overwrite_scene,
                "export.use_keyframe_filter": use_keyframe_filter,
                "export.keyframe_min_translation_m": keyframe_min_translation_m,
                "export.keyframe_min_rotation_deg": keyframe_min_rotation_deg,
                "export.keyframe_min_frame_gap": keyframe_min_frame_gap,
            },
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

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="True"),
            DeclareLaunchArgument("launch_isaac", default_value="False"),
            DeclareLaunchArgument("avia_params_file", default_value=default_avia_config),
            DeclareLaunchArgument("camera_params_file", default_value=default_camera_config),
            DeclareLaunchArgument("isaac_stage", default_value=default_stage_path),
            DeclareLaunchArgument("scene_name", default_value="isaac_turtlebot3"),
            DeclareLaunchArgument("output_root", default_value=default_output_root),
            DeclareLaunchArgument("config_root", default_value=default_config_root),
            DeclareLaunchArgument("frame_stride", default_value="1"),
            DeclareLaunchArgument("max_frames", default_value="-1"),
            DeclareLaunchArgument("overwrite_scene", default_value="True"),
            DeclareLaunchArgument("use_keyframe_filter", default_value="True"),
            DeclareLaunchArgument("keyframe_min_translation_m", default_value="0.25"),
            DeclareLaunchArgument("keyframe_min_rotation_deg", default_value="12.0"),
            DeclareLaunchArgument("keyframe_min_frame_gap", default_value="5"),
            fast_livo_launch,
            exporter,
            isaac_process,
        ]
    )
