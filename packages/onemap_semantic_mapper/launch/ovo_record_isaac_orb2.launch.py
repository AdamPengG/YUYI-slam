#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    semantic_share = get_package_share_directory("onemap_semantic_mapper")
    fast_livo_share = get_package_share_directory("fast_livo")
    base_launch = os.path.join(semantic_share, "launch", "ovo_record_isaac.launch.py")
    default_avia_config = os.path.join(fast_livo_share, "config", "avia_isaac.yaml")
    default_camera_config = os.path.join(fast_livo_share, "config", "camera_pinhole.yaml")
    default_stage_path = "/home/peng/isacc learned/tutle/turtle.usd"
    default_output_root = "/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica"
    default_config_root = "/home/peng/isacc_slam/reference/OVO/data/working/configs/Replica"

    use_rviz = LaunchConfiguration("use_rviz")
    launch_isaac = LaunchConfiguration("launch_isaac")
    avia_params_file = LaunchConfiguration("avia_params_file")
    camera_params_file = LaunchConfiguration("camera_params_file")
    isaac_stage = LaunchConfiguration("isaac_stage")
    scene_name = LaunchConfiguration("scene_name")
    output_root = LaunchConfiguration("output_root")
    config_root = LaunchConfiguration("config_root")
    frame_stride = LaunchConfiguration("frame_stride")
    max_frames = LaunchConfiguration("max_frames")
    overwrite_scene = LaunchConfiguration("overwrite_scene")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="True"),
            DeclareLaunchArgument("launch_isaac", default_value="False"),
            DeclareLaunchArgument("avia_params_file", default_value=default_avia_config),
            DeclareLaunchArgument("camera_params_file", default_value=default_camera_config),
            DeclareLaunchArgument("isaac_stage", default_value=default_stage_path),
            DeclareLaunchArgument("scene_name", default_value="isaac_turtlebot3_orb2_source"),
            DeclareLaunchArgument("output_root", default_value=default_output_root),
            DeclareLaunchArgument("config_root", default_value=default_config_root),
            DeclareLaunchArgument("frame_stride", default_value="1"),
            DeclareLaunchArgument("max_frames", default_value="-1"),
            DeclareLaunchArgument("overwrite_scene", default_value="True"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    "use_rviz": use_rviz,
                    "launch_isaac": launch_isaac,
                    "avia_params_file": avia_params_file,
                    "camera_params_file": camera_params_file,
                    "isaac_stage": isaac_stage,
                    "scene_name": scene_name,
                    "output_root": output_root,
                    "config_root": config_root,
                    "frame_stride": frame_stride,
                    "max_frames": max_frames,
                    "overwrite_scene": overwrite_scene,
                    "use_keyframe_filter": "False",
                    "keyframe_min_translation_m": "0.0",
                    "keyframe_min_rotation_deg": "0.0",
                    "keyframe_min_frame_gap": "1",
                }.items(),
            ),
        ]
    )
