#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    semantic_share = get_package_share_directory("onemap_semantic_mapper")
    fast_livo_share = get_package_share_directory("fast_livo")

    default_avia_config = os.path.join(fast_livo_share, "config", "avia_isaac.yaml")
    default_camera_config = os.path.join(fast_livo_share, "config", "camera_pinhole.yaml")

    use_rviz = LaunchConfiguration("use_rviz")
    launch_fast_livo = LaunchConfiguration("launch_fast_livo")
    scene_name = LaunchConfiguration("scene_name")
    avia_params = LaunchConfiguration("avia_params_file")
    camera_params = LaunchConfiguration("camera_params_file")
    output_root = LaunchConfiguration("output_root")
    config_root = LaunchConfiguration("config_root")
    run_root = LaunchConfiguration("run_root")
    inspection_output_root = LaunchConfiguration("inspection_output_root")
    inspection_session_name = LaunchConfiguration("inspection_session_name")
    max_frames = LaunchConfiguration("max_frames")
    class_set = LaunchConfiguration("class_set")
    topk_labels = LaunchConfiguration("topk_labels")
    abstain_min_score = LaunchConfiguration("abstain_min_score")
    abstain_min_margin = LaunchConfiguration("abstain_min_margin")

    fast_livo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(fast_livo_share, "launch", "mapping_isaac.launch.py")),
        condition=IfCondition(launch_fast_livo),
        launch_arguments={
            "use_rviz": use_rviz,
            "avia_params_file": avia_params,
            "camera_params_file": camera_params,
            "use_image_republish": "False",
        }.items(),
    )

    inspector = Node(
        package="onemap_semantic_mapper",
        executable="livo2_ovo_semantic_inspection_exporter",
        name="livo2_ovo_semantic_inspection_exporter",
        output="screen",
        parameters=[
            avia_params,
            camera_params,
            {
                "export.scene_name": scene_name,
                "export.output_root": output_root,
                "export.config_root": config_root,
                "export.run_root": run_root,
                "export.max_frames": max_frames,
                "export.overwrite_scene": True,
                "export.override_intrinsics_from_camera_info": True,
                "inspection.output_root": inspection_output_root,
                "inspection.session_name": inspection_session_name,
                "inspection.class_set": class_set,
                "inspection.topk_labels": topk_labels,
                "inspection.abstain_min_score": abstain_min_score,
                "inspection.abstain_min_margin": abstain_min_margin,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="True"),
            DeclareLaunchArgument("launch_fast_livo", default_value="False"),
            DeclareLaunchArgument("scene_name", default_value="semantic_inspection_scene"),
            DeclareLaunchArgument("avia_params_file", default_value=default_avia_config),
            DeclareLaunchArgument("camera_params_file", default_value=default_camera_config),
            DeclareLaunchArgument(
                "output_root",
                default_value="/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica",
            ),
            DeclareLaunchArgument(
                "config_root",
                default_value="/home/peng/isacc_slam/reference/OVO/data/working/configs/Replica",
            ),
            DeclareLaunchArgument(
                "run_root",
                default_value="/home/peng/isacc_slam/runs/ovo_pose_keyframes",
            ),
            DeclareLaunchArgument(
                "inspection_output_root",
                default_value="/home/peng/isacc_slam/runs/semantic_inspection",
            ),
            DeclareLaunchArgument("inspection_session_name", default_value=""),
            DeclareLaunchArgument("max_frames", default_value="-1"),
            DeclareLaunchArgument("class_set", default_value="full"),
            DeclareLaunchArgument("topk_labels", default_value="3"),
            DeclareLaunchArgument("abstain_min_score", default_value="0.15"),
            DeclareLaunchArgument("abstain_min_margin", default_value="0.05"),
            fast_livo_launch,
            inspector,
        ]
    )
