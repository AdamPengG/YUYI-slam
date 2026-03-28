#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    semantic_share = get_package_share_directory("onemap_semantic_mapper")
    fast_livo_share = get_package_share_directory("fast_livo")

    default_avia_config = os.path.join(fast_livo_share, "config", "avia_isaac.yaml")
    default_camera_config = os.path.join(fast_livo_share, "config", "camera_pinhole.yaml")
    default_output_root = "/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica"
    default_config_root = "/home/peng/isacc_slam/reference/OVO/data/working/configs/Replica"
    default_run_root = "/home/peng/isacc_slam/runs/ovo_pose_keyframes"
    default_ovo_root = "/home/peng/isacc_slam/reference/OVO"
    default_ovo_wrapper = "/home/peng/isacc_slam/scripts/run_ovo_eval_5090.sh"
    default_ovo_config = "data/working/configs/ovo_livo2_vanilla.yaml"
    default_output_artifact_root = "/home/peng/isacc_slam/reference/OVO/data/output/Replica"

    use_rviz = LaunchConfiguration("use_rviz")
    scene_name = LaunchConfiguration("scene_name")
    avia_params = LaunchConfiguration("avia_params_file")
    camera_params = LaunchConfiguration("camera_params_file")
    output_root = LaunchConfiguration("output_root")
    config_root = LaunchConfiguration("config_root")
    run_root = LaunchConfiguration("run_root")
    max_frames = LaunchConfiguration("max_frames")
    experiment_name = LaunchConfiguration("experiment_name")
    ovo_root = LaunchConfiguration("ovo_root")
    ovo_wrapper = LaunchConfiguration("ovo_wrapper")
    ovo_config = LaunchConfiguration("ovo_config")
    min_keyframes = LaunchConfiguration("min_keyframes")
    rerun_every = LaunchConfiguration("rerun_every_new_keyframes")
    render_every = LaunchConfiguration("render_overview_every_new_keyframes")
    artifact_path = LaunchConfiguration("artifact_path")
    clear_output_on_start = LaunchConfiguration("clear_output_on_start")
    resume_if_exists = LaunchConfiguration("resume_if_exists")

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
        executable="livo2_ovo_keyframe_exporter",
        name="livo2_ovo_keyframe_exporter",
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
                "export.override_intrinsics_from_camera_info": False,
            },
        ],
    )

    worker = Node(
        package="onemap_semantic_mapper",
        executable="ovo_async_worker",
        name="ovo_async_worker",
        output="screen",
        parameters=[
            {
                "scene_name": scene_name,
                "dataset_root": output_root,
                "ovo_root": ovo_root,
                "ovo_wrapper": ovo_wrapper,
                "ovo_config": ovo_config,
                "experiment_name": experiment_name,
                "min_keyframes": min_keyframes,
                "rerun_every_new_keyframes": rerun_every,
                "render_overview_every_new_keyframes": render_every,
                "clear_output_on_start": clear_output_on_start,
                "resume_if_exists": resume_if_exists,
            }
        ],
    )

    publisher = Node(
        package="onemap_semantic_mapper",
        executable="ovo_semantic_map_publisher",
        name="ovo_semantic_map_publisher",
        output="screen",
        respawn=True,
        respawn_delay=2.0,
        parameters=[
            {
                "artifact_path": artifact_path,
                "frame_id": "camera_init",
                "topic_name": "/ovo_semantic_rgbd_map",
                "marker_topic": "/ovo_instance_labels_rgbd",
            }
        ],
    )

    lidar_publisher = Node(
        package="onemap_semantic_mapper",
        executable="ovo_semantic_lidar_map_publisher",
        name="ovo_semantic_lidar_map_publisher",
        output="screen",
        respawn=True,
        respawn_delay=2.0,
        parameters=[
            {
                "artifact_path": artifact_path,
                "frame_id": "camera_init",
                "input_cloud_topic": "/cloud_registered",
                "topic_name": "/ovo_semantic_map",
                "marker_topic": "/ovo_instance_labels",
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="True"),
            DeclareLaunchArgument("scene_name", default_value="isaac_turtlebot3_livo2_online"),
            DeclareLaunchArgument("avia_params_file", default_value=default_avia_config),
            DeclareLaunchArgument("camera_params_file", default_value=default_camera_config),
            DeclareLaunchArgument("output_root", default_value=default_output_root),
            DeclareLaunchArgument("config_root", default_value=default_config_root),
            DeclareLaunchArgument("run_root", default_value=default_run_root),
            DeclareLaunchArgument("max_frames", default_value="-1"),
            DeclareLaunchArgument("experiment_name", default_value="isaac_livo2_online_vanilla"),
            DeclareLaunchArgument("ovo_root", default_value=default_ovo_root),
            DeclareLaunchArgument("ovo_wrapper", default_value=default_ovo_wrapper),
            DeclareLaunchArgument("ovo_config", default_value=default_ovo_config),
            DeclareLaunchArgument("min_keyframes", default_value="5"),
            DeclareLaunchArgument("rerun_every_new_keyframes", default_value="1"),
            DeclareLaunchArgument("render_overview_every_new_keyframes", default_value="5"),
            DeclareLaunchArgument("clear_output_on_start", default_value="True"),
            DeclareLaunchArgument("resume_if_exists", default_value="False"),
            DeclareLaunchArgument(
                "artifact_path",
                default_value=PathJoinSubstitution(
                    [
                        TextSubstitution(text=default_output_artifact_root),
                        experiment_name,
                        scene_name,
                        "semantic_snapshot.npz",
                    ]
                ),
            ),
            fast_livo_launch,
            exporter,
            worker,
            publisher,
            lidar_publisher,
        ]
    )
