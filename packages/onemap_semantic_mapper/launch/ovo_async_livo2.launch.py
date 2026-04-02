#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
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
    default_ovo_config = "data/working/configs/ovo_livo2_yoloe26x.yaml"
    default_output_artifact_root = "/home/peng/isacc_slam/reference/OVO/data/output/Replica"

    use_rviz = LaunchConfiguration("use_rviz")
    launch_fast_livo = LaunchConfiguration("launch_fast_livo")
    launch_exporter = LaunchConfiguration("launch_exporter")
    scene_name = LaunchConfiguration("scene_name")
    avia_params = LaunchConfiguration("avia_params_file")
    camera_params = LaunchConfiguration("camera_params_file")
    output_root = LaunchConfiguration("output_root")
    config_root = LaunchConfiguration("config_root")
    run_root = LaunchConfiguration("run_root")
    export_dir_override = LaunchConfiguration("export_dir_override")
    max_frames = LaunchConfiguration("max_frames")
    experiment_name = LaunchConfiguration("experiment_name")
    ovo_root = LaunchConfiguration("ovo_root")
    ovo_wrapper = LaunchConfiguration("ovo_wrapper")
    ovo_config = LaunchConfiguration("ovo_config")
    min_keyframes = LaunchConfiguration("min_keyframes")
    rerun_every = LaunchConfiguration("rerun_every_new_keyframes")
    class_set = LaunchConfiguration("class_set")
    render_every = LaunchConfiguration("render_overview_every_new_keyframes")
    artifact_path = LaunchConfiguration("artifact_path")
    clear_output_on_start = LaunchConfiguration("clear_output_on_start")
    resume_if_exists = LaunchConfiguration("resume_if_exists")
    geometry_voxel_size_m = LaunchConfiguration("geometry_voxel_size_m")

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

    exporter = Node(
        package="onemap_semantic_mapper",
        executable="livo2_ovo_keyframe_exporter",
        name="livo2_ovo_keyframe_exporter",
        output="screen",
        condition=IfCondition(launch_exporter),
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
                "selector.semantic_status_path": PathJoinSubstitution(
                    [
                        TextSubstitution(text=default_output_artifact_root),
                        experiment_name,
                        scene_name,
                        "online_status.json",
                    ]
                ),
                "selector.semantic_object_memory_path": PathJoinSubstitution(
                    [
                        TextSubstitution(text=default_output_artifact_root),
                        experiment_name,
                        scene_name,
                        "object_memory.json",
                    ]
                ),
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
                "run_root": run_root,
                "export_dir_override": export_dir_override,
                "ovo_root": ovo_root,
                "ovo_config": ovo_config,
                "experiment_name": experiment_name,
                "class_set": class_set,
                "min_keyframes": min_keyframes,
                "rerun_every_new_keyframes": rerun_every,
                "artifact_path": artifact_path,
                "clear_output_on_start": clear_output_on_start,
                "resume_if_exists": resume_if_exists,
                "geometry_voxel_size_m": geometry_voxel_size_m,
                "direct_semantic_only": True,
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
                "topic_name": "/ovo_semantic_map",
                "marker_topic": "/ovo_instance_labels",
                "semantic_only": True,
                "min_instance_points_for_label": 20,
                "min_instance_views_for_label": 1,
                "load_existing_artifact_on_start": resume_if_exists,
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
                "topic_name": "/ovo_semantic_lidar_map",
                "marker_topic": "/ovo_instance_labels_lidar",
                "match_radius_m": 0.5,
                "semantic_only": True,
                "min_instance_points_for_label": 20,
                "min_instance_views_for_label": 1,
                "load_existing_artifact_on_start": resume_if_exists,
            }
        ],
    )

    cleanup_old_ovo = ExecuteProcess(
        cmd=[
            "python3",
            "-c",
            (
                "import os, signal, subprocess; "
                "targets=('ovo_async_worker','ovo_semantic_map_publisher',"
                "'ovo_semantic_lidar_map_publisher','run_semantic_observer_online.py'); "
                "me=os.getpid(); "
                "out=subprocess.check_output(['ps','-eo','pid=,args='], text=True); "
                "lines=[line.strip() for line in out.splitlines() if line.strip()]; "
                "[(lambda pid: os.kill(pid, signal.SIGTERM) if pid != me else None)(int(line.split(None,1)[0])) "
                "for line in lines if any(t in line for t in targets)]"
            ),
        ],
        shell=False,
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="True"),
            DeclareLaunchArgument("launch_fast_livo", default_value="False"),
            DeclareLaunchArgument("launch_exporter", default_value="True"),
            DeclareLaunchArgument("scene_name", default_value="isaac_turtlebot3_livo2_online"),
            DeclareLaunchArgument("avia_params_file", default_value=default_avia_config),
            DeclareLaunchArgument("camera_params_file", default_value=default_camera_config),
            DeclareLaunchArgument("output_root", default_value=default_output_root),
            DeclareLaunchArgument("config_root", default_value=default_config_root),
            DeclareLaunchArgument("run_root", default_value=default_run_root),
            DeclareLaunchArgument("export_dir_override", default_value=""),
            DeclareLaunchArgument("max_frames", default_value="-1"),
            DeclareLaunchArgument("experiment_name", default_value="isaac_livo2_online_vanilla"),
            DeclareLaunchArgument("ovo_root", default_value=default_ovo_root),
            DeclareLaunchArgument("ovo_wrapper", default_value=default_ovo_wrapper),
            DeclareLaunchArgument("ovo_config", default_value=default_ovo_config),
            DeclareLaunchArgument("class_set", default_value="full"),
            DeclareLaunchArgument("min_keyframes", default_value="5"),
            DeclareLaunchArgument("rerun_every_new_keyframes", default_value="1"),
            DeclareLaunchArgument("render_overview_every_new_keyframes", default_value="5"),
            DeclareLaunchArgument("clear_output_on_start", default_value="True"),
            DeclareLaunchArgument("resume_if_exists", default_value="False"),
            DeclareLaunchArgument("geometry_voxel_size_m", default_value="0.02"),
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
            cleanup_old_ovo,
            TimerAction(
                period=1.0,
                actions=[
                    fast_livo_launch,
                    exporter,
                    publisher,
                    lidar_publisher,
                ],
            ),
            TimerAction(
                period=2.0,
                actions=[
                    worker,
                ],
            ),
        ]
    )
