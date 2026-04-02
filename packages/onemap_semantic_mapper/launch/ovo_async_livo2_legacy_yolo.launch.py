#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    fast_livo_share = get_package_share_directory("fast_livo")

    default_avia_config = os.path.join(fast_livo_share, "config", "avia_isaac.yaml")
    default_camera_config = os.path.join(fast_livo_share, "config", "camera_pinhole.yaml")
    default_output_root = "/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica"
    default_config_root = "/home/peng/isacc_slam/reference/OVO/data/working/configs/Replica"
    default_run_root = "/home/peng/isacc_slam/runs/ovo_pose_keyframes"
    default_ovo_root = "/home/peng/isacc_slam/reference/OVO"
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
    force_every_synced_frame = LaunchConfiguration("force_every_synced_frame")
    selector_min_time_gap_sec = LaunchConfiguration("selector_min_time_gap_sec")
    selector_max_time_gap_sec = LaunchConfiguration("selector_max_time_gap_sec")
    selector_translation_thresh_m = LaunchConfiguration("selector_translation_thresh_m")
    selector_rotation_thresh_deg = LaunchConfiguration("selector_rotation_thresh_deg")
    selector_coverage_novelty_thresh = LaunchConfiguration("selector_coverage_novelty_thresh")
    selector_min_translation_for_rotation_trigger_m = LaunchConfiguration(
        "selector_min_translation_for_rotation_trigger_m"
    )
    selector_min_translation_for_time_trigger_m = LaunchConfiguration(
        "selector_min_translation_for_time_trigger_m"
    )
    selector_semantic_trigger_min_translation_m = LaunchConfiguration(
        "selector_semantic_trigger_min_translation_m"
    )
    selector_semantic_trigger_cooldown_sec = LaunchConfiguration("selector_semantic_trigger_cooldown_sec")
    camera_pose_topic = LaunchConfiguration("camera_pose_topic")
    require_direct_camera_pose = LaunchConfiguration("require_direct_camera_pose")
    require_exact_direct_camera_pose = LaunchConfiguration("require_exact_direct_camera_pose")
    direct_camera_pose_exact_tolerance_sec = LaunchConfiguration("direct_camera_pose_exact_tolerance_sec")
    direct_camera_pose_apply_isaac_optical_fix = LaunchConfiguration("direct_camera_pose_apply_isaac_optical_fix")
    align_direct_camera_pose_to_odom_world = LaunchConfiguration("align_direct_camera_pose_to_odom_world")
    align_direct_camera_pose_to_odom_max_dt_sec = LaunchConfiguration("align_direct_camera_pose_to_odom_max_dt_sec")
    require_exact_rgbd_sync = LaunchConfiguration("require_exact_rgbd_sync")
    use_rgbd_local_cloud = LaunchConfiguration("use_rgbd_local_cloud")
    rgbd_stride = LaunchConfiguration("rgbd_stride")
    rgbd_min_depth_m = LaunchConfiguration("rgbd_min_depth_m")
    rgbd_exact_tolerance_sec = LaunchConfiguration("rgbd_exact_tolerance_sec")
    dynamic_semantic_persistence_enabled = LaunchConfiguration("dynamic_semantic_persistence_enabled")
    dynamic_semantic_target_labels = LaunchConfiguration("dynamic_semantic_target_labels")
    dynamic_semantic_min_detections = LaunchConfiguration("dynamic_semantic_min_detections")
    dynamic_semantic_min_score = LaunchConfiguration("dynamic_semantic_min_score")
    dynamic_semantic_keepalive_sec = LaunchConfiguration("dynamic_semantic_keepalive_sec")
    dynamic_semantic_fallback_sec = LaunchConfiguration("dynamic_semantic_fallback_sec")
    experiment_name = LaunchConfiguration("experiment_name")
    ovo_root = LaunchConfiguration("ovo_root")
    ovo_config = LaunchConfiguration("ovo_config")
    min_keyframes = LaunchConfiguration("min_keyframes")
    rerun_every = LaunchConfiguration("rerun_every_new_keyframes")
    min_observation_points = LaunchConfiguration("min_observation_points")
    min_mask_area = LaunchConfiguration("min_mask_area")
    snapshot_voxel_size_m = LaunchConfiguration("snapshot_voxel_size_m")
    snapshot_mode = LaunchConfiguration("snapshot_mode")
    support_expansion_radius_m = LaunchConfiguration("support_expansion_radius_m")
    near_ground_filter_height_m = LaunchConfiguration("near_ground_filter_height_m")
    near_ground_floor_percentile = LaunchConfiguration("near_ground_floor_percentile")
    assoc_score_min = LaunchConfiguration("assoc_score_min")
    reproj_iou_min = LaunchConfiguration("reproj_iou_min")
    surface_hit_min = LaunchConfiguration("surface_hit_min")
    reproj_dilate_px = LaunchConfiguration("reproj_dilate_px")
    track_pending_hits = LaunchConfiguration("track_pending_hits")
    track_dormant_after_sec = LaunchConfiguration("track_dormant_after_sec")
    track_delete_after_sec = LaunchConfiguration("track_delete_after_sec")
    new_track_min_points = LaunchConfiguration("new_track_min_points")
    fuse_voxel_size_m = LaunchConfiguration("fuse_voxel_size_m")
    support_expansion_max_points = LaunchConfiguration("support_expansion_max_points")
    use_semantic_subset_projection = LaunchConfiguration("use_semantic_subset_projection")
    artifact_path = LaunchConfiguration("artifact_path")
    clear_output_on_start = LaunchConfiguration("clear_output_on_start")
    resume_if_exists = LaunchConfiguration("resume_if_exists")
    class_set = LaunchConfiguration("class_set")
    artifact_dir = PathJoinSubstitution(
        [
            TextSubstitution(text=default_output_artifact_root),
            experiment_name,
            scene_name,
        ]
    )
    semantic_status_path = PathJoinSubstitution([artifact_dir, "online_status.json"])
    semantic_object_memory_path = PathJoinSubstitution([artifact_dir, "object_memory.json"])

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
                "export.camera_pose_topic": camera_pose_topic,
                "export.require_direct_camera_pose": require_direct_camera_pose,
                "export.require_exact_direct_camera_pose": require_exact_direct_camera_pose,
                "export.direct_camera_pose_exact_tolerance_sec": direct_camera_pose_exact_tolerance_sec,
                "export.direct_camera_pose_apply_isaac_optical_fix": direct_camera_pose_apply_isaac_optical_fix,
                "export.align_direct_camera_pose_to_odom_world": align_direct_camera_pose_to_odom_world,
                "export.align_direct_camera_pose_to_odom_max_dt_sec": align_direct_camera_pose_to_odom_max_dt_sec,
                "export.require_exact_rgbd_sync": require_exact_rgbd_sync,
                "export.use_rgbd_local_cloud": use_rgbd_local_cloud,
                "export.rgbd_stride": rgbd_stride,
                "export.rgbd_min_depth_m": rgbd_min_depth_m,
                "export.rgbd_exact_tolerance_sec": rgbd_exact_tolerance_sec,
                "export.dynamic_semantic_persistence_enabled": dynamic_semantic_persistence_enabled,
                "export.dynamic_semantic_target_labels": dynamic_semantic_target_labels,
                "export.dynamic_semantic_min_detections": dynamic_semantic_min_detections,
                "export.dynamic_semantic_min_score": dynamic_semantic_min_score,
                "export.dynamic_semantic_keepalive_sec": dynamic_semantic_keepalive_sec,
                "export.dynamic_semantic_fallback_sec": dynamic_semantic_fallback_sec,
                "selector.force_every_synced_frame": force_every_synced_frame,
                "selector.min_time_gap_sec": selector_min_time_gap_sec,
                "selector.max_time_gap_sec": selector_max_time_gap_sec,
                "selector.translation_thresh_m": selector_translation_thresh_m,
                "selector.rotation_thresh_deg": selector_rotation_thresh_deg,
                "selector.coverage_novelty_thresh": selector_coverage_novelty_thresh,
                "selector.min_translation_for_rotation_trigger_m": selector_min_translation_for_rotation_trigger_m,
                "selector.min_translation_for_time_trigger_m": selector_min_translation_for_time_trigger_m,
                "selector.semantic_trigger_min_translation_m": selector_semantic_trigger_min_translation_m,
                "selector.semantic_trigger_cooldown_sec": selector_semantic_trigger_cooldown_sec,
                "selector.semantic_status_path": semantic_status_path,
                "selector.semantic_object_memory_path": semantic_object_memory_path,
            },
        ],
    )

    worker = Node(
        package="onemap_semantic_mapper",
        executable="ovo_async_worker_legacy_yolo",
        name="ovo_async_worker_legacy_yolo",
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
                "min_observation_points": min_observation_points,
                "min_mask_area": min_mask_area,
                "snapshot_voxel_size_m": snapshot_voxel_size_m,
                "snapshot_mode": snapshot_mode,
                "support_expansion_radius_m": support_expansion_radius_m,
                "near_ground_filter_height_m": near_ground_filter_height_m,
                "near_ground_floor_percentile": near_ground_floor_percentile,
                "assoc_score_min": assoc_score_min,
                "reproj_iou_min": reproj_iou_min,
                "surface_hit_min": surface_hit_min,
                "reproj_dilate_px": reproj_dilate_px,
                "track_pending_hits": track_pending_hits,
                "track_dormant_after_sec": track_dormant_after_sec,
                "track_delete_after_sec": track_delete_after_sec,
                "new_track_min_points": new_track_min_points,
                "fuse_voxel_size_m": fuse_voxel_size_m,
                "support_expansion_max_points": support_expansion_max_points,
                "use_semantic_subset_projection": use_semantic_subset_projection,
                "artifact_path": artifact_path,
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
                "topic_name": "/ovo_semantic_map",
                "marker_topic": "/ovo_instance_labels",
                "semantic_only": True,
                "min_instance_points_for_label": 50,
                "min_instance_views_for_label": 1,
                "load_existing_artifact_on_start": resume_if_exists,
                "max_points": 1000000,
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
                "semantic_only": True,
                "min_instance_points_for_label": 50,
                "min_instance_views_for_label": 1,
                "load_existing_artifact_on_start": resume_if_exists,
            }
        ],
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
            DeclareLaunchArgument("force_every_synced_frame", default_value="False"),
            DeclareLaunchArgument("selector_min_time_gap_sec", default_value="0.15"),
            DeclareLaunchArgument("selector_max_time_gap_sec", default_value="1.20"),
            DeclareLaunchArgument("selector_translation_thresh_m", default_value="0.08"),
            DeclareLaunchArgument("selector_rotation_thresh_deg", default_value="8.0"),
            DeclareLaunchArgument("selector_coverage_novelty_thresh", default_value="0.12"),
            DeclareLaunchArgument("selector_min_translation_for_rotation_trigger_m", default_value="0.02"),
            DeclareLaunchArgument("selector_min_translation_for_time_trigger_m", default_value="0.03"),
            DeclareLaunchArgument("selector_semantic_trigger_min_translation_m", default_value="0.02"),
            DeclareLaunchArgument("selector_semantic_trigger_cooldown_sec", default_value="0.8"),
            DeclareLaunchArgument("camera_pose_topic", default_value="/camera_pose_gt_at_image"),
            DeclareLaunchArgument("require_direct_camera_pose", default_value="True"),
            DeclareLaunchArgument("require_exact_direct_camera_pose", default_value="True"),
            DeclareLaunchArgument("direct_camera_pose_exact_tolerance_sec", default_value="4.0e-2"),
            DeclareLaunchArgument("direct_camera_pose_apply_isaac_optical_fix", default_value="True"),
            DeclareLaunchArgument("align_direct_camera_pose_to_odom_world", default_value="True"),
            DeclareLaunchArgument("align_direct_camera_pose_to_odom_max_dt_sec", default_value="5.0e-2"),
            DeclareLaunchArgument("require_exact_rgbd_sync", default_value="False"),
            DeclareLaunchArgument("use_rgbd_local_cloud", default_value="True"),
            DeclareLaunchArgument("rgbd_stride", default_value="4"),
            DeclareLaunchArgument("rgbd_min_depth_m", default_value="0.05"),
            DeclareLaunchArgument("rgbd_exact_tolerance_sec", default_value="5.0e-2"),
            DeclareLaunchArgument("dynamic_semantic_persistence_enabled", default_value="False"),
            DeclareLaunchArgument("dynamic_semantic_target_labels", default_value=""),
            DeclareLaunchArgument("dynamic_semantic_min_detections", default_value="1"),
            DeclareLaunchArgument("dynamic_semantic_min_score", default_value="0.10"),
            DeclareLaunchArgument("dynamic_semantic_keepalive_sec", default_value="1.5"),
            DeclareLaunchArgument("dynamic_semantic_fallback_sec", default_value="0.0"),
            DeclareLaunchArgument("experiment_name", default_value="isaac_livo2_online_legacy_yolo"),
            DeclareLaunchArgument("ovo_root", default_value=default_ovo_root),
            DeclareLaunchArgument("ovo_config", default_value=default_ovo_config),
            DeclareLaunchArgument("class_set", default_value="full"),
            DeclareLaunchArgument("min_keyframes", default_value="1"),
            DeclareLaunchArgument("rerun_every_new_keyframes", default_value="1"),
            DeclareLaunchArgument("min_observation_points", default_value="4"),
            DeclareLaunchArgument("min_mask_area", default_value="24"),
            DeclareLaunchArgument("snapshot_voxel_size_m", default_value="0.0"),
            DeclareLaunchArgument("snapshot_mode", default_value="registered_frame"),
            DeclareLaunchArgument("support_expansion_radius_m", default_value="0.20"),
            DeclareLaunchArgument("near_ground_filter_height_m", default_value="0.08"),
            DeclareLaunchArgument("near_ground_floor_percentile", default_value="1.0"),
            DeclareLaunchArgument("assoc_score_min", default_value="0.42"),
            DeclareLaunchArgument("reproj_iou_min", default_value="0.08"),
            DeclareLaunchArgument("surface_hit_min", default_value="0.10"),
            DeclareLaunchArgument("reproj_dilate_px", default_value="3"),
            DeclareLaunchArgument("track_pending_hits", default_value="1"),
            DeclareLaunchArgument("track_dormant_after_sec", default_value="2.0"),
            DeclareLaunchArgument("track_delete_after_sec", default_value="30.0"),
            DeclareLaunchArgument("new_track_min_points", default_value="6"),
            DeclareLaunchArgument("fuse_voxel_size_m", default_value="0.03"),
            DeclareLaunchArgument("support_expansion_max_points", default_value="12000"),
            DeclareLaunchArgument("use_semantic_subset_projection", default_value="False"),
            DeclareLaunchArgument("clear_output_on_start", default_value="True"),
            DeclareLaunchArgument("resume_if_exists", default_value="False"),
            DeclareLaunchArgument(
                "artifact_path",
                default_value=PathJoinSubstitution(
                    [
                        artifact_dir,
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
