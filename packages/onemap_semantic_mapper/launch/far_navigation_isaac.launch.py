import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    semantic_share = get_package_share_directory("onemap_semantic_mapper")
    fast_livo_share = get_package_share_directory("fast_livo")
    default_config = os.path.join(semantic_share, "config", "far_warehouse.yaml")

    launch_fast_livo_arg = DeclareLaunchArgument(
        "launch_fast_livo",
        default_value="true",
        description="Launch FAST-LIVO2 together with FAR planner.",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz through FAST-LIVO2 launch.",
    )
    rviz_config_file_arg = DeclareLaunchArgument(
        "rviz_config_file",
        default_value=os.path.join(semantic_share, "config", "semantic_fast_livo2.rviz"),
        description="RViz config file passed through to FAST-LIVO2 launch.",
    )
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Absolute path to a FAR planner config yaml.",
    )
    terrain_ground_percentile_arg = DeclareLaunchArgument(
        "terrain_ground_percentile",
        default_value="2.0",
        description="Percentile used to estimate floor height from /cloud_registered.",
    )
    terrain_obstacle_height_m_arg = DeclareLaunchArgument(
        "terrain_obstacle_height_m",
        default_value="0.18",
        description="Height above floor considered obstacle in terrain maps.",
    )
    terrain_local_voxel_size_m_arg = DeclareLaunchArgument(
        "terrain_local_voxel_size_m",
        default_value="0.10",
        description="Voxel size for local /terrain_map publication.",
    )
    terrain_ext_voxel_size_m_arg = DeclareLaunchArgument(
        "terrain_ext_voxel_size_m",
        default_value="0.18",
        description="Voxel size for accumulated /terrain_map_ext publication.",
    )

    fast_livo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_livo_share, "launch", "mapping_isaac.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("launch_fast_livo")),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
            "rviz_config_file": LaunchConfiguration("rviz_config_file"),
        }.items(),
    )

    bridge_node = Node(
        package="onemap_semantic_mapper",
        executable="fast_livo_far_bridge",
        name="fast_livo_far_bridge",
        output="screen",
        parameters=[
            {
                "terrain_ground_percentile": LaunchConfiguration("terrain_ground_percentile"),
                "terrain_obstacle_height_m": LaunchConfiguration("terrain_obstacle_height_m"),
                "terrain_local_voxel_size_m": LaunchConfiguration("terrain_local_voxel_size_m"),
                "terrain_ext_voxel_size_m": LaunchConfiguration("terrain_ext_voxel_size_m"),
            }
        ],
    )

    far_node = Node(
        package="far_planner",
        executable="far_planner",
        name="far_planner",
        output="screen",
        parameters=[LaunchConfiguration("config_file")],
        remappings=[
            ("/odom_world", "/state_estimation"),
            ("/terrain_cloud", "/terrain_map_ext"),
            ("/scan_cloud", "/terrain_map"),
            ("/terrain_local_cloud", "/registered_scan"),
        ],
    )

    follower_node = Node(
        package="onemap_semantic_mapper",
        executable="far_waypoint_follower",
        name="far_waypoint_follower",
        output="screen",
    )

    auto_goal_node = Node(
        package="onemap_semantic_mapper",
        executable="far_auto_goal_manager",
        name="far_auto_goal_manager",
        output="screen",
    )

    return LaunchDescription(
        [
            launch_fast_livo_arg,
            use_rviz_arg,
            rviz_config_file_arg,
            config_file_arg,
            terrain_ground_percentile_arg,
            terrain_obstacle_height_m_arg,
            terrain_local_voxel_size_m_arg,
            terrain_ext_voxel_size_m_arg,
            fast_livo_launch,
            bridge_node,
            far_node,
            follower_node,
            auto_goal_node,
        ]
    )
