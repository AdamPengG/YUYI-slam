from glob import glob
import os

from setuptools import find_packages, setup


package_name = "onemap_semantic_mapper"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "scripts"), glob("scripts/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="peng",
    maintainer_email="peng@example.com",
    description="Semantic point-cloud mapping node for FAST-LIVO2 + Isaac Sim.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_mapper = onemap_semantic_mapper.semantic_mapper_node:main",
            "ovo_dataset_exporter = onemap_semantic_mapper.ovo_dataset_exporter:main",
            "livo2_ovo_keyframe_exporter = onemap_semantic_mapper.livo2_ovo_keyframe_exporter:main",
            "livo2_ovo_semantic_inspection_exporter = onemap_semantic_mapper.semantic_inspection_exporter:main",
            "fast_livo_far_bridge = onemap_semantic_mapper.fast_livo_far_bridge:main",
            "far_waypoint_follower = onemap_semantic_mapper.far_waypoint_follower:main",
            "far_auto_goal_manager = onemap_semantic_mapper.far_auto_goal_manager:main",
            "ovo_async_worker = onemap_semantic_mapper.ovo_async_worker:main",
            "ovo_async_worker_legacy_yolo = onemap_semantic_mapper.ovo_async_worker_legacy_yolo:main",
            "ovo_semantic_map_publisher = onemap_semantic_mapper.ovo_semantic_map_publisher:main",
            "ovo_semantic_lidar_map_publisher = onemap_semantic_mapper.ovo_semantic_lidar_map_publisher:main",
            "livo2_ovo_final_consolidation = onemap_semantic_mapper.final_consolidation:main",
            "livo2_semantic_optimizer = onemap_semantic_mapper.semantic_optimizer:main",
            "livo2_dense_semantic_exporter = onemap_semantic_mapper.exporter:main",
        ],
    },
)
