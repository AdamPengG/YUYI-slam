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
        ],
    },
)
