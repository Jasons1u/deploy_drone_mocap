import os
from glob import glob

from setuptools import setup

package_name = "px4_mocap_hover"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "policy"), glob("policy/*.onnx")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jason",
    maintainer_email="jasonsiu@caltech.edu",
    description="RL hover policy on PX4 (offboard direct-actuator) with OptiTrack as perception.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hover_node = px4_mocap_hover.hover_node:main",
            "mocap_bridge = px4_mocap_hover.mocap_bridge:main",
        ],
    },
)
