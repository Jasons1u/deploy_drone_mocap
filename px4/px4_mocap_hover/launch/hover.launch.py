"""Mocap bridge + hover policy.

    ros2 launch px4_mocap_hover hover.launch.py server_ip:=192.168.1.10
    ros2 launch px4_mocap_hover hover.launch.py with_policy:=false server_ip:=...   # ground station
    ros2 launch px4_mocap_hover hover.launch.py with_bridge:=false                  # onboard
    ros2 launch px4_mocap_hover hover.launch.py state_source:=mocap

Tuning lives in config/hover.yaml and config/mocap.yaml. The arguments below override the few
values that change per session; left empty, the YAML value stands.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

TRUE = ("1", "true", "yes")


def _nodes(context):
    share = get_package_share_directory("px4_mocap_hover")
    arg = lambda name: context.launch_configurations[name]  # noqa: E731

    def overrides(*names):
        return {n: arg(n) for n in names if arg(n)}

    nodes = []
    if arg("with_bridge").lower() in TRUE:
        nodes.append(Node(
            package="px4_mocap_hover", executable="mocap_bridge", name="mocap_bridge",
            output="screen", emulate_tty=True,
            parameters=[os.path.join(share, "config", "mocap.yaml"), overrides("server_ip")]))
    if arg("with_policy").lower() in TRUE:
        nodes.append(Node(
            package="px4_mocap_hover", executable="hover_node", name="hover_node",
            output="screen", emulate_tty=True,
            parameters=[os.path.join(share, "config", "hover.yaml"), overrides("state_source")]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("with_bridge", default_value="true"),
        DeclareLaunchArgument("with_policy", default_value="true"),
        DeclareLaunchArgument("server_ip", default_value="", description="Motive PC IP"),
        DeclareLaunchArgument("state_source", default_value="", description="px4 | mocap"),
        OpaqueFunction(function=_nodes),
    ])
