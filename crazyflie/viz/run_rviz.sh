#!/usr/bin/env bash
# RViz view for fly_policy.py --viz: the ROS 2 bridge (system python, Humble) + rviz2 with the saved layout.
# Runs in a clean environment so an active conda env cannot break rclpy / rviz2.
cd "$(dirname "${BASH_SOURCE[0]}")"
exec env -i HOME="$HOME" USER="$USER" DISPLAY="${DISPLAY:-:0}" XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" PATH=/usr/bin:/bin bash -c '
    source /opt/ros/humble/setup.bash
    /usr/bin/python3 rviz_bridge.py "$@" &
    bridge=$!
    trap "kill $bridge 2>/dev/null" EXIT
    rviz2 -d cf_policy.rviz
' _ "$@"
