"""
Launch file for cuRobo IK service node.

Usage:
    ros2 launch curobo_ik_ros curobo_ik.launch.py \
        config_path:=/path/to/nero_curobo.yml \
        ee_link:=gripper_tip \
        namespace:=nero

This will create the service at: /nero/solve_ik
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Detect conda env path for curobo/torch packages
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        # Default fallback: try to find the 'curobo' conda env
        conda_base = os.path.expanduser("~/miniconda3")
        candidate = os.path.join(conda_base, "envs/curobo/lib/python3.12/site-packages")
        if os.path.isdir(candidate):
            conda_prefix = os.path.join(conda_base, "envs/curobo")

    conda_site_packages = ""
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "lib/python3.12/site-packages")
        if os.path.isdir(candidate):
            conda_site_packages = candidate

    # Find editable-installed curobo source path (pip install -e)
    curobo_src = ""
    if conda_site_packages:
        pth_file = os.path.join(conda_site_packages, "__editable__.nvidia_curobo-0.0.0.pth")
        if os.path.isfile(pth_file):
            # Editable install: .pth hooks don't work via PYTHONPATH,
            # so find the actual source directory from the finder module
            finder = os.path.join(conda_site_packages, "__editable___nvidia_curobo_0_0_0_finder.py")
            if os.path.isfile(finder):
                with open(finder) as f:
                    for line in f:
                        if "MAPPING" in line and "{" in line:
                            # Extract path from MAPPING = {'curobo': '/path/to/curobo/curobo'}
                            import ast
                            try:
                                mapping_str = line.split("=", 1)[1].strip()
                                mapping = ast.literal_eval(mapping_str)
                                if isinstance(mapping, dict) and "curobo" in mapping:
                                    curobo_src = os.path.dirname(mapping["curobo"])
                            except Exception:
                                pass
                            break
        if not curobo_src:
            # Fallback: check common location
            fallback = os.path.expanduser("~/projects/cc/curobo")
            if os.path.isdir(os.path.join(fallback, "curobo")):
                curobo_src = fallback

    # Build PYTHONPATH with conda packages prepended
    pythonpath_parts = filter(None, [
        curobo_src,
        conda_site_packages,
        os.environ.get("PYTHONPATH", ""),
    ])
    new_pythonpath = ":".join(pythonpath_parts)

    # The ROS node needs the conda Python interpreter (not system Python) so that
    # cuda.pathfinder and editable installs work correctly. We use prefix to wrap
    # the command with the conda Python and inject PYTHONPATH for ROS packages.
    conda_python = ""
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin/python3")
        if os.path.isfile(candidate):
            conda_python = candidate

    prefix_cmd = ""
    if conda_python:
        # Re-exec the script with conda python, keeping ROS PYTHONPATH
        prefix_cmd = (
            f"bash -c '"
            f"export PYTHONPATH=\"$PYTHONPATH\"; "
            f'exec {conda_python} "$@"'
            f"' --"
        )

    actions = [
        # Required
        DeclareLaunchArgument(
            "config_path",
            description="Absolute path to cuRobo YAML config file",
        ),

        # Optional overrides
        DeclareLaunchArgument(
            "ee_link",
            default_value="",
            description="End-effector link name (empty = use YAML default)",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Node namespace (service will be /<namespace>/solve_ik)",
        ),
        DeclareLaunchArgument(
            "num_seeds",
            default_value="20",
            description="Number of IK solver seeds (more = higher solve rate, slower)",
        ),
        DeclareLaunchArgument(
            "self_collision_check",
            default_value="true",
            description="Enable self-collision checking in IK",
        ),
        DeclareLaunchArgument(
            "warmup",
            default_value="true",
            description="Run CUDA warmup at startup (adds 2-5s to launch)",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda:0",
            description="CUDA device (cuda:0, cuda:1, etc.)",
        ),
        DeclareLaunchArgument(
            "rotation_threshold",
            default_value="0.05",
            description="IK rotation convergence threshold (rad)",
        ),
        DeclareLaunchArgument(
            "position_threshold",
            default_value="0.005",
            description="IK position convergence threshold (meters)",
        ),

        # Node
        Node(
            package="curobo_ik_ros",
            executable="ik_service_node.py",
            name="curobo_ik_server",
            namespace=LaunchConfiguration("namespace"),
            prefix=prefix_cmd,
            parameters=[{
                "config_path": LaunchConfiguration("config_path"),
                "ee_link": LaunchConfiguration("ee_link"),
                "num_seeds": LaunchConfiguration("num_seeds"),
                "self_collision_check": LaunchConfiguration("self_collision_check"),
                "warmup": LaunchConfiguration("warmup"),
                "device": LaunchConfiguration("device"),
                "rotation_threshold": LaunchConfiguration("rotation_threshold"),
                "position_threshold": LaunchConfiguration("position_threshold"),
            }],
            output="screen",
        ),
    ]

    return LaunchDescription(actions)
