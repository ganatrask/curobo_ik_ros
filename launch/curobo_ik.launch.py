"""
Launch file for cuRobo IK service node.

Usage:
    ros2 launch curobo_ik_ros curobo_ik.launch.py \
        config_path:=/path/to/nero_curobo.yml \
        ee_link:=gripper_tip \
        namespace:=nero

This will create the service at: /nero/solve_ik
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
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
    ])
