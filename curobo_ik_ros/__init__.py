"""
curobo_ik_ros — GPU-accelerated IK via NVIDIA cuRobo, with ROS 2 service layer.

Library usage (no ROS needed):
    from curobo_ik_ros import CuRoboIKSolver

    solver = CuRoboIKSolver.from_config("nero_curobo.yml", ee_link="gripper_tip")
    q, ok = solver.ik(target_pose_4x4)

ROS 2 service:
    ros2 launch curobo_ik_ros curobo_ik.launch.py config_path:=nero_curobo.yml
"""

from curobo_ik_ros.solver import CuRoboIKSolver

__all__ = ["CuRoboIKSolver"]
