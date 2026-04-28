#!/usr/bin/env python3
"""
ROS 2 service node wrapping CuRoboIKSolver.

Exposes a /solve_ik service (curobo_ik_ros/srv/SolveIK) that accepts a
target EE pose and returns joint angles computed by cuRobo on the GPU.

Launch:
    ros2 launch curobo_ik_ros curobo_ik.launch.py \
        config_path:=/path/to/nero_curobo.yml ee_link:=gripper_tip

CLI test:
    ros2 service call /solve_ik curobo_ik_ros/srv/SolveIK \
        "{target_pose: {position: {x: -0.2, y: 0.0, z: 0.4}, \
          orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}}, position_only: false}"
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node

from curobo_ik_ros.solver import CuRoboIKSolver
from curobo_ik_ros.conversions import ros_pose_to_4x4
from curobo_ik_ros.srv import SolveIK


class CuRoboIKServiceNode(Node):
    """ROS 2 service server for cuRobo GPU inverse kinematics."""

    def __init__(self):
        super().__init__("curobo_ik_server")

        # Declare parameters
        self.declare_parameter("config_path", "")
        self.declare_parameter("ee_link", "")
        self.declare_parameter("num_seeds", 20)
        self.declare_parameter("self_collision_check", True)
        self.declare_parameter("warmup", True)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("rotation_threshold", 0.05)
        self.declare_parameter("position_threshold", 0.005)

        # Read parameters
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        if not config_path:
            self.get_logger().fatal("Parameter 'config_path' is required")
            raise SystemExit(1)

        ee_link = self.get_parameter("ee_link").get_parameter_value().string_value or None
        num_seeds = self.get_parameter("num_seeds").get_parameter_value().integer_value
        self_collision = self.get_parameter("self_collision_check").get_parameter_value().bool_value
        do_warmup = self.get_parameter("warmup").get_parameter_value().bool_value
        device = self.get_parameter("device").get_parameter_value().string_value
        rot_thresh = self.get_parameter("rotation_threshold").get_parameter_value().double_value
        pos_thresh = self.get_parameter("position_threshold").get_parameter_value().double_value

        # Build solver (this takes a few seconds due to CUDA init)
        self.get_logger().info(f"Loading cuRobo IK solver from: {config_path}")
        self._solver = CuRoboIKSolver.from_config(
            config_path=config_path,
            ee_link=ee_link,
            device=device,
            num_seeds=num_seeds,
            self_collision_check=self_collision,
            rotation_threshold=rot_thresh,
            position_threshold=pos_thresh,
            warmup=do_warmup,
        )

        self.get_logger().info(
            f"cuRobo IK server ready: {self._solver.nq} DOF, "
            f"ee={self._solver.ee_link}, base={self._solver.base_link}, "
            f"joints={self._solver.joint_names}"
        )

        # Create service
        self._srv = self.create_service(SolveIK, "solve_ik", self._handle_request)
        self.get_logger().info("Service 'solve_ik' is now available")

    def _handle_request(self, request: SolveIK.Request, response: SolveIK.Response) -> SolveIK.Response:
        """Handle incoming IK service requests."""

        # Validate target frame if provided
        if request.target_frame and request.target_frame != self._solver.base_link:
            response.success = False
            response.error_message = (
                f"Frame mismatch: target_pose is in '{request.target_frame}' "
                f"but solver expects '{self._solver.base_link}'. "
                f"Transform the pose to '{self._solver.base_link}' before calling."
            )
            response.base_frame = self._solver.base_link
            self.get_logger().warning(response.error_message)
            return response

        # Validate joint names if provided
        if request.joint_names:
            expected = self._solver.joint_names
            received = list(request.joint_names)
            if received != expected:
                response.success = False
                response.error_message = (
                    f"Joint name mismatch. "
                    f"Expected: {expected}, got: {received}"
                )
                self.get_logger().warning(response.error_message)
                return response

        # Convert ROS Pose -> numpy 4x4
        target_4x4 = ros_pose_to_4x4(request.target_pose)

        # Parse optional seed
        q_seed = None
        if request.seed_joint_positions:
            q_seed = np.array(request.seed_joint_positions, dtype=np.float64)

        # Solve
        t0 = time.perf_counter()
        if request.position_only:
            q, ok = self._solver.ik_position(target_4x4[:3, 3], q_seed)
        else:
            q, ok = self._solver.ik(target_4x4, q_seed)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Fill response metadata (always populated regardless of success)
        response.solved_joint_names = self._solver.joint_names
        response.ee_link = self._solver.ee_link
        response.base_frame = self._solver.base_link
        response.solve_time_ms = dt_ms

        if ok:
            response.success = True
            response.joint_positions = q.tolist()
            if dt_ms > 200.0:
                self.get_logger().warning(
                    f"IK solve took {dt_ms:.0f}ms (expected <100ms) — "
                    f"possible GPU contention or thermal throttling"
                )
            self.get_logger().debug(
                f"IK solved in {dt_ms:.1f}ms: {q.tolist()}"
            )
        else:
            response.success = False
            # Return empty array — not NaN. A client that indexes into an
            # empty array gets an immediate crash rather than silent NaN
            # propagation to motor drivers.
            response.joint_positions = []
            response.error_message = "IK solver failed to find a valid solution"
            pos = target_4x4[:3, 3]
            self.get_logger().warning(
                f"IK failed for target position [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                f"({dt_ms:.1f}ms)"
            )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = CuRoboIKServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
