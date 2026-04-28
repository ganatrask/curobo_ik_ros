#!/usr/bin/env python3
"""
ROS 2 client helper for the cuRobo IK service.

Provides both a reusable class (CuRoboIKClient) for embedding in other nodes,
and a standalone CLI for quick testing.

Reusable:
    from curobo_ik_ros.ik_client import CuRoboIKClient
    client = CuRoboIKClient(node, service_name="/nero/solve_ik")
    q, ok = client.solve(target_pose_4x4)

CLI test:
    ros2 run curobo_ik_ros ik_client.py -- \
        --x -0.2 --y 0.0 --z 0.4 --qw 1.0
"""

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from curobo_ik_ros.conversions import numpy_4x4_to_ros_pose, ros_pose_to_4x4
from curobo_ik_ros.srv import SolveIK


class CuRoboIKClient:
    """Convenience wrapper for calling the cuRobo IK service.

    Attach to an existing ROS 2 node — does not create its own.
    """

    def __init__(
        self,
        node: Node,
        service_name: str = "solve_ik",
        timeout_sec: float = 30.0,
    ):
        """
        Args:
            node: An existing rclpy Node to attach the client to.
            service_name: Fully qualified service name (e.g. /nero/solve_ik).
            timeout_sec: How long to wait for service availability at startup.

        Raises:
            RuntimeError: If the service is not available within timeout.
        """
        self._node = node
        self._client = node.create_client(SolveIK, service_name)
        if not self._client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                f"Service '{service_name}' not available after {timeout_sec}s. "
                f"Is the curobo_ik_server running?"
            )
        node.get_logger().info(f"Connected to IK service: {service_name}")

    def solve(
        self,
        target_pose: np.ndarray,
        joint_names: Optional[list[str]] = None,
        seed_joint_positions: Optional[np.ndarray] = None,
        position_only: bool = False,
        target_frame: str = "",
    ) -> tuple[np.ndarray, bool]:
        """Call IK service synchronously.

        WARNING: This method calls rclpy.spin_until_future_complete(), which
        takes over the node's executor. Do NOT call this from inside a
        callback on a node that is already being spun — it will deadlock.
        Use solve_async() instead for nodes with their own executor.
        This method is safe for standalone scripts and CLI tools.

        Args:
            target_pose: 4x4 numpy homogeneous matrix of desired EE pose.
            joint_names: Optional joint name list for server-side validation.
            seed_joint_positions: Optional seed (cuRobo ignores, for API compat).
            position_only: If True, solve position-only IK.
            target_frame: Frame the pose is expressed in (server validates
                against its base_link; empty = skip check).

        Returns:
            (joint_positions, success): numpy array (nq,) and bool.
            On failure, joint_positions is an empty array.

        Raises:
            RuntimeError: If the service call itself fails (network error,
                server crashed, etc.).
        """
        req = SolveIK.Request()
        req.target_pose = numpy_4x4_to_ros_pose(target_pose)
        req.target_frame = target_frame
        req.joint_names = joint_names or []
        req.seed_joint_positions = (
            seed_joint_positions.tolist() if seed_joint_positions is not None else []
        )
        req.position_only = position_only

        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        resp = future.result()
        if resp is None:
            raise RuntimeError(
                "IK service call failed — no response received. "
                "Is the curobo_ik_server still running?"
            )

        q = np.array(resp.joint_positions, dtype=np.float64)

        if not resp.success:
            self._node.get_logger().warning(
                f"IK failed: {resp.error_message} "
                f"(ee={resp.ee_link}, base={resp.base_frame})"
            )

        return q, resp.success

    async def solve_async(
        self,
        target_pose: np.ndarray,
        joint_names: Optional[list[str]] = None,
        seed_joint_positions: Optional[np.ndarray] = None,
        position_only: bool = False,
        target_frame: str = "",
    ) -> tuple[np.ndarray, bool]:
        """Async version for use inside ROS callbacks.

        Must be awaited in an async context (e.g. async service/action callback).
        """
        req = SolveIK.Request()
        req.target_pose = numpy_4x4_to_ros_pose(target_pose)
        req.target_frame = target_frame
        req.joint_names = joint_names or []
        req.seed_joint_positions = (
            seed_joint_positions.tolist() if seed_joint_positions is not None else []
        )
        req.position_only = position_only

        resp = await self._client.call_async(req)
        if resp is None:
            raise RuntimeError(
                "IK service call failed — no response received. "
                "Is the curobo_ik_server still running?"
            )

        q = np.array(resp.joint_positions, dtype=np.float64)
        return q, resp.success


# ---------------------------------------------------------------------------
# Standalone CLI for quick testing
# ---------------------------------------------------------------------------

def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Call cuRobo IK service")
    parser.add_argument("--service", default="solve_ik", help="Service name")
    parser.add_argument("--x", type=float, required=True, help="Target X position")
    parser.add_argument("--y", type=float, required=True, help="Target Y position")
    parser.add_argument("--z", type=float, required=True, help="Target Z position")
    parser.add_argument("--qw", type=float, default=1.0, help="Quaternion W")
    parser.add_argument("--qx", type=float, default=0.0, help="Quaternion X")
    parser.add_argument("--qy", type=float, default=0.0, help="Quaternion Y")
    parser.add_argument("--qz", type=float, default=0.0, help="Quaternion Z")
    parser.add_argument("--frame", default="base_link",
                        help="Frame the target pose is in (default: base_link)")
    parser.add_argument("--position-only", action="store_true",
                        help="Solve position-only IK")
    cli_args = parser.parse_args()

    # Build 4x4 target pose from CLI args
    from scipy.spatial.transform import Rotation
    quat_xyzw = [cli_args.qx, cli_args.qy, cli_args.qz, cli_args.qw]
    rot = Rotation.from_quat(quat_xyzw)
    target = np.eye(4)
    target[:3, :3] = rot.as_matrix()
    target[:3, 3] = [cli_args.x, cli_args.y, cli_args.z]

    rclpy.init(args=args)
    node = rclpy.create_node("curobo_ik_client_cli")

    try:
        client = CuRoboIKClient(node, service_name=cli_args.service)
        q, ok = client.solve(
            target, position_only=cli_args.position_only,
            target_frame=cli_args.frame,
        )
        if ok:
            print(f"IK SUCCESS")
            print(f"  Joint positions: {q.tolist()}")
        else:
            print(f"IK FAILED")
    except RuntimeError as e:
        print(f"ERROR: {e}")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
