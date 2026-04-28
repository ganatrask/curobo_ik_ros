"""
CuRoboIKSolver — GPU-accelerated inverse kinematics using NVIDIA cuRobo.

Standalone library with no ROS dependency. All public methods accept and
return numpy arrays; torch/CUDA details are internal.

Usage:
    from curobo_ik_ros import CuRoboIKSolver

    solver = CuRoboIKSolver.from_config("nero_curobo.yml", ee_link="gripper_tip")
    q, ok = solver.ik(target_pose_4x4)
"""

import time
import logging
from typing import Optional

import numpy as np

from curobo_ik_ros.config_loader import load_config
from curobo_ik_ros.conversions import (
    ee_pose_to_4x4,
    numpy_to_torch,
    pose_4x4_to_curobo,
    poses_4x4_to_curobo_batch,
    torch_to_numpy,
)

logger = logging.getLogger(__name__)


class CuRoboIKSolver:
    """GPU-accelerated IK solver using NVIDIA cuRobo.

    All public methods accept and return numpy arrays.
    GPU/torch details are internal.

    Thread safety: NOT safe to use from multiple threads. One instance
    per process (GPU context). ROS service callbacks are serialized by
    the single-threaded executor, so this is fine for the service node.
    """

    def __init__(self, ik_solver, ik_solver_pos, joint_names,
                 ee_link, base_link, device):
        self._ik = ik_solver
        self._ik_pos = ik_solver_pos
        self._joint_names = list(joint_names)
        self._ee_link = ee_link
        self._base_link = base_link
        self._device = device

        # Cache joint limits (avoid repeated GPU->CPU transfer)
        jl = ik_solver.kinematics.get_joint_limits()
        self._q_min = torch_to_numpy(jl.position[0])
        self._q_max = torch_to_numpy(jl.position[1])

    @classmethod
    def from_config(
        cls,
        config_path: str,
        ee_link: Optional[str] = None,
        device: str = "cuda:0",
        num_seeds: int = 20,
        self_collision_check: bool = True,
        rotation_threshold: float = 0.05,
        position_threshold: float = 0.005,
        max_batch_size: int = 10,
        warmup: bool = True,
    ) -> "CuRoboIKSolver":
        """Create solver from a cuRobo YAML config file.

        Args:
            config_path: Path to cuRobo YAML (e.g. nero_curobo.yml).
            ee_link: Override EE link from YAML (None = use YAML default).
            device: CUDA device string.
            num_seeds: Number of random IK seeds (more = higher solve rate).
            self_collision_check: Enable self-collision checking.
            rotation_threshold: Orientation convergence threshold (rad).
            position_threshold: Position convergence threshold (meters).
            warmup: Run a dummy solve at init to trigger CUDA kernel compilation.

        Returns:
            Configured CuRoboIKSolver instance.
        """
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

        logger.info("Loading cuRobo config from %s", config_path)
        cfg_dict = load_config(config_path)

        robot_cfg = cfg_dict["robot_cfg"]

        # Override EE link if specified
        if ee_link:
            robot_cfg["kinematics"]["tool_frames"] = [ee_link]
        resolved_ee = robot_cfg["kinematics"]["tool_frames"][0]
        base_link = robot_cfg["kinematics"].get("base_link", "base_link")

        # Full 6-DOF IK solver
        logger.info("Building full IK solver (num_seeds=%d)", num_seeds)
        ik_config = InverseKinematicsCfg.create(
            robot=robot_cfg,
            orientation_tolerance=rotation_threshold,
            position_tolerance=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=self_collision_check,
            use_cuda_graph=False,
            max_batch_size=max_batch_size,
        )
        ik_solver = InverseKinematics(ik_config)

        # Position-only IK solver (very loose rotation threshold)
        logger.info("Building position-only IK solver")
        ik_config_pos = InverseKinematicsCfg.create(
            robot=robot_cfg,
            orientation_tolerance=100.0,  # effectively unconstrained
            position_tolerance=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=self_collision_check,
            use_cuda_graph=False,
            max_batch_size=max_batch_size,
        )
        ik_solver_pos = InverseKinematics(ik_config_pos)

        # Read joint names from the solver
        joint_names = ik_solver.joint_names

        instance = cls(
            ik_solver=ik_solver,
            ik_solver_pos=ik_solver_pos,
            joint_names=joint_names,
            ee_link=resolved_ee,
            base_link=base_link,
            device=device,
        )

        if warmup:
            logger.info("Warming up CUDA kernels...")
            t0 = time.perf_counter()
            instance._warmup()
            dt = time.perf_counter() - t0
            logger.info("CUDA warmup complete (%.1fs)", dt)

        logger.info(
            "CuRoboIKSolver ready: %d DOF, ee=%s, base=%s, joints=%s",
            instance.nq, resolved_ee, base_link, joint_names,
        )
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nq(self) -> int:
        """Number of actuated joints."""
        return len(self._joint_names)

    @property
    def joint_names(self) -> list[str]:
        """Joint names in solution order."""
        return list(self._joint_names)

    @property
    def q_min(self) -> np.ndarray:
        """Lower joint limits (radians)."""
        return self._q_min.copy()

    @property
    def q_max(self) -> np.ndarray:
        """Upper joint limits (radians)."""
        return self._q_max.copy()

    @property
    def ee_link(self) -> str:
        """End-effector link name."""
        return self._ee_link

    @property
    def base_link(self) -> str:
        """Base link name."""
        return self._base_link

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint angles -> 4x4 homogeneous matrix.

        Args:
            q: Joint positions, shape (nq,).

        Returns:
            4x4 numpy array (float64) representing EE pose in base frame.
        """
        import torch
        from curobo._src.state.state_joint import JointState

        q_t = numpy_to_torch(q, self._device)
        js = JointState(position=q_t.unsqueeze(0), joint_names=self._joint_names)
        kin_state = self._ik.compute_kinematics(js)
        pose = kin_state.tool_poses[self._ee_link]
        return ee_pose_to_4x4(pose)

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def ik(
        self,
        target_pose: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
        **kwargs,
    ) -> tuple[np.ndarray, bool]:
        """Full 6-DOF inverse kinematics.

        Args:
            target_pose: 4x4 homogeneous matrix of desired EE pose.
            q_seed: Optional seed joint positions (accepted for API compat,
                    cuRobo uses its own internal multi-seed strategy).

        Returns:
            (joint_positions, success): numpy array of shape (nq,) and bool.
            On failure, returns q_seed (or zeros) and False.
        """
        from curobo.types import GoalToolPose

        goal = pose_4x4_to_curobo(target_pose, self._device)
        goal_tool = GoalToolPose.from_poses(
            {self._ee_link: goal}, num_goalset=1
        )
        result = self._ik.solve_pose(goal_tool)
        if result.success.item():
            return torch_to_numpy(result.js_solution.position[0, 0, :self.nq]), True
        fallback = q_seed if q_seed is not None else np.zeros(self.nq)
        return fallback.copy(), False

    def ik_position(
        self,
        target_position: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
        **kwargs,
    ) -> tuple[np.ndarray, bool]:
        """Position-only IK (orientation unconstrained).

        Args:
            target_position: [x, y, z] target in base frame.
            q_seed: Optional seed (accepted for API compat, ignored by cuRobo).

        Returns:
            (joint_positions, success)
        """
        from curobo.types import GoalToolPose

        # Build a target pose with identity orientation (the position-only
        # solver has orientation_tolerance=100 so orientation doesn't matter)
        pose_4x4 = np.eye(4)
        pose_4x4[:3, 3] = target_position
        goal = pose_4x4_to_curobo(pose_4x4, self._device)
        goal_tool = GoalToolPose.from_poses(
            {self._ee_link: goal}, num_goalset=1
        )
        result = self._ik_pos.solve_pose(goal_tool)
        if result.success.item():
            return torch_to_numpy(result.js_solution.position[0, 0, :self.nq]), True
        fallback = q_seed if q_seed is not None else np.zeros(self.nq)
        return fallback.copy(), False

    def ik_batch(
        self,
        target_poses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch IK for N poses.

        Args:
            target_poses: (N, 4, 4) array of target EE poses.

        Returns:
            (joint_positions, success): shapes (N, nq) and (N,) bool.
        """
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose

        goals = poses_4x4_to_curobo_batch(target_poses, self._device)
        goal_tool = GoalToolPose.from_poses(
            {self._ee_link: goals}, num_goalset=1
        )
        # For batch, we need a solver configured with the right batch size.
        # Re-solve with the existing solver (it handles batch internally).
        result = self._ik.solve_pose(goal_tool)
        q_all = torch_to_numpy(result.js_solution.position[:, 0, :self.nq])  # (N, nq)
        ok_all = result.success.squeeze(-1).cpu().numpy().astype(bool)  # (N,)
        return q_all, ok_all

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _warmup(self):
        """Run a dummy IK solve to trigger CUDA kernel compilation."""
        import torch
        from curobo.types import Pose, GoalToolPose
        dummy = Pose(
            position=torch.zeros(1, 3, device=self._device, dtype=torch.float32),
            quaternion=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=self._device, dtype=torch.float32
            ),
        )
        goal_tool = GoalToolPose.from_poses(
            {self._ee_link: dummy}, num_goalset=1
        )
        self._ik.solve_pose(goal_tool)
        self._ik_pos.solve_pose(goal_tool)
