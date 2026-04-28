"""
Conversion utilities between numpy, torch, cuRobo Pose, and ROS geometry_msgs.

All cuRobo/torch details are isolated here — nothing else in the package
imports torch directly.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from curobo.types import Pose
    from geometry_msgs.msg import Pose as RosPose


# ---------------------------------------------------------------------------
# numpy <-> torch
# ---------------------------------------------------------------------------

def numpy_to_torch(arr: np.ndarray, device: str = "cuda:0") -> "torch.Tensor":
    """Convert numpy array to float32 torch tensor on device."""
    import torch
    return torch.tensor(arr, dtype=torch.float32, device=device)


def torch_to_numpy(t: "torch.Tensor") -> np.ndarray:
    """Convert torch tensor to numpy (CPU, float64)."""
    return t.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# 4x4 matrix <-> position + quaternion
# ---------------------------------------------------------------------------

def mat4_to_pos_quat_wxyz(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position [3] and quaternion wxyz [4] from a 4x4 matrix."""
    pos = pose[:3, 3].copy()
    rot = Rotation.from_matrix(pose[:3, :3])
    quat_xyzw = rot.as_quat()  # scipy convention
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def pos_quat_wxyz_to_mat4(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous matrix from position and wxyz quaternion."""
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = Rotation.from_quat(quat_xyzw)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T


# ---------------------------------------------------------------------------
# numpy 4x4 <-> cuRobo Pose
# ---------------------------------------------------------------------------

def pose_4x4_to_curobo(pose: np.ndarray, device: str = "cuda:0") -> "Pose":
    """Convert a 4x4 numpy matrix to a cuRobo Pose (single, batched as [1,...])."""
    from curobo.types import Pose
    pos, quat_wxyz = mat4_to_pos_quat_wxyz(pose)
    return Pose(
        position=numpy_to_torch(pos, device).unsqueeze(0),
        quaternion=numpy_to_torch(quat_wxyz, device).unsqueeze(0),
    )


def poses_4x4_to_curobo_batch(poses: np.ndarray, device: str = "cuda:0") -> "Pose":
    """Convert (N, 4, 4) numpy array to a batched cuRobo Pose."""
    from curobo.types import Pose
    positions = poses[:, :3, 3].copy()  # (N, 3)
    rots = Rotation.from_matrix(poses[:, :3, :3])
    quats_xyzw = rots.as_quat()  # (N, 4) xyzw
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]  # -> wxyz
    return Pose(
        position=numpy_to_torch(positions, device),
        quaternion=numpy_to_torch(quats_wxyz, device),
    )


def ee_pose_to_4x4(pose) -> np.ndarray:
    """Convert a cuRobo Pose (from tool_poses) to a 4x4 numpy matrix."""
    pos = torch_to_numpy(pose.position[0])
    quat_wxyz = torch_to_numpy(pose.quaternion[0])
    return pos_quat_wxyz_to_mat4(pos, quat_wxyz)


# ---------------------------------------------------------------------------
# numpy 4x4 <-> ROS geometry_msgs/Pose
# ---------------------------------------------------------------------------

def ros_pose_to_4x4(ros_pose: "RosPose") -> np.ndarray:
    """Convert geometry_msgs/Pose to 4x4 numpy matrix.

    ROS quaternion is xyzw convention in the message fields.
    """
    pos = np.array([ros_pose.position.x, ros_pose.position.y, ros_pose.position.z])
    quat_xyzw = np.array([
        ros_pose.orientation.x,
        ros_pose.orientation.y,
        ros_pose.orientation.z,
        ros_pose.orientation.w,
    ])
    rot = Rotation.from_quat(quat_xyzw)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T


def numpy_4x4_to_ros_pose(pose: np.ndarray) -> "RosPose":
    """Convert 4x4 numpy matrix to geometry_msgs/Pose.

    Returns a Pose with xyzw quaternion fields (ROS convention).
    """
    from geometry_msgs.msg import Pose as RosPose, Point, Quaternion
    pos = pose[:3, 3]
    rot = Rotation.from_matrix(pose[:3, :3])
    quat_xyzw = rot.as_quat()
    return RosPose(
        position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation=Quaternion(
            x=float(quat_xyzw[0]),
            y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]),
            w=float(quat_xyzw[3]),
        ),
    )
