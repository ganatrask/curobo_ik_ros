# curobo_ik_ros

GPU-accelerated inverse kinematics via NVIDIA cuRobo, exposed as a ROS 2 Jazzy service.

## Package Structure

- **Library layer** (`solver.py`, `conversions.py`, `config_loader.py`) — standalone, no ROS dependency
- **ROS service layer** (`ik_service_node.py`, `ik_client.py`, `SolveIK.srv`) — wraps library as a ROS 2 service

## Quick Start

### Library usage (no ROS)

```python
from curobo_ik_ros import CuRoboIKSolver

solver = CuRoboIKSolver.from_config("config/nero_curobo.yml", ee_link="gripper_tip")

# FK
pose_4x4 = solver.fk(q_joints)

# IK
q, ok = solver.ik(target_pose_4x4)
```

### ROS 2 service

```bash
# Build
cd ~/ros2_ws/src
ln -s /path/to/curobo_ik_ros .
cd .. && colcon build --packages-select curobo_ik_ros
source install/setup.bash

# Launch server
ros2 launch curobo_ik_ros curobo_ik.launch.py \
    config_path:=/path/to/nero_curobo.yml \
    ee_link:=gripper_tip \
    namespace:=nero

# Call service
ros2 service call /nero/solve_ik curobo_ik_ros/srv/SolveIK \
    "{target_pose: {position: {x: -0.2, y: 0.0, z: 0.4}, orientation: {w: 1.0}}, position_only: false}"
```

## Dependencies

- `nvidia-curobo` >= 0.7
- `torch` >= 2.0
- `numpy`, `scipy`, `pyyaml`
- ROS 2 Jazzy (`rclpy`, `geometry_msgs`)

## Supported Robots

- AgileX Nero 7-DOF (config included in `config/`)
