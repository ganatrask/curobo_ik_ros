# curobo_ik_ros

GPU-accelerated inverse kinematics via [NVIDIA cuRobo](https://curobo.org), exposed as a ROS 2 Jazzy service.

## Overview

This package wraps cuRobo's GPU-based IK solver into two layers:

- **Library layer** -- standalone Python API, numpy-in/numpy-out, no ROS dependency
- **ROS 2 service layer** -- exposes the library as a `/solve_ik` service

## Supported robots

| Robot | Config | DOF | EE link |
|-------|--------|-----|---------|
| AgileX Nero + Pika gripper | `config/nero_curobo.yml` | 7 | `gripper_tip` |
| WidowX AI (Trossen) + spatula | `config/wxai_spatula_curobo.yml` | 6 | `spatula_tip` |

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Ubuntu | 24.04 | |
| NVIDIA GPU | Compute capability 6.0+ | RTX 3060 or better recommended |
| NVIDIA Driver | 525+ | Check with `nvidia-smi` |
| ROS 2 | Jazzy | `sudo apt install ros-jazzy-desktop` |
| Miniconda | Latest | [Install guide](https://docs.anaconda.com/miniconda/) |
| git-lfs | Any | `sudo apt install git-lfs` |

## Installation

### 1. Clone the repository

```bash
mkdir -p ~/cc_ros_ws/src && cd ~/cc_ros_ws/src
git clone <repo-url> curobo_ik_ros
```

### 2. Create conda environment

Python version **must be 3.12** to match ROS 2 Jazzy's system Python:

```bash
conda create -n curobo python=3.12 -y
```

### 3. Install PyTorch with CUDA

```bash
conda run -n curobo pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
conda run -n curobo python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
# Expected: 2.6.x+cu124 | CUDA: True
```

### 4. Install cuRobo from source

```bash
cd ~/cc_ros_ws/src
git clone https://github.com/NVlabs/curobo.git
cd curobo
conda run -n curobo pip install -e . --no-build-isolation   # ~20 minutes (CUDA kernel compilation)
conda run -n curobo pip install cuda-core
```

Verify:

```bash
conda run -n curobo python -c "from curobo.inverse_kinematics import InverseKinematics; print('cuRobo OK')"
```

### 5. Build the ROS 2 package

```bash
source /opt/ros/jazzy/setup.bash
cd ~/cc_ros_ws
colcon build --packages-select curobo_ik_ros
source install/setup.bash
```

### 6. Verify

```bash
cd ~/cc_ros_ws/src/curobo_ik_ros
~/miniconda3/envs/curobo/bin/python test/test_solver.py
```

## Quick start

### Library (no ROS)

```python
from curobo_ik_ros import CuRoboIKSolver

solver = CuRoboIKSolver.from_config("config/nero_curobo.yml", ee_link="gripper_tip")

pose_4x4 = solver.fk(q_joints)                     # forward kinematics
q, ok = solver.ik(target_pose_4x4)                  # 6-DOF IK
q, ok = solver.ik_position(target_xyz)              # position-only IK
q_batch, ok_batch = solver.ik_batch(poses_Nx4x4)   # batch IK
```

### ROS 2 service

**Nero arm:**

```bash
# Terminal 1: Launch
source /opt/ros/jazzy/setup.bash && source ~/cc_ros_ws/install/setup.bash
ros2 launch curobo_ik_ros curobo_ik.launch.py \
    config_path:=$HOME/cc_ros_ws/src/curobo_ik_ros/config/nero_curobo.yml \
    ee_link:=gripper_tip \
    namespace:=nero

# Terminal 2: Call
ros2 service call /nero/solve_ik curobo_ik_ros/srv/SolveIK \
    "{target_pose: {position: {x: -0.54, y: 0.11, z: 0.56}, \
      orientation: {x: -0.197, y: -0.586, z: -0.150, w: 0.771}}, \
      position_only: false}"
```

**WidowX AI (Trossen) with spatula:**

```bash
# Terminal 1: Launch
source /opt/ros/jazzy/setup.bash && source ~/cc_ros_ws/install/setup.bash
ros2 launch curobo_ik_ros curobo_ik.launch.py \
    config_path:=$HOME/cc_ros_ws/src/curobo_ik_ros/config/wxai_spatula_curobo.yml \
    ee_link:=spatula_tip \
    namespace:=wxai

# Terminal 2: Call
ros2 service call /wxai/solve_ik curobo_ik_ros/srv/SolveIK \
    "{target_pose: {position: {x: 0.56, y: 0.0, z: 0.318}, \
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, \
      position_only: false}"
```

> `config_path` must be absolute. Use `$HOME` instead of `~`.

> The orientation quaternion must be physically reachable at the target position.
> Use `position_only: true` if you only care about position.

## Project structure

```
curobo_ik_ros/
├── curobo_ik_ros/              # Python package
│   ├── __init__.py             # Exports CuRoboIKSolver
│   ├── solver.py               # Core GPU IK solver (library layer)
│   ├── conversions.py          # numpy/torch/ROS pose conversions
│   ├── config_loader.py        # YAML config loader with path resolution
│   ├── ik_service_node.py      # ROS 2 service server
│   └── ik_client.py            # ROS 2 service client + CLI
├── config/                     # Robot configs (cuRobo 0.8.x YAML)
│   ├── nero_curobo.yml         # Nero 7-DOF arm
│   └── wxai_spatula_curobo.yml # WidowX AI 6-DOF + spatula
├── robots/                     # Robot URDFs and meshes
│   ├── nero/                   # AgileX Nero
│   └── wxai/                   # WidowX AI (Trossen)
├── srv/SolveIK.srv             # ROS 2 service definition
├── launch/curobo_ik.launch.py  # ROS 2 launch file
├── test/test_solver.py         # Library-layer smoke test
└── docs/                       # Detailed design documentation
```

## Documentation

- [Design and architecture](docs/design.md) -- how cuRobo IK works, package design decisions, data flow, configuration format, and conda+ROS integration details

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'curobo'` | Using system Python | Use `~/miniconda3/envs/curobo/bin/python` |
| `No module named 'cuda.core'` | Missing dependency | `conda run -n curobo pip install cuda-core` |
| `No module named 'lark'` | ROS PYTHONPATH leaking into conda | `PYTHONPATH="" conda run -n curobo ...` |
| `batch_size exceeds max_batch_size` | Solver batch limit | Pass `max_batch_size=N` to `from_config()` |
| `add_custom_target cannot create target` | Stale build | `rm -rf build/curobo_ik_ros install/curobo_ik_ros` and rebuild |
| URDF not found | Wrong path in config | Verify `urdf_path` is absolute and file exists |
| IK always returns `success: false` | Unreachable orientation | Use `position_only: true` or provide a feasible quaternion |
| `stat: path should be ... not NoneType` | System Python can't find CUDA headers | Use conda Python binary directly |
| ROS node can't import `curobo`/`torch` | Conda env not detected | `export CONDA_PREFIX=~/miniconda3/envs/curobo` before launch |
