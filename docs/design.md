# Design and Architecture

This document covers how cuRobo IK works, how this package wraps it, the data flow through the system, configuration format, and the conda+ROS integration challenges and solutions.

## Table of contents

- [How cuRobo IK works](#how-curobo-ik-works)
- [Package architecture](#package-architecture)
- [Data flow](#data-flow)
- [File-by-file breakdown](#file-by-file-breakdown)
- [Configuration format](#configuration-format)
- [Quaternion conventions](#quaternion-conventions)
- [cuRobo vs Pinocchio](#curobo-vs-pinocchio)
- [Conda + ROS 2 integration](#conda--ros-2-integration)
- [Build system](#build-system)
- [Adding a new robot](#adding-a-new-robot)

---

## How cuRobo IK works

[NVIDIA cuRobo](https://curobo.org) formulates inverse kinematics as a **nonlinear optimization problem** and solves it on the GPU using parallel random seeds.

### The optimization approach

Traditional IK solvers (e.g. Pinocchio, KDL) use iterative Jacobian methods from a single seed. cuRobo instead:

1. **Generates N random seed configurations** (default: 20-32) spread across the joint space
2. **Runs parallel optimization** on all seeds simultaneously on the GPU
3. **Uses two-stage optimization**: first Levenberg-Marquardt (LM) for coarse convergence, then L-BFGS for fine-tuning
4. **Evaluates collision costs** as differentiable penalty terms in the objective function
5. **Returns the best solution** that meets position and orientation tolerances

This multi-seed GPU approach achieves:
- **High solve rates** (80-95% for reachable poses) because multiple seeds explore different solution branches
- **Sub-100ms solve times** because GPU parallelism offsets the cost of running many seeds
- **Collision-aware solutions** without post-filtering -- collisions are part of the optimization cost

### Key classes (cuRobo 0.8.x API)

| Class | Purpose |
|-------|---------|
| `InverseKinematicsCfg` | Configuration: robot model, solver params, tolerances |
| `InverseKinematicsCfg.create()` | Factory that loads robot YAML and builds the config |
| `InverseKinematics` | The solver instance. Holds GPU state and CUDA graphs |
| `Pose` | Position (Bx3) + quaternion wxyz (Bx4) on GPU |
| `GoalToolPose` | Maps tool frame names to target Poses |
| `JointState` | Joint positions/velocities on GPU with joint name metadata |

### Solve pipeline

```
Target Pose (position + quaternion)
        |
        v
GoalToolPose.from_poses({ee_link: pose}, num_goalset=1)
        |
        v
InverseKinematics.solve_pose(goal_tool_poses)
        |
        |--- Generate N seed joint configurations
        |--- For each seed (in parallel on GPU):
        |      |--- LM optimizer (coarse)
        |      |--- L-BFGS optimizer (fine)
        |      |--- Evaluate: pose error + collision cost
        |--- Select best feasible solution
        |
        v
IKResult:
  .success          -- bool tensor (Bx1)
  .js_solution      -- JointState with solution positions
  .position_error   -- float tensor (Bx1) in meters
```

### CUDA warmup

The first solve triggers Just-In-Time (JIT) compilation of CUDA kernels via `cuda.core`. This adds 2-5 seconds. Subsequent solves reuse cached kernels. The solver's `warmup=True` parameter runs a dummy solve at init to front-load this cost.

---

## Package architecture

```
                    +--------------------------+
                    |     ROS 2 Service Layer   |
                    |  ik_service_node.py       |
                    |  ik_client.py             |
                    |  SolveIK.srv              |
                    +------------+-------------+
                                 |
                    geometry_msgs/Pose <-> numpy 4x4
                                 |
                    +------------v-------------+
                    |      Library Layer        |
                    |  solver.py                |
                    |  conversions.py           |
                    |  config_loader.py         |
                    +------------+-------------+
                                 |
                    numpy arrays <-> torch tensors
                                 |
                    +------------v-------------+
                    |    NVIDIA cuRobo 0.8.x    |
                    |  (GPU IK solver)          |
                    +------------+-------------+
                                 |
                    +------------v-------------+
                    |   PyTorch + CUDA          |
                    +--------------------------+
```

### Why two layers?

**Library layer** (no ROS dependency):
- Can be used in standalone scripts, Jupyter notebooks, or non-ROS applications
- All public methods take numpy arrays and return numpy arrays
- torch/CUDA details are completely internal
- Testable without a ROS environment

**ROS service layer** (thin wrapper):
- Converts `geometry_msgs/Pose` to/from numpy 4x4 matrices
- Exposes ROS 2 parameters for runtime configuration
- Handles service request/response lifecycle
- ~70 lines of code -- all logic lives in the library layer

---

## Data flow

### Service call flow

```
Client (ROS)                    Service Node                   Library (solver.py)
    |                               |                               |
    |-- SolveIK.Request ----------->|                               |
    |   (geometry_msgs/Pose)        |                               |
    |                               |-- ros_pose_to_4x4() -------->|
    |                               |   (numpy 4x4 matrix)         |
    |                               |                               |-- pose_4x4_to_curobo()
    |                               |                               |   (cuRobo Pose on GPU)
    |                               |                               |
    |                               |                               |-- ik.solve_pose()
    |                               |                               |   (GPU optimization)
    |                               |                               |
    |                               |                               |-- torch_to_numpy()
    |                               |<-- (numpy q, bool ok) --------|
    |                               |                               |
    |<-- SolveIK.Response ----------|                               |
    |   (joint_positions, success)  |                               |
```

### Conversion chain

```
ROS Pose (xyzw quat)
    --> ros_pose_to_4x4()           # scipy Rotation, builds 4x4 matrix
    --> numpy 4x4 homogeneous matrix
    --> pose_4x4_to_curobo()        # extracts pos + wxyz quat, converts to torch
    --> cuRobo Pose (position tensor [1,3], quaternion tensor [1,4] wxyz)
    --> GoalToolPose.from_poses()   # wraps in {ee_link: pose} dict
    --> IK solver on GPU
    --> JointState.position tensor
    --> torch_to_numpy()            # .detach().cpu().numpy()
    --> numpy 1D joint array
```

---

## File-by-file breakdown

### solver.py -- CuRoboIKSolver

The core class. Wraps two `InverseKinematics` instances:

| Instance | Purpose | Orientation tolerance |
|----------|---------|----------------------|
| `_ik` | Full 6-DOF IK | 0.05 rad (configurable) |
| `_ik_pos` | Position-only IK | 100.0 rad (effectively unconstrained) |

**Key methods:**

- `from_config(yaml_path, ...)` -- Factory. Loads YAML, creates both solvers, optionally warms up CUDA
- `fk(q)` -- Forward kinematics via `ik.compute_kinematics(JointState)`
- `ik(pose_4x4)` -- Single 6-DOF solve via `ik.solve_pose(GoalToolPose)`
- `ik_position(xyz)` -- Position-only solve using the loose-orientation solver
- `ik_batch(poses_Nx4x4)` -- Batch solve for N targets in one GPU call

**Thread safety:** Not thread-safe. One instance per process. ROS service callbacks are serialized by the single-threaded executor, so this is safe for the service node.

### conversions.py -- Pose format conversions

Isolated conversion module. Nothing else in the package imports `torch` directly.

| Function | From | To |
|----------|------|-----|
| `numpy_to_torch(arr, device)` | numpy array | float32 torch tensor on GPU |
| `torch_to_numpy(tensor)` | torch tensor | float64 numpy array on CPU |
| `mat4_to_pos_quat_wxyz(pose)` | 4x4 numpy | position [3] + quaternion wxyz [4] |
| `pos_quat_wxyz_to_mat4(pos, quat)` | position + wxyz quat | 4x4 numpy |
| `pose_4x4_to_curobo(pose, device)` | 4x4 numpy | cuRobo `Pose` on GPU |
| `poses_4x4_to_curobo_batch(poses, device)` | (N,4,4) numpy | batched cuRobo `Pose` |
| `ee_pose_to_4x4(pose)` | cuRobo `Pose` (from FK) | 4x4 numpy |
| `ros_pose_to_4x4(ros_pose)` | `geometry_msgs/Pose` | 4x4 numpy |
| `numpy_4x4_to_ros_pose(pose)` | 4x4 numpy | `geometry_msgs/Pose` |

### config_loader.py -- YAML config loading

Loads the cuRobo YAML and resolves relative paths (URDF, assets) to absolute paths based on the YAML file's directory. Handles both string paths and inline dicts (e.g. `collision_spheres` can be a file path or inline data).

### ik_service_node.py -- ROS 2 service server

Declares ROS parameters, creates a `CuRoboIKSolver` at init, and handles `/solve_ik` service requests. Validates joint names if provided. Logs solve times and failures.

**ROS parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | string | (required) | Absolute path to cuRobo YAML |
| `ee_link` | string | "" (use YAML) | End-effector link override |
| `num_seeds` | int | 20 | Parallel IK seeds |
| `self_collision_check` | bool | true | Enable self-collision |
| `warmup` | bool | true | CUDA warmup at startup |
| `device` | string | "cuda:0" | CUDA device |
| `rotation_threshold` | double | 0.05 | Orientation tolerance (rad) |
| `position_threshold` | double | 0.005 | Position tolerance (m) |

### SolveIK.srv -- Service definition

**Request:**
- `geometry_msgs/Pose target_pose` -- Target end-effector pose
- `string[] joint_names` -- Expected joint order (empty = skip validation)
- `float64[] seed_joint_positions` -- Optional seed (cuRobo ignores, accepted for API compat)
- `bool position_only` -- If true, orientation is unconstrained

**Response:**
- `bool success` -- True if a valid collision-free solution was found
- `float64[] joint_positions` -- Joint angles in radians
- `string[] solved_joint_names` -- Joint names in solution order
- `string ee_link` -- EE link name
- `string base_frame` -- Base frame name
- `float64 solve_time_ms` -- Wall-clock solve time
- `string error_message` -- Empty on success

### curobo_ik.launch.py -- Launch file

Declares launch arguments, auto-detects the conda `curobo` environment, and launches the service node with the conda Python interpreter (see [Conda + ROS 2 integration](#conda--ros-2-integration)).

---

## Configuration format

Robot configs use the cuRobo 0.8.x YAML format. Example structure:

```yaml
robot_cfg:
  kinematics:
    format_version: 2.0
    urdf_path: "/absolute/path/to/robot.urdf"
    asset_root_path: "/absolute/path/to/urdf/directory"
    base_link: "base_link"
    tool_frames:
      - "end_effector_link"

    cspace:
      joint_names: ["joint1", "joint2", ...]
      default_joint_position: [0.0, 0.0, ...]
      null_space_weight: [1.0, 1.0, ...]
      cspace_distance_weight: [1.0, 1.0, ...]
      max_acceleration: 10.0
      max_jerk: 150.0

    collision_link_names: [...]
    collision_sphere_buffer: 0.005
    collision_spheres:
      link_name:
        - "center": [x, y, z]
          "radius": r

    lock_joints:
      gripper_joint: 0.025

    self_collision_ignore:
      "link_a": ["link_b", "link_c"]

    self_collision_buffer:
      "link_a": 0.02

    mesh_link_names: [...]
    use_global_cumul: true
```

### Key fields

| Field | Purpose |
|-------|---------|
| `format_version: 2.0` | Required for cuRobo 0.8.x |
| `tool_frames` | End-effector link(s). Replaces the old `ee_link` field |
| `default_joint_position` | Retract/home configuration. Replaces old `retract_config` |
| `collision_spheres` | Inline sphere definitions per link (center + radius). Older versions used a separate YAML file |
| `lock_joints` | Joints to freeze (e.g. gripper joints during arm planning) |
| `self_collision_ignore` | Link pairs to skip during self-collision checking (adjacent links) |
| `self_collision_buffer` | Per-link padding for self-collision spheres |

### Path resolution

- **Absolute paths** are used as-is
- **Relative paths** (in `urdf_path`, `asset_root_path`, `collision_spheres` if it's a file path) are resolved relative to the YAML file's directory by `config_loader.py`
- `collision_spheres` can be either a string (file path) or a dict (inline data). The config loader handles both

---

## Quaternion conventions

Three different quaternion conventions are used across the system:

| System | Convention | Order |
|--------|-----------|-------|
| ROS 2 (`geometry_msgs/Quaternion`) | Hamilton | x, y, z, w |
| cuRobo (`Pose.quaternion`) | Hamilton | w, x, y, z |
| scipy (`Rotation.as_quat()`) | Hamilton | x, y, z, w |

`conversions.py` handles all translations between these formats. The key conversions:

- **ROS -> numpy 4x4**: `ros_pose_to_4x4()` reads xyzw from the message, uses scipy to build rotation matrix
- **numpy 4x4 -> cuRobo**: `pose_4x4_to_curobo()` extracts rotation via scipy (xyzw), reorders to wxyz for cuRobo
- **cuRobo FK -> numpy 4x4**: `ee_pose_to_4x4()` reads wxyz from cuRobo Pose, reorders to xyzw for scipy

### Common pitfall

Sending `orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}` (identity quaternion) means the end-effector is aligned with the world frame. Most robot arms can't achieve this at arbitrary positions due to joint limits and kinematic constraints. Always provide a physically feasible orientation, or use `position_only: true`.

---

## cuRobo vs Pinocchio

| Concern | Pinocchio | cuRobo |
|---------|-----------|--------|
| Model input | URDF path | YAML config file (references URDF internally) |
| Compute | CPU (numpy) | GPU (torch tensors, CUDA) |
| Pose format | 4x4 homogeneous matrix | `Pose(position, quaternion_wxyz)` |
| Joint format | 1D numpy array | 1D torch tensor on GPU |
| Seed | User provides `q_seed` | Internal: `num_seeds` random starts |
| Collision | Not supported | Self-collision + world collision (optional) |
| First-call cost | None | ~2-5s CUDA warmup (kernel compilation) |
| Batch support | No (one pose at a time) | Yes (`solve_pose` for N poses) |
| Thread safety | One instance per thread | One instance per process (GPU context) |
| Dependency | `pinocchio` (~50MB) | `curobo` + `torch` + CUDA (~5GB) |

**When to use Pinocchio:** Lightweight, CPU-only environments. Single-pose IK where solve rate isn't critical. No GPU available.

**When to use cuRobo:** Need collision-aware IK. Batch solving many poses. Sub-100ms solve times. GPU available.

---

## Conda + ROS 2 integration

This is the trickiest part of the setup. Here's why and how it's solved.

### The problem

ROS 2 Jazzy (installed via apt) uses system Python 3.12. cuRobo requires PyTorch + CUDA which can't be installed into system Python (Ubuntu 24.04 blocks `pip install` system-wide via PEP 668). A conda env with Python 3.12 has the right packages but:

1. **Simply adding conda site-packages to `PYTHONPATH` is insufficient** -- cuRobo is installed as an editable package (`pip install -e .`) which uses `.pth` import hooks. These hooks are only processed by the Python interpreter that owns the site-packages directory.

2. **`cuda.pathfinder` needs conda's package metadata** -- cuRobo's CUDA kernel compiler resolves header paths from pip-installed packages (`nvidia-cuda-nvrtc`). System Python can't find these.

3. **ROS 2 launch overwrites `PYTHONPATH`** -- Using `SetEnvironmentVariable` or `additional_env` in the launch file doesn't work because the ROS 2 launch system rebuilds `PYTHONPATH` with its own paths.

### The solution

The launch file (`curobo_ik.launch.py`) uses a **`prefix` wrapper** that re-executes the node with the conda Python binary:

```python
prefix_cmd = f"bash -c 'export PYTHONPATH=\"$PYTHONPATH\"; exec {conda_python} \"$@\"' --"
```

This means:
1. ROS 2 launches the node normally, setting up `PYTHONPATH` with all ROS package paths
2. The `prefix` intercepts the execution, preserves the ROS `PYTHONPATH`, and re-executes with the conda Python
3. The conda Python processes its own `.pth` hooks (making editable cuRobo importable) and can also find ROS packages via the preserved `PYTHONPATH`

### Auto-detection

The launch file automatically finds the conda environment:

1. Checks `CONDA_PREFIX` environment variable (set when a conda env is active)
2. Falls back to `~/miniconda3/envs/curobo` (hardcoded default)
3. Reads the editable install finder module to locate the cuRobo source directory
4. Falls back to `~/projects/cc/curobo` if the finder can't be parsed

If auto-detection fails, set `CONDA_PREFIX` manually:

```bash
export CONDA_PREFIX=~/miniconda3/envs/curobo
ros2 launch curobo_ik_ros curobo_ik.launch.py config_path:=...
```

### Running without ROS

For library-only usage (no ROS), just use the conda Python directly:

```bash
~/miniconda3/envs/curobo/bin/python your_script.py
```

---

## Build system

### CMakeLists.txt

The package uses `ament_cmake` with a mixed C++ (for service generation) and Python setup. There is a known collision between `rosidl_generate_interfaces()` and `ament_python_install_package()` -- both create CMake targets with the same name when the project name matches the Python package name.

**Solution:** The Python package is installed manually via `install(DIRECTORY ...)` instead of using `ament_python_install_package()`:

```cmake
# Detect Python version for correct install path
execute_process(
  COMMAND python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')"
  OUTPUT_VARIABLE PYTHON_VERSION_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Install to the same path where rosidl puts generated Python packages
install(DIRECTORY curobo_ik_ros/
  DESTINATION lib/${PYTHON_VERSION_DIR}/site-packages/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)
```

This ensures the Python package ends up in `lib/python3.12/site-packages/curobo_ik_ros/`, right next to the generated `srv/` subpackage from `rosidl_generate_interfaces`.

### Build commands

```bash
# Clean build (if you hit target collision errors)
rm -rf build/curobo_ik_ros install/curobo_ik_ros

# Build
source /opt/ros/jazzy/setup.bash
cd ~/cc_ros_ws
colcon build --packages-select curobo_ik_ros
source install/setup.bash
```

---

## Adding a new robot

### 1. Prepare the URDF

Place the URDF and any mesh files in `robots/<robot_name>/urdf/`:

```
robots/
  my_robot/
    urdf/
      my_robot.urdf
      meshes/
        link1.stl
        link2.stl
```

### 2. Create the config YAML

Create `config/my_robot_curobo.yml` following the format in `nero_curobo.yml`. Key things to set:

- `format_version: 2.0` (required for cuRobo 0.8.x)
- `urdf_path` -- absolute path to the URDF file
- `tool_frames` -- list of end-effector link names
- `base_link` -- base link name
- `cspace.joint_names` -- actuated joint names (order matters)
- `cspace.default_joint_position` -- home/retract configuration
- `lock_joints` -- any joints to freeze (e.g. gripper joints)
- `collision_spheres` -- inline sphere definitions per collision link
- `self_collision_ignore` -- adjacent link pairs to skip

### Supported robots reference

| Robot | Config | DOF | EE link | Notes |
|-------|--------|-----|---------|-------|
| AgileX Nero 7-DOF + Pika gripper | `nero_curobo.yml` | 7 | `gripper_tip` | Gripper joints locked at half-open |
| WidowX AI (Trossen) 6-DOF + spatula | `wxai_spatula_curobo.yml` | 6 | `spatula_tip` | Gripper joints excluded from cspace |

### 3. Generate collision spheres

cuRobo uses sphere approximations for collision checking. Each link needs a set of spheres that approximate its geometry. You can:

- Manually define spheres by inspecting the URDF mesh geometry
- Use cuRobo's sphere fitting tools (see cuRobo documentation)
- Start with a few large spheres per link and refine

### 4. Test

```bash
~/miniconda3/envs/curobo/bin/python -c "
from curobo_ik_ros import CuRoboIKSolver
solver = CuRoboIKSolver.from_config('config/my_robot_curobo.yml')
print('Joints:', solver.joint_names)
print('FK at home:', solver.fk([0.0] * solver.nq)[:3, 3])
"
```

### 5. Tune

If IK solve rates are low:
- Increase `num_seeds` (e.g. 32 or 64)
- Relax `rotation_threshold` and `position_threshold`
- Check `self_collision_buffer` values -- too large causes valid solutions to be rejected
- Verify `self_collision_ignore` includes all adjacent link pairs
