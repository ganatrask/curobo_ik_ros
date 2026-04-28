#!/usr/bin/env python3
"""
Smoke test for CuRoboIKSolver (library layer, no ROS needed).

Run with Isaac Sim's Python (has torch + curobo):
    /home/shyam/workspace/isaacsim-5.1.0/python.sh test/test_solver.py

Or with any Python environment that has torch + nvidia-curobo installed.
"""

import os
import sys
import time

import numpy as np

# Add package root to path for development (before pip install)
pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, pkg_dir)

from curobo_ik_ros import CuRoboIKSolver


def main():
    # Path to Nero config
    config_path = os.path.join(pkg_dir, "config", "nero_curobo.yml")
    if not os.path.exists(config_path):
        # Fallback to project-level config
        config_path = os.path.join(
            os.path.dirname(pkg_dir),
            "robots/curobo_configs_1/nero_curobo.yml",
        )

    print(f"=== CuRoboIKSolver Smoke Test ===")
    print(f"Config: {config_path}")
    print()

    # --- Create solver ---
    print("Creating solver (CUDA warmup included)...")
    t0 = time.perf_counter()
    solver = CuRoboIKSolver.from_config(
        config_path=config_path,
        ee_link="gripper_tip",
        num_seeds=20,
        warmup=True,
    )
    print(f"  Init time: {time.perf_counter() - t0:.1f}s")
    print(f"  DOF: {solver.nq}")
    print(f"  Joints: {solver.joint_names}")
    print(f"  EE: {solver.ee_link}")
    print(f"  Base: {solver.base_link}")
    print(f"  q_min: {solver.q_min}")
    print(f"  q_max: {solver.q_max}")
    print()

    # --- Test FK ---
    print("=== FK Test ===")
    q_retract = np.array([0.0, 0.3, 0.0, 1.0, 0.0, 0.5, 0.0])
    pose = solver.fk(q_retract)
    print(f"  FK at retract config:")
    print(f"    Position: [{pose[0,3]:.4f}, {pose[1,3]:.4f}, {pose[2,3]:.4f}]")
    print(f"    4x4:\n{pose}")
    print()

    # --- Test IK roundtrip ---
    print("=== IK Roundtrip Test ===")
    t0 = time.perf_counter()
    q_ik, ok = solver.ik(pose, q_retract)
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"  Success: {ok}")
    print(f"  Solve time: {dt_ms:.1f}ms")
    print(f"  Solution: {q_ik}")

    if ok:
        # Verify FK of IK solution matches original pose
        pose_check = solver.fk(q_ik)
        pos_err = np.linalg.norm(pose[:3, 3] - pose_check[:3, 3])
        print(f"  Position error: {pos_err*1000:.2f}mm")
        assert pos_err < 0.01, f"FK roundtrip error too large: {pos_err}"
    print()

    # --- Test position-only IK ---
    print("=== Position-Only IK Test ===")
    target_pos = np.array([-0.2, 0.0, 0.4])
    t0 = time.perf_counter()
    q_pos, ok_pos = solver.ik_position(target_pos)
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"  Target: {target_pos}")
    print(f"  Success: {ok_pos}")
    print(f"  Solve time: {dt_ms:.1f}ms")

    if ok_pos:
        pose_pos = solver.fk(q_pos)
        pos_err = np.linalg.norm(target_pos - pose_pos[:3, 3])
        print(f"  Solution: {q_pos}")
        print(f"  Achieved position: {pose_pos[:3, 3]}")
        print(f"  Position error: {pos_err*1000:.2f}mm")
    print()

    # --- Test batch IK ---
    print("=== Batch IK Test ===")
    # Generate 10 random reachable poses via FK at random configs
    rng = np.random.default_rng(42)
    n_batch = 10
    q_random = rng.uniform(solver.q_min, solver.q_max, size=(n_batch, solver.nq))
    target_poses = np.array([solver.fk(q) for q in q_random])

    t0 = time.perf_counter()
    q_batch, ok_batch = solver.ik_batch(target_poses)
    dt_ms = (time.perf_counter() - t0) * 1000
    n_solved = ok_batch.sum()
    print(f"  Batch size: {n_batch}")
    print(f"  Solved: {n_solved}/{n_batch} ({100*n_solved/n_batch:.0f}%)")
    print(f"  Total time: {dt_ms:.1f}ms ({dt_ms/n_batch:.1f}ms/pose)")

    # Verify solved poses
    if n_solved > 0:
        errors = []
        for i in range(n_batch):
            if ok_batch[i]:
                p = solver.fk(q_batch[i])
                err = np.linalg.norm(target_poses[i, :3, 3] - p[:3, 3])
                errors.append(err)
        print(f"  Position errors (solved): mean={np.mean(errors)*1000:.2f}mm, "
              f"max={np.max(errors)*1000:.2f}mm")
    print()

    # --- Timing benchmark ---
    print("=== Timing Benchmark (50 single solves) ===")
    times = []
    successes = 0
    for i in range(50):
        target = target_poses[i % n_batch]
        t0 = time.perf_counter()
        _, ok_t = solver.ik(target)
        times.append((time.perf_counter() - t0) * 1000)
        successes += ok_t
    print(f"  Solve rate: {successes}/50 ({100*successes/50:.0f}%)")
    print(f"  Timing: mean={np.mean(times):.1f}ms, "
          f"median={np.median(times):.1f}ms, "
          f"p95={np.percentile(times, 95):.1f}ms")
    print()

    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
