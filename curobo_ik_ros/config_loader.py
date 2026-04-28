"""
cuRobo YAML config loader with relative path resolution.

Loads a cuRobo robot config YAML and resolves urdf_path, collision_spheres,
and asset_root_path relative to the YAML file's directory. This avoids
depending on cuRobo's internal path resolution or eval_utils.
"""

import os
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load a cuRobo YAML config file and resolve relative paths.

    Args:
        config_path: Absolute or relative path to the cuRobo YAML config.

    Returns:
        Parsed config dict with resolved absolute paths for:
        - robot_cfg.kinematics.urdf_path
        - robot_cfg.kinematics.collision_spheres
        - robot_cfg.kinematics.asset_root_path

    Raises:
        FileNotFoundError: If config_path does not exist.
        KeyError: If required robot_cfg.kinematics section is missing.
    """
    config_path = str(Path(config_path).resolve())
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"cuRobo config not found: {config_path}")

    yaml_dir = os.path.dirname(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "robot_cfg" not in cfg or "kinematics" not in cfg["robot_cfg"]:
        raise KeyError(
            f"Config file missing robot_cfg.kinematics section: {config_path}"
        )

    kin = cfg["robot_cfg"]["kinematics"]

    # Resolve paths that may be relative to the YAML file's directory
    for key in ("urdf_path", "collision_spheres", "asset_root_path"):
        val = kin.get(key)
        if isinstance(val, str) and val and not os.path.isabs(val):
            kin[key] = os.path.normpath(os.path.join(yaml_dir, val))

    return cfg
