"""Load ``specimen_config.yaml``; resolve_path joins ``paths.*`` to the config file's directory.

YAML keys may include unit hints (``fy_ksi``, ``L_T_in``, …); values are read as floats in that system.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .jfeat_landmarks import yield_displacement_dy


def load_specimen_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dy_from_config(cfg: dict[str, Any]) -> float | None:
    return yield_displacement_dy(
        fy=float(cfg["fy_ksi"]),
        E=float(cfg["E_ksi"]),
        L_T=float(cfg["L_T_in"]),
        L_y=float(cfg["L_y_in"]),
        A_sc=float(cfg["A_sc_in2"]),
        A_t=float(cfg["A_t_in2"]),
    )


def resolve_path(cfg: dict[str, Any], key: str, base_dir: Path) -> Path:
    rel = cfg.get("paths", {}).get(key)
    if not rel:
        raise KeyError(f"paths.{key} missing in specimen config")
    p = Path(rel)
    if not p.is_absolute():
        p = base_dir / p
    return p
