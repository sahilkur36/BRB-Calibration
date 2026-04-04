"""
Build landmark_cache.json from experimental target_displacement / target_force.

Those single-row CSVs are normally produced by ``scripts/setup_cycle_targets.py`` alongside
``cycle_meta.json``. Re-run when targets, ``cycle_meta.json``, or specimen geometry (fy, A_sc, Dy)
change. Forward ``model.py`` then only needs the cache + simulated forces for ``results.out``
(no ``target_force.csv`` read for landmarks).

  python scripts/precompute_landmark_cache.py
  python scripts/precompute_landmark_cache.py -o data/my_cache.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.landmark_vector import build_landmark_feature_cache, load_cycle_meta_json, write_landmark_cache_json
from lib.specimen_config import dy_from_config, load_specimen_config, resolve_path


def main() -> None:
    default_cfg = _ROOT / "config" / "specimen_config.yaml"
    p = argparse.ArgumentParser(description="Precompute landmark cache from experimental targets.")
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument(
        "--displacement",
        type=Path,
        default=None,
        help="CSV deformation row (default: <bayesian>/target_displacement.csv)",
    )
    p.add_argument(
        "--force",
        type=Path,
        default=None,
        help="CSV experimental force row (default: <bayesian>/target_force.csv)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: paths.landmark_cache from config)",
    )
    args = p.parse_args()

    cfg = load_specimen_config(args.config)
    base = args.config.parent
    meta_path = resolve_path(cfg, "cycle_meta", base)
    meta, s_f_m, s_d_m, _ = load_cycle_meta_json(meta_path)

    d_path = args.displacement or resolve_path(cfg, "target_displacement", base)
    f_path = args.force or resolve_path(cfg, "target_force", base)
    out_path = args.output or resolve_path(cfg, "landmark_cache", base)

    D = np.loadtxt(d_path, delimiter=",", dtype=np.float64).ravel()
    F_exp = np.loadtxt(f_path, delimiter=",", dtype=np.float64).ravel()
    if D.shape != F_exp.shape:
        raise ValueError(f"{d_path} length {D.size} != {f_path} length {F_exp.size}")

    dy = dy_from_config(cfg)
    s_d = float(s_d_m) if s_d_m is not None else None
    s_f = float(s_f_m) if s_f_m is not None else None

    cache = build_landmark_feature_cache(
        D,
        F_exp,
        meta,
        fy=float(cfg["fy_ksi"]),
        a_sc=float(cfg["A_sc_in2"]),
        dy=dy,
        s_d=s_d,
        s_f=s_f,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_landmark_cache_json(out_path, cache)
    print(f"Wrote {out_path} ({len(cache['cycles'])} cycles, n={cache['n']})")


if __name__ == "__main__":
    main()
