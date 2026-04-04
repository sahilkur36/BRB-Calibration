"""
Build ``cycle_meta.json`` and single-row target drive CSVs from the resampled F–u history.

Reads ``paths.force_deformation`` (columns ``Deformation[in]``, ``Force[kip]`` — same grid as
``paths.cycle_points`` / ``n`` in the cycle JSON) and writes:

- ``paths.cycle_meta`` — cycle ranges and amplitude weights ``w_c``
- ``paths.target_displacement`` — one comma-separated row of deformation [in]
- ``paths.target_force`` — one comma-separated row of experimental force [kip]

Run from the ``bayesian/`` directory (or pass absolute paths):

  python scripts/setup_cycle_targets.py
  python scripts/setup_cycle_targets.py --output config/cycle_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.cycle_points_io import load_cycle_points_json
from lib.cycle_weights import build_amplitude_weights
from lib.jfeat_landmarks import deformation_scale_s_d
from lib.landmark_vector import force_scale_s_f, load_force_deformation_csv
from lib.specimen_config import load_specimen_config, resolve_path


def _target_csv_path(cfg: dict, base: Path, key: str, default_name: str) -> Path:
    rel = cfg.get("paths", {}).get(key)
    if rel:
        p = Path(rel)
        return p if p.is_absolute() else base / p
    return base / default_name


def main() -> None:
    default_cfg = _ROOT / "config" / "specimen_config.yaml"
    parser = argparse.ArgumentParser(
        description="Write cycle_meta.json and target_displacement/target_force CSVs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to specimen_config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="cycle_meta JSON (default: paths.cycle_meta from config)",
    )
    args = parser.parse_args()
    cfg = load_specimen_config(args.config)
    base = args.config.parent

    fd_path = resolve_path(cfg, "force_deformation", base)
    cp_path = resolve_path(cfg, "cycle_points", base)
    out_path = args.output or resolve_path(cfg, "cycle_meta", base)
    d_out = _target_csv_path(cfg, base, "target_displacement", "target_displacement.csv")
    f_out = _target_csv_path(cfg, base, "target_force", "target_force.csv")

    D, F = load_force_deformation_csv(fd_path)
    points, n_json = load_cycle_points_json(cp_path)
    if len(D) != n_json:
        raise ValueError(
            f"Length mismatch: force_deformation has {len(D)} rows, cycle JSON n={n_json}"
        )

    use_amp = bool(cfg.get("use_amplitude_weights", False))
    p = float(cfg.get("amplitude_weight_power", 2.0))
    eps = float(cfg.get("amplitude_weight_eps", 0.05))

    _w, meta = build_amplitude_weights(
        D,
        points,
        p=p,
        eps=eps,
        use_amplitude_weights=use_amp,
    )

    s_d = float(deformation_scale_s_d(D))
    s_f = float(force_scale_s_f(F))

    serializable_meta = []
    for m in meta:
        row = dict(m)
        for k, v in list(row.items()):
            if isinstance(v, (np.bool_,)):
                row[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
        serializable_meta.append(row)

    payload = {
        "specimen_id": cfg.get("specimen_id", ""),
        "n": int(len(D)),
        "s_d": s_d,
        "s_f": s_f,
        "use_amplitude_weights": use_amp,
        "amplitude_weight_power": p,
        "amplitude_weight_eps": eps,
        "meta": serializable_meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} ({len(serializable_meta)} cycles)")

    d_out.parent.mkdir(parents=True, exist_ok=True)
    f_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(d_out, D[np.newaxis, :], delimiter=",", fmt="%.17g")
    np.savetxt(f_out, F[np.newaxis, :], delimiter=",", fmt="%.17g")
    print(f"Wrote {d_out}")
    print(f"Wrote {f_out}")


if __name__ == "__main__":
    main()
