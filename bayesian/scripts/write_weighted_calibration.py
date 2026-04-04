"""
Write calibration_data.csv: weighted landmark row from experiment only (F_sim = F_exp).

Order matches model.py results.out when the same force history is used.

  python scripts/write_weighted_calibration.py
  python scripts/write_weighted_calibration.py -o calibration_data.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.landmark_vector import (
    load_cycle_meta_json,
    load_force_deformation_csv,
    weighted_landmark_vector_experiment,
    write_csv_single_row,
)
from lib.specimen_config import dy_from_config, load_specimen_config, resolve_path


def main() -> None:
    default_cfg = _ROOT / "config" / "specimen_config.yaml"
    parser = argparse.ArgumentParser(description="Write calibration_data.csv (weighted landmarks).")
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_ROOT / "calibration_data.csv",
        help="Output CSV (single row)",
    )
    args = parser.parse_args()

    cfg = load_specimen_config(args.config)
    base = args.config.parent
    fd_path = resolve_path(cfg, "force_deformation", base)
    meta_path = resolve_path(cfg, "cycle_meta", base)

    D, F = load_force_deformation_csv(fd_path)
    meta, s_f_meta, s_d_meta, n_meta = load_cycle_meta_json(meta_path)
    if n_meta is not None and len(D) != n_meta:
        raise ValueError(f"Length mismatch: n_meta={n_meta} len(D)={len(D)}")

    dy = dy_from_config(cfg)
    vec = weighted_landmark_vector_experiment(
        D,
        F,
        meta,
        fy=float(cfg["fy_ksi"]),
        a_sc=float(cfg["A_sc_in2"]),
        dy=dy,
        s_d=float(s_d_meta) if s_d_meta is not None else None,
        s_f=float(s_f_meta) if s_f_meta is not None else None,
    )
    write_csv_single_row(args.output, vec)
    print(f"Wrote {args.output} ({len(vec)} values)")


if __name__ == "__main__":
    main()
