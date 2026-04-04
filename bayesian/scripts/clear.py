"""
Remove generated artifacts so the bayesian workflow can be re-run from scratch.

Deletes:

- Paths from ``specimen_config.yaml`` ``paths.*``: ``cycle_meta``, ``target_displacement``,
  ``target_force``, ``landmark_cache``
- Root outputs: ``calibration_data.csv``, ``results.out``, ``predicted_force.csv``,
  ``predicted_vs_calibration.png``

Does **not** delete inputs (e.g. ``data/force_deformation.csv``, cycle JSON, ``specimen_config.yaml``).

Run from the ``bayesian/`` directory:

  python scripts/clear.py
  python scripts/clear.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.specimen_config import load_specimen_config, resolve_path

# Default outputs not always listed under paths.* (fixed names in bayesian root).
_ROOT_OUTPUT_NAMES = (
    "calibration_data.csv",
    "results.out",
    "predicted_force.csv",
    "predicted_vs_calibration.png",
)

_PATH_KEYS = (
    "cycle_meta",
    "target_displacement",
    "target_force",
    "landmark_cache",
)


def _targets(cfg_path: Path) -> list[Path]:
    cfg = load_specimen_config(cfg_path)
    base = cfg_path.parent
    seen: set[Path] = set()
    out: list[Path] = []
    for key in _PATH_KEYS:
        try:
            p = resolve_path(cfg, key, base).resolve()
        except KeyError:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    for name in _ROOT_OUTPUT_NAMES:
        p = (_ROOT / name).resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> None:
    default_cfg = _ROOT / "config" / "specimen_config.yaml"
    p = argparse.ArgumentParser(description="Remove bayesian bundle generated outputs.")
    p.add_argument("--config", type=Path, default=default_cfg, help="specimen_config.yaml")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths that would be removed, do not delete",
    )
    args = p.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")

    removed = 0
    for path in _targets(args.config):
        if not path.is_file():
            continue
        if args.dry_run:
            print(f"would remove {path}")
        else:
            path.unlink()
            print(f"removed {path}")
        removed += 1

    if removed == 0:
        print("No generated files found to remove.")
    elif args.dry_run:
        print(f"(dry-run) {removed} file(s)")


if __name__ == "__main__":
    main()
