"""
Build ``results/calibration/individual_optimize/initial_brb_parameters.csv`` from the specimen catalog and
``extract_bn_bp.py`` output.

**SteelMPF** and apparent-**b** sourcing for each calibration ``set_id`` are read from
``config/calibration/set_id_settings.csv`` (unified per-``set_id`` config), unless
``--set-id-settings`` points elsewhere.

One CSV row per calibration ``set_id``. Columns ``E``, ``R0``, ``cR1``, ``cR2``, ``a1``–``a4`` are optional steel
overrides (blank → ``STEEL_DEFAULT``). Columns ``b_p`` and ``b_n`` are each either a **numeric** literal or a **stat
name** (case-insensitive) referencing ``specimen_apparent_bn_bp.csv``: ``median``, ``mean``, ``q1``, ``q3``, ``min``,
``max`` (segment-wise stats for path-ordered specimens; scatter-cloud rows often only have ``mean`` finite). Blank
``b_p`` / ``b_n`` default to ``median``. When a stat is missing or non-finite, fallbacks use other apparent-$b$
columns, then ``STEEL_DEFAULT["b_p"]`` / ``STEEL_DEFAULT["b_n"]``.

The settings CSV must exist (default: ``config/calibration/set_id_settings.csv``; override ``--set-id-settings``).

Optional per-set ``optimize_params`` in ``config/calibration/set_id_settings.csv`` does not affect this script; it only selects which
SteelMPF columns later stages optimize. Resolved ``b_p`` / ``b_n`` in the output CSV are the starting values
if those names appear in that file.

Geometry and ``fyp`` / ``fyn`` come from the catalog.

Run after ``resample_filtered.py`` and ``extract_bn_bp.py``.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from calibrate.calibration_paths import (  # noqa: E402
    BRB_SPECIMENS_CSV,
    INITIAL_BRB_PARAMETERS_PATH,
    SPECIMEN_APPARENT_BN_BP_PATH,
    SET_ID_SETTINGS_CSV,
)
from calibrate.set_id_settings import load_set_id_settings  # noqa: E402

CATALOG_PATH = BRB_SPECIMENS_CSV
DEFAULT_BN_BP_PATH = SPECIMEN_APPARENT_BN_BP_PATH
DEFAULT_OUTPUT_PATH = INITIAL_BRB_PARAMETERS_PATH

# SteelMPF columns in the seed CSV + default apparent-$b$ fallbacks (when stats are missing).
_STEEL_OVERRIDE_KEYS = frozenset({"E", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4"})
_STEEL_DEFAULT_ALL_KEYS = _STEEL_OVERRIDE_KEYS | frozenset({"b_p", "b_n"})

STEEL_DEFAULT: dict[str, float] = {
    "E": 29000.0,
    "R0": 20.0,
    "cR1": 0.925,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    "b_p": 0.01,
    "b_n": 0.025,
}

B_STAT_NAMES: frozenset[str] = frozenset(
    {"median", "mean", "weighted_mean", "q1", "q3", "min", "max"}
)

_BN_BP_STAT_COLUMNS: tuple[str, ...] = (
    "b_p_mean",
    "b_p_median",
    "b_p_weighted_mean",
    "b_p_q1",
    "b_p_q3",
    "b_p_min",
    "b_p_max",
    "b_n_mean",
    "b_n_median",
    "b_n_weighted_mean",
    "b_n_q1",
    "b_n_q3",
    "b_n_min",
    "b_n_max",
)


@dataclass(frozen=True)
class InitialBrbSeedRow:
    set_id: int
    steel: dict[str, float]
    b_p_spec: float | str
    b_n_spec: float | str


def _validate_steel_default(d: dict[str, float]) -> None:
    """Check default steel_seed_sets row."""
    keys = set(d)
    if keys != _STEEL_DEFAULT_ALL_KEYS:
        miss = sorted(_STEEL_DEFAULT_ALL_KEYS - keys)
        extra = sorted(keys - _STEEL_DEFAULT_ALL_KEYS)
        parts = []
        if miss:
            parts.append(f"missing {miss}")
        if extra:
            parts.append(f"extra {extra}")
        raise ValueError(f"STEEL_DEFAULT: {', '.join(parts)}; expected exactly {_STEEL_DEFAULT_ALL_KEYS}")


def _validate_steel_overrides(d: dict[str, float], *, context: str) -> None:
    """Check override rows for required columns."""
    keys = set(d)
    if not keys.issubset(_STEEL_OVERRIDE_KEYS):
        extra = sorted(keys - _STEEL_OVERRIDE_KEYS)
        raise ValueError(f"{context}: unknown keys {extra}; allowed {_STEEL_OVERRIDE_KEYS}")


def _row_set_id_label(row: pd.Series) -> str:
    """Human-readable label for a seed row."""
    return f"set_id={row.get('set_id', '?')!r}"


def _row_steel_overrides_from_series(row: pd.Series) -> dict[str, float]:
    """Steel fields overridden by one steel_seed_sets row."""
    out: dict[str, float] = {}
    for k in _STEEL_OVERRIDE_KEYS:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        out[k] = float(v)
    _validate_steel_overrides(out, context=f"steel seed row {_row_set_id_label(row)}")
    return out


def _merged_steel_from_overrides(ovr: dict[str, float]) -> dict[str, float]:
    """Merge seed overrides into a steel parameter dict."""
    return {k: STEEL_DEFAULT[k] for k in _STEEL_OVERRIDE_KEYS} | ovr


def _parse_b_spec(raw: object, *, label: str) -> float | str:
    """Parse b_p/b_n cell: float or statistic keyword."""
    if raw is None:
        return "median"
    if isinstance(raw, (float, np.floating)) and np.isnan(float(raw)):
        return "median"
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        v = float(raw)
        if np.isfinite(v):
            return v
    if pd.isna(raw):
        return "median"
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return "median"
    try:
        v = float(s)
        if np.isfinite(v):
            return v
    except ValueError:
        pass
    key = s.strip().lower().replace(" ", "_")
    aliases = {
        "1st_quartile": "q1",
        "first_quartile": "q1",
        "3rd_quartile": "q3",
        "third_quartile": "q3",
    }
    key = aliases.get(key, key)
    if key in B_STAT_NAMES:
        return key
    raise ValueError(f"{label}: expected a number or stat in {sorted(B_STAT_NAMES)}, got {raw!r}")


def _finite_scalar(x: float) -> bool:
    """Return float if finite, else NaN."""
    return isinstance(x, (int, float)) and bool(np.isfinite(float(x)))


def _get_bn_col(row: pd.Series, col: str) -> float:
    """Resolve apparent b_n / b_p column name from stats row."""
    if col not in row.index:
        return float("nan")
    v = row[col]
    if pd.isna(v):
        return float("nan")
    return float(v)


def _resolve_b_arm(row: pd.Series, *, arm: str, spec: float | str) -> float:
    """Resolve one b value from stats row and keyword."""
    if arm not in ("p", "n"):
        raise ValueError("arm must be 'p' or 'n'")
    if isinstance(spec, float):
        return float(spec)

    med = _get_bn_col(row, f"b_{arm}_median")
    mean = _get_bn_col(row, f"b_{arm}_mean")
    wmean = _get_bn_col(row, f"b_{arm}_weighted_mean")
    q1 = _get_bn_col(row, f"b_{arm}_q1")
    q3 = _get_bn_col(row, f"b_{arm}_q3")
    vmin = _get_bn_col(row, f"b_{arm}_min")
    vmax = _get_bn_col(row, f"b_{arm}_max")
    dflt = float(STEEL_DEFAULT["b_p"] if arm == "p" else STEEL_DEFAULT["b_n"])

    stat = spec
    if stat == "median":
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "mean":
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "weighted_mean":
        if _finite_scalar(wmean):
            return float(wmean)
        if _finite_scalar(mean):
            return float(mean)
        if _finite_scalar(med):
            return float(med)
        return dflt
    if stat == "q1":
        if _finite_scalar(q1):
            return float(q1)
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "q3":
        if _finite_scalar(q3):
            return float(q3)
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "min":
        if _finite_scalar(vmin):
            return float(vmin)
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "max":
        if _finite_scalar(vmax):
            return float(vmax)
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    raise ValueError(f"unknown b stat {stat!r}")


def _ensure_bn_bp_stat_columns(bn_bp: pd.DataFrame) -> pd.DataFrame:
    """Add apparent-b statistic columns if missing from table."""
    b = bn_bp.copy()
    for c in _BN_BP_STAT_COLUMNS:
        if c not in b.columns:
            b[c] = np.nan
    return b


def _load_seeds(df: pd.DataFrame, path: Path) -> list[InitialBrbSeedRow]:
    """Load set_id_settings.csv seed fields (skip comment lines)."""
    df = df.copy()
    if "set_id" not in df.columns:
        raise ValueError(f"{path}: seed CSV must have column 'set_id' (one row per calibration set)")
    for col in _STEEL_OVERRIDE_KEYS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ("b_p", "b_n"):
        if col not in df.columns:
            df[col] = np.nan
    seen: set[int] = set()
    rows: list[InitialBrbSeedRow] = []
    for i, row in df.iterrows():
        sid = row["set_id"]
        if pd.isna(sid):
            raise ValueError(f"{path}: row {i + 2}: set_id is required")
        try:
            set_id = int(pd.to_numeric(sid, errors="raise"))
        except (ValueError, TypeError) as e:
            raise ValueError(f"{path}: row {i + 2}: set_id must be an integer") from e
        if set_id < 1:
            raise ValueError(f"{path}: row {i + 2}: set_id must be >= 1")
        if set_id in seen:
            raise ValueError(f"{path}: duplicate set_id {set_id}")
        seen.add(set_id)
        ovr = _row_steel_overrides_from_series(row)
        steel = _merged_steel_from_overrides(ovr)
        bp = _parse_b_spec(row.get("b_p"), label=f"{path} row {i + 2} b_p")
        bn = _parse_b_spec(row.get("b_n"), label=f"{path} row {i + 2} b_n")
        rows.append(InitialBrbSeedRow(set_id, steel, bp, bn))
    rows.sort(key=lambda r: r.set_id)
    return rows


def load_initial_brb_seeds(path: Path) -> list[InitialBrbSeedRow]:
    """Parse ``set_id_settings.csv`` into one ``InitialBrbSeedRow`` per calibration ``set_id``."""
    df = load_set_id_settings(path)
    return _load_seeds(df, path)


OUT_COLS = [
    "ID",
    "Name",
    "set_id",
    "L_T",
    "L_y",
    "A_sc",
    "A_t",
    "fyp",
    "fyn",
    "E",
    "b_p",
    "b_n",
    "R0",
    "cR1",
    "cR2",
    "a1",
    "a2",
    "a3",
    "a4",
]


def build_initial_rows(
    catalog: pd.DataFrame,
    bn_bp: pd.DataFrame,
    *,
    seeds: list[InitialBrbSeedRow],
) -> list[dict]:
    """Assemble initial_brb_parameters rows from catalog and seeds."""
    _validate_steel_default(STEEL_DEFAULT)
    if not seeds:
        raise ValueError("initial BRB seeds list is empty; add rows to set_id_settings.csv")
    cat = catalog.copy()
    if "Name" not in cat.columns:
        raise ValueError("Catalog must have a Name column")
    cat["Name"] = cat["Name"].astype(str)

    bfull = _ensure_bn_bp_stat_columns(bn_bp)
    need_bn = ["Name", *_BN_BP_STAT_COLUMNS]
    missing = [c for c in need_bn if c not in bfull.columns]
    if missing:
        raise ValueError(f"bn_bp CSV missing columns: {missing}")
    bn = bfull[need_bn].copy()
    bn["Name"] = bn["Name"].astype(str)

    merged = cat.merge(bn, on="Name", how="left")
    if merged.empty:
        return []

    merged = merged.sort_values("ID" if "ID" in merged.columns else "Name")

    rows_out: list[dict] = []
    for _, r in merged.iterrows():
        name = str(r["Name"])
        cid = int(r["ID"])
        L_T = float(r["L_T_in"])
        L_y = float(r["L_y_in"])
        A_sc = float(r["A_c_in2"])
        A_t = float(r["A_t_in2"])
        fy = float(r["f_yc_ksi"])

        for seed in seeds:
            steel = seed.steel
            b_p = _resolve_b_arm(r, arm="p", spec=seed.b_p_spec)
            b_n = _resolve_b_arm(r, arm="n", spec=seed.b_n_spec)
            rows_out.append(
                {
                    "ID": cid,
                    "Name": name,
                    "set_id": seed.set_id,
                    "L_T": L_T,
                    "L_y": L_y,
                    "A_sc": A_sc,
                    "A_t": A_t,
                    "fyp": fy,
                    "fyn": fy,
                    "E": steel["E"],
                    "b_p": b_p,
                    "b_n": b_n,
                    "R0": steel["R0"],
                    "cR1": steel["cR1"],
                    "cR2": steel["cR2"],
                    "a1": steel["a1"],
                    "a2": steel["a2"],
                    "a3": steel["a3"],
                    "a4": steel["a4"],
                }
            )
    return rows_out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Build initial_brb_parameters.csv from catalog + extract_bn_bp output.")
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="BRB-Specimens.csv")
    p.add_argument(
        "--bn-bp",
        type=Path,
        default=DEFAULT_BN_BP_PATH,
        help="Output CSV from extract_bn_bp.py",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="initial_brb_parameters.csv path")
    p.add_argument(
        "--set-id-settings",
        type=Path,
        default=SET_ID_SETTINGS_CSV,
        help=(
            "set_id_settings.csv: one row per set_id (steel overrides + b_p/b_n as number or "
            "median/mean/q1/q3/min/max). Must exist. Default: config/calibration/set_id_settings.csv."
        ),
    )
    args = p.parse_args()

    catalog = pd.read_csv(args.catalog)
    if not args.bn_bp.is_file():
        print(f"Missing {args.bn_bp}; run extract_bn_bp.py after resample_filtered.py.")
        sys.exit(1)
    bn_bp = pd.read_csv(args.bn_bp)

    seed_path = Path(args.set_id_settings).expanduser().resolve()
    if not seed_path.is_file():
        print(f"Steel seed CSV not found: {seed_path}", file=sys.stderr)
        sys.exit(1)
    try:
        seeds = load_initial_brb_seeds(seed_path)
    except ValueError as e:
        print(e)
        sys.exit(1)

    try:
        rows = build_initial_rows(catalog, bn_bp, seeds=seeds)
    except ValueError as e:
        print(e)
        sys.exit(1)
    if not rows:
        print("No rows after catalog merge; check BRB-Specimens.csv.")
        sys.exit(1)

    out = pd.DataFrame(rows)
    out = out[OUT_COLS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    n_sp = out["Name"].nunique()
    n_sets = len(seeds)
    print(
        f"Wrote {args.output} ({len(out)} rows, {n_sp} specimens × {n_sets} set_ids). "
        f"Steel/b seeds: {seed_path}."
    )


if __name__ == "__main__":
    main()
