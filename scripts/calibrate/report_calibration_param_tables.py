"""
Write Markdown tables of ``PARAMS_IN_SUMMARY_TABLES`` (SteelMPF order: ``b_p``, ``b_n``, then ``PARAMS_TO_OPTIMIZE``, ``a2``, ``a4``)
by calibration ``set_id`` (default 1–10; match ``set_id_settings.csv``).

**Generalized** rows: one value per ``set_id`` for the optimized parameter subset (default ``PARAMS_TO_OPTIMIZE``; optional per-``set_id`` overrides via ``set_id_settings.csv``). ``b_p`` / ``b_n`` remain per-specimen in the merged CSV when they are not part of that shared vector, so their ``optimum_value`` column is left blank in that case.
The generalized Markdown table also appends **specimen-weighted mean** eval metrics per ``set_id`` (raw ``J_feat`` / ``J_E`` / ``J_binenv`` L2 and L1, ``J_total`` from ``generalized_params_eval_metrics.csv``).
**Individual** rows: mean across specimen rows with finite optimized parameters for that ``set_id``.

With ``--write report.md``, also writes CSV summaries next to the Markdown:
**Rollup (one row per parameter):** ``{stem}_generalized.csv``, ``{stem}_individual.csv`` — columns
``parameter``, ``mean``, ``min``, ``max`` across the selected sets (same definitions as the Markdown summary tables).
**By set_id (one row per set):** ``{stem}_generalized_by_set.csv`` (shared parameters + specimen-weighted eval
columns matching the Markdown generalized table), ``{stem}_individual_by_set.csv`` (mean optimized parameters per set).
The individual CSV also includes ``mean_optima``: for each specimen, the optimized
parameters from the ``set_id`` with lowest ``final_J_feat_raw`` (from the companion
``*_metrics.csv``), then the mean of those values across specimens.
It also includes ``mean_optima_weighted``: the same per-specimen vectors weighted by
``1 / (final_J_feat_raw + eps)`` at that best set (better fits count more); default ``eps``
is set via ``--weighted-optima-eps``.

The generalized CSV adds ``optimum_value``: for each **shared** optimized parameter, its value on the **generalized**
vector at the specimen-set-optimal ``set_id`` (among selected sets: minimum
specimen-weighted mean ``final_J_feat_raw`` over **contributing** rows, same rule as
``report_averaged_vs_generalized_metrics``). ``b_p`` and ``b_n`` are omitted (blank) because they are not
merged into a single value per ``set_id`` with the default optimized subset (unless ``b_p`` / ``b_n`` are included there via ``set_id_settings.csv``). Requires ``generalized_params_eval_metrics.csv``
(or ``--generalized-metrics``).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from calibrate.calibration_paths import (  # noqa: E402
    CALIBRATION_PARAMETER_SUMMARY_MD,
    GENERALIZED_BRB_PARAMETERS_PATH,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    SET_ID_SETTINGS_CSV,
)
from calibrate.params_to_optimize import PARAMS_IN_SUMMARY_TABLES  # noqa: E402

GENERALIZED_PER_SET_METRICS: list[tuple[str, str]] = [
    ("final_J_feat_raw", "J_feat"),
    ("final_J_feat_l1_raw", "J_feat_L1"),
    ("final_J_E_raw", "J_E"),
    ("final_J_E_l1_raw", "J_E_L1"),
    ("final_J_total", "J_total"),
    ("final_unordered_J_binenv", "J_binenv"),
    ("final_unordered_J_binenv_l1", "J_binenv_L1"),
]


def _parse_set_ids(spec: str) -> list[int]:
    """Parse '1-10' or '1,2,3' into sorted int list."""
    spec = spec.strip().lower().replace(" ", "")
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if hi < lo:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",") if x]


def _set_ids_from_set_id_settings(path: Path) -> list[int]:
    """Load set_id list from set_id_settings.csv (skip comment lines)."""
    if not path.is_file():
        raise SystemExit(f"Missing set_id settings CSV: {path}")
    df = pd.read_csv(path, comment="#")
    if "set_id" not in df.columns:
        raise SystemExit(f"set_id_settings.csv missing 'set_id' column: {path}")
    sid = pd.to_numeric(df["set_id"], errors="coerce")
    sid = sid[np.isfinite(sid)]
    out = sorted({int(x) for x in sid.to_numpy(dtype=float)})
    if not out:
        raise SystemExit(f"No set_id rows found in set_id_settings.csv: {path}")
    return out


def _fmt(x: float) -> str:
    """Format float for Markdown table (or em dash)."""
    if not np.isfinite(x):
        return "—"
    ax = abs(x)
    if ax != 0 and (ax >= 1e4 or ax < 1e-3):
        return f"{x:.6g}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """GitHub-flavored markdown table from headers and rows."""
    w = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    sep = "|" + "|".join("-" * (n + 2) for n in w) + "|"
    head = "| " + " | ".join(headers[i].ljust(w[i]) for i in range(len(headers))) + " |"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(r[i].ljust(w[i]) for i in range(len(r))) + " |")
    return "\n".join(lines)


def _generalized_by_set(df: pd.DataFrame, set_ids: list[int], params: list[str]) -> pd.DataFrame:
    """One row per set_id with generalized optimized parameters."""
    if "set_id" not in df.columns:
        raise SystemExit("generalized CSV: missing set_id column")
    sid = pd.to_numeric(df["set_id"], errors="coerce").astype("Int64")
    d = df.assign(_sid=sid)
    d = d[d["_sid"].isin(set_ids)]
    if d.empty:
        raise SystemExit("generalized CSV: no rows for requested set_id values")
    out_rows = []
    for s in set_ids:
        g = d[d["_sid"] == s]
        row: dict[str, object] = {"set_id": int(s)}
        if g.empty:
            for p in params:
                row[p] = float("nan")
        else:
            one = g.iloc[0]
            for p in params:
                row[p] = float(one[p]) if p in one.index and pd.notna(one[p]) else float("nan")
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _individual_means_by_set(
    df: pd.DataFrame, set_ids: list[int], params: list[str]
) -> pd.DataFrame:
    """Mean of finite optimized parameters per set_id."""
    if "set_id" not in df.columns:
        raise SystemExit("individual CSV: missing set_id column")
    sid = pd.to_numeric(df["set_id"], errors="coerce").astype("Int64")
    d = df.assign(_sid=sid)
    d = d[d["_sid"].isin(set_ids)]
    have = [p for p in params if p in d.columns]
    if len(have) != len(params):
        missing = [p for p in params if p not in d.columns]
        raise SystemExit(f"individual CSV: missing columns {missing}")
    ok = d[params].notna().all(axis=1)
    d_opt = d.loc[ok]
    if d_opt.empty:
        raise SystemExit("individual CSV: no rows with all optimized parameters finite")
    out_rows = []
    for s in set_ids:
        g = d_opt[d_opt["_sid"] == s]
        row: dict[str, object] = {"set_id": int(s)}
        if g.empty:
            for p in params:
                row[p] = float("nan")
        else:
            for p in params:
                row[p] = float(g[p].astype(float).mean())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _individual_best_row_per_specimen_by_cost(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    set_ids: list[int],
    params: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    One row per specimen (Name): parameters from the set_id in ``set_ids`` with
    minimum ``final_J_feat_raw``, among rows with finite optimized parameters and finite cost.

    Returns the parameters table and ``j_best``, aligned row-wise: ``final_J_feat_raw`` at
    that best (Name, set_id) for each specimen.
    """
    if "Name" not in df.columns:
        raise SystemExit("individual CSV: missing Name column")
    if "set_id" not in df.columns:
        raise SystemExit("individual CSV: missing set_id column")
    for col in ("Name", "set_id", "final_J_feat_raw"):
        if col not in metrics.columns:
            raise SystemExit(f"individual metrics CSV: missing {col!r} column")
    sid = pd.to_numeric(df["set_id"], errors="coerce").astype("Int64")
    d = df.assign(_sid=sid)
    d = d[d["_sid"].isin(set_ids)]
    have = [p for p in params if p in d.columns]
    if len(have) != len(params):
        missing = [p for p in params if p not in d.columns]
        raise SystemExit(f"individual CSV: missing columns {missing}")
    ok = d[params].notna().all(axis=1)
    d_opt = d.loc[ok]
    if d_opt.empty:
        raise SystemExit("individual CSV: no rows with all optimized parameters finite")
    m = metrics[["Name", "set_id", "final_J_feat_raw"]].copy()
    m["set_id"] = pd.to_numeric(m["set_id"], errors="coerce").astype("Int64")
    merged = d_opt.merge(m, on=["Name", "set_id"], how="left")
    cost = pd.to_numeric(merged["final_J_feat_raw"], errors="coerce")
    merged = merged.assign(_cost=cost)
    merged = merged[np.isfinite(merged["_cost"])]
    if merged.empty:
        raise SystemExit(
            "individual + metrics: no rows with finite final_J_feat_raw after merge"
        )
    best_idx: list[object] = []
    for _, g in merged.groupby("Name", sort=False):
        best_idx.append(g["_cost"].idxmin())
    best = merged.loc[best_idx]
    j_best = best["_cost"].to_numpy(dtype=float)
    return best[params].reset_index(drop=True), j_best


def _as_bool_series(s: pd.Series) -> pd.Series:
    """Coerce a Series to bool (accepts true/1/yes)."""
    if s.dtype == object:
        return s.map(lambda v: str(v).lower() in ("true", "1", "yes")).astype(bool)
    return s.astype(bool)


def _contributing_mask(metrics_df: pd.DataFrame) -> pd.Series:
    """Same as ``report_averaged_vs_generalized_metrics._contributing_mask`` (no OpenSees import)."""
    jt = pd.to_numeric(metrics_df["final_J_feat_raw"], errors="coerce")
    return (
        _as_bool_series(metrics_df["contributes_to_aggregate"])
        & _as_bool_series(metrics_df["success"])
        & jt.notna()
        & np.isfinite(jt)
    )


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Weighted average; returns NaN if weight sum is zero."""
    w = np.asarray(weights, dtype=float)
    x = np.asarray(series, dtype=float)
    sw = float(np.sum(w))
    if sw <= 0.0 or not np.isfinite(sw):
        return float("nan")
    return float(np.sum(x * w) / sw)


def _aggregate_by_set_metrics(
    df: pd.DataFrame, mask: pd.Series, metrics: list[str]
) -> pd.DataFrame:
    """Weighted mean per ``set_id`` (same pattern as ``report_averaged_vs_generalized_metrics``)."""
    sub = df.loc[mask, ["set_id", "specimen_weight", *metrics]].copy()
    rows: list[dict[str, object]] = []
    for sid, g in sub.groupby("set_id", sort=True):
        w = pd.to_numeric(g["specimen_weight"], errors="coerce").fillna(0.0)
        rec: dict[str, object] = {"set_id": int(sid)}
        for m in metrics:
            rec[m] = _weighted_mean(g[m], w)
        rows.append(rec)
    return pd.DataFrame(rows)


def _best_overall_set_id_from_agg(agg: pd.DataFrame) -> tuple[int | None, float]:
    """Best set_id and cost from one-row-per-set aggregate."""
    if agg.empty or "final_J_feat_raw" not in agg.columns:
        return None, float("nan")
    i = agg["final_J_feat_raw"].idxmin()
    return int(agg.loc[i, "set_id"]), float(agg.loc[i, "final_J_feat_raw"])


def _generalized_agg_metrics_for_selected_sets(
    metrics_df: pd.DataFrame, set_ids: list[int], metric_cols: list[str]
) -> pd.DataFrame:
    """
    One row per ``set_id`` that has contributing rows: specimen-weighted mean of each metric.
    Only ``set_id`` in ``set_ids``; only rows passing ``_contributing_mask`` are used.
    """
    base = ("set_id", "specimen_weight", "contributes_to_aggregate", "success")
    for c in (*base, *metric_cols):
        if c not in metrics_df.columns:
            raise SystemExit(f"generalized metrics CSV: missing {c!r} column")
    allowed = {int(x) for x in set_ids}
    mask = _contributing_mask(metrics_df)
    sid = pd.to_numeric(metrics_df["set_id"], errors="coerce")
    mask = mask & sid.isin(list(allowed))
    if not mask.any():
        return pd.DataFrame(columns=["set_id", *metric_cols])
    mdf = metrics_df.copy()
    for col in metric_cols:
        mdf[col] = pd.to_numeric(mdf[col], errors="coerce")
    return _aggregate_by_set_metrics(mdf, mask, metric_cols)


def _metric_cells_for_set_ids(
    agg: pd.DataFrame, set_ids: list[int], metric_cols: list[str]
) -> list[list[str]]:
    """Markdown cells per requested ``set_id``, same order as ``j_by``."""
    if agg.empty or not metric_cols:
        return [["—"] * len(metric_cols) for _ in set_ids]
    idx = agg.set_index("set_id")
    rows: list[list[str]] = []
    for s in set_ids:
        sid = int(s)
        if sid not in idx.index:
            rows.append(["—"] * len(metric_cols))
            continue
        row = idx.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cells: list[str] = []
        for c in metric_cols:
            v = row[c] if c in row.index else float("nan")
            cells.append(_fmt(float(v)) if pd.notna(v) else "—")
        rows.append(cells)
    return rows


def summary_stats_df(by_set: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """One row per parameter; columns mean, min, max over per-set values (finite only)."""
    records: list[dict[str, object]] = []
    for p in params:
        v = by_set[p].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            records.append({"parameter": p, "mean": np.nan, "min": np.nan, "max": np.nan})
        else:
            records.append(
                {
                    "parameter": p,
                    "mean": float(np.mean(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                }
            )
    return pd.DataFrame.from_records(records)


def _summary_rows_markdown(summary: pd.DataFrame) -> list[list[str]]:
    """Markdown table rows for mean/min/max (and extras)."""
    rows: list[list[str]] = []
    has_mo = "mean_optima" in summary.columns
    has_mow = "mean_optima_weighted" in summary.columns
    has_ov = "optimum_value" in summary.columns
    for _, r in summary.iterrows():
        row = [
            str(r["parameter"]),
            _fmt(float(r["mean"])) if pd.notna(r["mean"]) else "—",
            _fmt(float(r["min"])) if pd.notna(r["min"]) else "—",
            _fmt(float(r["max"])) if pd.notna(r["max"]) else "—",
        ]
        if has_mo:
            mo = r["mean_optima"]
            row.append(_fmt(float(mo)) if pd.notna(mo) else "—")
            if has_mow:
                mow = r["mean_optima_weighted"]
                row.append(_fmt(float(mow)) if pd.notna(mow) else "—")
        elif has_ov:
            ov = r["optimum_value"]
            row.append(_fmt(float(ov)) if pd.notna(ov) else "—")
        rows.append(row)
    return rows


def build_report(
    *,
    individual_path: Path,
    individual_metrics_path: Path,
    generalized_path: Path,
    generalized_metrics_path: Path,
    set_ids: list[int],
    params: list[str],
    weighted_optima_eps: float,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Markdown report string plus summary and per-set DataFrames for CSV export."""
    ind_df = pd.read_csv(individual_path)
    metrics_df = pd.read_csv(individual_metrics_path)
    j_df = pd.read_csv(generalized_path)
    generalized_metrics_df = pd.read_csv(generalized_metrics_path)

    j_by = _generalized_by_set(j_df, set_ids, params)
    i_by = _individual_means_by_set(ind_df, set_ids, params)
    i_best, j_at_best = _individual_best_row_per_specimen_by_cost(
        ind_df, metrics_df, set_ids, params
    )

    metric_csv_cols = [c for c, _ in GENERALIZED_PER_SET_METRICS]
    metric_headers = [h for _, h in GENERALIZED_PER_SET_METRICS]
    agg_m = _generalized_agg_metrics_for_selected_sets(
        generalized_metrics_df, set_ids, metric_csv_cols
    )
    best_id, best_j = _best_overall_set_id_from_agg(agg_m)
    metric_cells = _metric_cells_for_set_ids(agg_m, set_ids, metric_csv_cols)

    lines: list[str] = [
        "# Optimized SteelMPF parameters by calibration set",
        "",
        "This file summarizes **`PARAMS_IN_SUMMARY_TABLES`** (SteelMPF order: ``b_p``, ``b_n``, then optimized steel, ``a2``, ``a4``) "
        "from the individual and generalized parameter CSVs. Values are shown for requested **set_id** values: "
        f"{', '.join(str(int(s)) for s in set_ids)}.",
        "",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Individual parameters CSV:** `{individual_path.as_posix()}`",
        f"- **Individual metrics CSV (cost):** `{individual_metrics_path.as_posix()}`",
        f"- **Generalized parameters CSV:** `{generalized_path.as_posix()}`",
        f"- **Generalized eval metrics CSV:** `{generalized_metrics_path.as_posix()}`",
        "",
        "## Generalized optimize — shared vector per set",
        "",
        "One row per **set_id**: the generalized optimizer assigns the same shared steel-parameter values to every "
        "specimen row for that set (default columns: **`PARAMS_TO_OPTIMIZE`**; optional per-`set_id` list in "
        "**`set_id_settings.csv`**). "
        "**`b_p` / `b_n`** in the merged CSV remain per-specimen when not in that shared vector; the summary "
        "**optimum_value** for those columns is then left blank.",
        "",
    ]
    if best_id is not None and np.isfinite(best_j):
        lines.append(
            f"**Best generalized set** among the selected ``set_id`` values (lowest specimen-weighted mean "
            f"``final_J_feat_raw`` over contributing rows—``contributes_to_aggregate``, ``success``, finite cost, "
            f"same rule as ``report_averaged_vs_generalized_metrics``): **`set_id` = {best_id}** "
            f"(mean ``J_feat`` = {_fmt(float(best_j))}). "
            "The **optimum_value** column in the summary table below uses that shared parameter vector."
        )
    else:
        lines.append(
            "**Best generalized set** could not be determined (no contributing rows in the eval metrics for the "
            "selected ``set_id`` list)."
        )
    lines.extend(
        [
            "",
            "Trailing columns are specimen-weighted means from ``generalized_params_eval_metrics.csv``: "
            "``J_feat``/``J_feat_L1`` → ``final_J_feat_raw``/``final_J_feat_l1_raw`` (cycle feature, L2/L1), "
            "``J_E``/``J_E_L1`` → ``final_J_E_raw``/``final_J_E_l1_raw`` (energy, L2/L1), "
            "``J_total`` → ``final_J_total``, "
            "``J_binenv``/``J_binenv_L1`` → ``final_unordered_J_binenv``/``final_unordered_J_binenv_l1`` (L2/L1).",
            "",
        ]
    )

    j_headers = ["set_id", *params, *metric_headers]
    j_rows = []
    for i, (_, r) in enumerate(j_by.iterrows()):
        j_rows.append(
            [str(int(r["set_id"]))]
            + [_fmt(float(r[p])) for p in params]
            + metric_cells[i]
        )
    lines.append(_table(j_headers, j_rows))
    lines.extend(
        [
            "",
            "### Generalized — mean, min, max across selected sets",
            "",
        ]
    )
    j_summary = summary_stats_df(j_by, params)
    opt_set = best_id
    ov_map: dict[str, float] = {}
    if opt_set is not None:
        jrow = j_by.loc[j_by["set_id"] == int(opt_set)]
        if not jrow.empty:
            jr = jrow.iloc[0]
            for p in params:
                v = float(jr[p]) if p in jr.index and pd.notna(jr[p]) else float("nan")
                ov_map[p] = v
        else:
            ov_map = {p: float("nan") for p in params}
    else:
        ov_map = {p: float("nan") for p in params}
    # b_p / b_n stay per-specimen when not part of the shared optimized subset for that set_id.
    for _p in ("b_p", "b_n"):
        if _p in ov_map:
            ov_map[_p] = float("nan")
    j_summary["optimum_value"] = j_summary["parameter"].map(ov_map)
    s_rows = _summary_rows_markdown(j_summary)
    lines.append(_table(["parameter", "mean", "min", "max", "optimum_value"], s_rows))

    lines.extend(
        [
            "",
            "## Individual optimize — mean per set (specimen set)",
            "",
            "For each **set_id**, values are the **mean** of optimized parameters over specimen rows "
            "that had a successful individual fit (finite values for every optimized column).",
            "",
        ]
    )
    i_headers = ["set_id", *params]
    i_rows = []
    for _, r in i_by.iterrows():
        i_rows.append([str(int(r["set_id"]))] + [_fmt(float(r[p])) for p in params])
    lines.append(_table(i_headers, i_rows))
    lines.extend(
        [
            "",
            "### Individual — mean, min, max across selected sets",
            "",
            "(Statistics apply to the **per-set means** in the table above, not raw per-specimen rows.)",
            "**mean_optima** is the unweighted mean of each parameter over specimens at that specimen's best "
            "`set_id` (minimum `final_J_feat_raw`). **mean_optima_weighted** uses weights "
            f"`1/(final_J_feat_raw + eps)` at that best set with `eps` = {weighted_optima_eps:g} "
            "(override with `--weighted-optima-eps`).",
            "",
        ]
    )
    i_summary = summary_stats_df(i_by, params)
    mo_map: dict[str, float] = {}
    mow_map: dict[str, float] = {}
    jv = np.asarray(j_at_best, dtype=float)
    j_eff = np.maximum(jv, 0.0)
    w = 1.0 / (j_eff + float(weighted_optima_eps))
    for p in params:
        v = i_best[p].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if np.any(ok):
            mo_map[p] = float(np.mean(v[ok]))
            mow_map[p] = _weighted_mean(
                pd.Series(v[ok], dtype=float), pd.Series(w[ok], dtype=float)
            )
        else:
            mo_map[p] = float("nan")
            mow_map[p] = float("nan")
    i_summary["mean_optima"] = i_summary["parameter"].map(mo_map)
    i_summary["mean_optima_weighted"] = i_summary["parameter"].map(mow_map)
    s2 = _summary_rows_markdown(i_summary)
    lines.append(
        _table(
            ["parameter", "mean", "min", "max", "mean_optima", "mean_optima_weighted"],
            s2,
        )
    )
    lines.append("")

    j_by_set = j_by.copy()
    if agg_m.empty or not all(c in agg_m.columns for c in metric_csv_cols):
        for _, hdr in GENERALIZED_PER_SET_METRICS:
            j_by_set[hdr] = np.nan
    else:
        rename_map = {c: h for c, h in GENERALIZED_PER_SET_METRICS}
        mpart = agg_m[["set_id", *metric_csv_cols]].rename(columns=rename_map)
        j_by_set = j_by.merge(mpart, on="set_id", how="left")

    return "\n".join(lines), j_summary, i_summary, j_by_set, i_by


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--individual-params",
        type=Path,
        default=OPTIMIZED_BRB_PARAMETERS_PATH,
        help="optimized_brb_parameters.csv (individual L-BFGS output).",
    )
    p.add_argument(
        "--individual-metrics",
        type=Path,
        default=None,
        help=(
            "optimized_brb_parameters_metrics.csv (final_J_feat_raw per Name/set_id). "
            "Default: same directory as --individual-params, stem + '_metrics.csv'."
        ),
    )
    p.add_argument(
        "--generalized-params",
        type=Path,
        default=GENERALIZED_BRB_PARAMETERS_PATH,
        help="generalized_brb_parameters.csv (generalized optimization output).",
    )
    p.add_argument(
        "--generalized-metrics",
        type=Path,
        default=None,
        help=(
            "generalized_params_eval_metrics.csv (specimen-set eval after generalized optimize). "
            "Default: same directory as --generalized-params, file name generalized_params_eval_metrics.csv."
        ),
    )
    p.add_argument(
        "--sets",
        type=str,
        default=None,
        help=(
            "set_id list, e.g. '1-10' or '1,2,3'. "
            "Default: all set_id values present in --set-id-settings (set_id_settings.csv)."
        ),
    )
    p.add_argument(
        "--set-id-settings",
        type=Path,
        default=SET_ID_SETTINGS_CSV,
        help=(
            "set_id_settings.csv path used to infer set_ids when --sets is omitted. "
            f"Default: {SET_ID_SETTINGS_CSV}."
        ),
    )
    p.add_argument(
        "--weighted-optima-eps",
        type=float,
        default=1e-9,
        metavar="EPS",
        help=(
            "Small positive constant for individual mean_optima_weighted: weights are "
            "1/(max(final_J_feat_raw,0)+eps) per specimen at that specimen's best set. Default: 1e-9."
        ),
    )
    _w_rel = CALIBRATION_PARAMETER_SUMMARY_MD
    try:
        _w_rel = CALIBRATION_PARAMETER_SUMMARY_MD.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    p.add_argument(
        "--write",
        type=Path,
        default=None,
        help=(
            "Write Markdown report to this path (UTF-8). Also writes in the same directory: "
            "{stem}_generalized.csv, {stem}_individual.csv (rollup by parameter); "
            "{stem}_generalized_by_set.csv, {stem}_individual_by_set.csv (one row per set_id). "
            f"Typical: {_w_rel}. "
            "Default: print Markdown to stdout only (no CSV)."
        ),
    )
    args = p.parse_args()

    ind = Path(args.individual_params).expanduser().resolve()
    j = Path(args.generalized_params).expanduser().resolve()
    j_metrics = (
        Path(args.generalized_metrics).expanduser().resolve()
        if args.generalized_metrics
        else j.with_name("generalized_params_eval_metrics.csv")
    )
    ind_metrics = (
        Path(args.individual_metrics).expanduser().resolve()
        if args.individual_metrics
        else ind.with_name(f"{ind.stem}_metrics.csv")
    )
    if not ind.is_file():
        raise SystemExit(f"Missing individual parameters CSV: {ind}")
    if not ind_metrics.is_file():
        raise SystemExit(f"Missing individual metrics CSV: {ind_metrics}")
    if not j.is_file():
        raise SystemExit(f"Missing generalized parameters CSV: {j}")
    if not j_metrics.is_file():
        raise SystemExit(f"Missing generalized eval metrics CSV: {j_metrics}")

    if args.sets:
        set_ids = _parse_set_ids(args.sets)
    else:
        set_ids = _set_ids_from_set_id_settings(Path(args.set_id_settings).expanduser().resolve())
    params = list(PARAMS_IN_SUMMARY_TABLES)
    text, j_summary, i_summary, j_by_set, i_by_set = build_report(
        individual_path=ind,
        individual_metrics_path=ind_metrics,
        generalized_path=j,
        generalized_metrics_path=j_metrics,
        set_ids=set_ids,
        params=params,
        weighted_optima_eps=float(args.weighted_optima_eps),
    )

    if args.write:
        out = Path(args.write).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        generalized_csv = out.with_name(f"{out.stem}_generalized.csv")
        indiv_csv = out.with_name(f"{out.stem}_individual.csv")
        generalized_by_set_csv = out.with_name(f"{out.stem}_generalized_by_set.csv")
        individual_by_set_csv = out.with_name(f"{out.stem}_individual_by_set.csv")
        md_footer = (
            "\n## Machine-readable summaries\n\n"
            f"**Rollup** (one row per parameter, mean/min/max across sets): `{generalized_csv.name}`, "
            f"`{indiv_csv.name}`. Generalized adds **`optimum_value`** (shared optimized parameters at the "
            f"specimen-set best `set_id`; `b_p` / `b_n` blank); individual adds **`mean_optima`** and "
            f"**`mean_optima_weighted`** (inverse-loss–weighted mean using `1/(final_J_feat_raw+eps)` at each "
            f"specimen's best set; default eps = {float(args.weighted_optima_eps):g}).\n\n"
            f"**By `set_id`** (wide tables, same numeric content as the Markdown set tables): "
            f"`{generalized_by_set_csv.name}` (parameters + specimen-weighted eval columns), "
            f"`{individual_by_set_csv.name}` (mean parameters per set).\n"
        )
        out.write_text(text.rstrip() + md_footer, encoding="utf-8")
        print(f"Wrote {out}")
        j_summary.to_csv(generalized_csv, index=False)
        i_summary.to_csv(indiv_csv, index=False)
        j_by_set.to_csv(generalized_by_set_csv, index=False)
        i_by_set.to_csv(individual_by_set_csv, index=False)
        print(f"Wrote {generalized_csv}")
        print(f"Wrote {indiv_csv}")
        print(f"Wrote {generalized_by_set_csv}")
        print(f"Wrote {individual_by_set_csv}")
    else:
        print(text)


if __name__ == "__main__":
    main()
