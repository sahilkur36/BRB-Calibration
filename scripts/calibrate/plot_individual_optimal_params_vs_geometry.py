"""
Scatter optimal individual-calibration parameters vs specimen geometry.

- Geometry: 12 metrics in a 3×4 panel grid (L_y, L_T, A_sc, L²/A_sc for L_y and L_T, E/f_y,
  Q, QE/f_y, then E A_sc/(f_y L²) and Q E A_sc/(f_y L²) for L_y and L_T). X-limits snap to
  metric-specific multiples (50, 10, 5, …) so every panel contains its scatter data.
- Steel parameters (generalized-train cohort only): R0, cR1, cR2, R0(1−cR1), a1, a3.
- b_p / b_n: train-only PNGs plus extended PNGs that mix optimal b for individually
  optimized non-train specimens and apparent (digitized) means for the rest. Extended
  figures overlay the train cohort mean and least-squares linear fit (legend: ``Mean (train)``,
  ``Linear fit (train)``). Y-limits use the same snapped range for a1 and a3 (and separately
  for b_p / b_n per cohort).

Optimal set per specimen: minimum final_J_feat_raw (J_feat L2) over successful metrics rows in the
requested set_id range, joined to optimized_brb_parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

# Multiplier on matplotlib text sizes (1.75 → 75% larger than baseline 10/11/9 pt).
FONT_SCALE = 1.75

# Scatter marker area (points²): base 18 × prior 1.75 × 1.30 (30% larger than before).
SCATTER_S = 18.0 * 1.75 * 1.3

PARAM_COLS_STEEL = ["R0", "cR1", "cR2", "a1", "a3"]
METRIC_PARAM_CHECK = ["R0", "cR1", "cR2", "a1", "a3", "b_p", "b_n", "E"]

_R0_1_MINUS_CR1_COL = "R0(1−cR1)"  # Unicode minus, matches DataFrame column name

# X-axis limit step per geometry column (limits = multiples of step, enclosing all data).
_X_STEP_BY_XKEY: dict[str, float] = {
    "L_y": 50.0,
    "L_T": 50.0,
    "A_sc": 10.0,
    "Ly2_over_A_sc": 5000.0,
    "LT2_over_A_sc": 5000.0,
    "E_div_fy": 50.0,
    "QE_div_fy": 50.0,
    "Q": 0.05,
    "E_Asc_over_fy_Ly2": 0.25,
    "E_Asc_over_fy_LT2": 0.25,
    "QE_Asc_over_fy_Ly2": 0.25,
    "QE_Asc_over_fy_LT2": 0.25,
}

# Y-axis limit step per response column (or pass shared y_limits for paired quantities).
_Y_STEP_BY_YCOL: dict[str, float] = {
    "a1": 0.01,
    "a3": 0.01,
    "b_p": 0.01,
    "b_n": 0.01,
    "cR1": 0.5,
    "cR2": 0.1,
    _R0_1_MINUS_CR1_COL: 1.0,
    "R0": 10.0,
}


def _limits_from_data_multiples(values: np.ndarray, step: float) -> tuple[float, float]:
    """Axis limits on multiples of ``step`` that contain all finite ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, step)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    if hi <= lo:
        hi = lo + step
    return (lo, hi)


def _three_ticks_inclusive(lo: float, hi: float) -> np.ndarray:
    """Three evenly spaced major ticks, endpoints equal to axis limits."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.asarray([lo, hi], dtype=float)
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    atol = max(span, abs(lo), abs(hi), 1.0) * 1e-12
    if span <= atol:
        return np.asarray([lo], dtype=float)
    return np.linspace(lo, hi, 3, dtype=float)


def _shared_y_limits_from_columns(
    df: pd.DataFrame, cols: list[str], step: float
) -> tuple[float, float]:
    parts: list[np.ndarray] = []
    for c in cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        parts.append(arr[np.isfinite(arr)])
    if not parts:
        return (0.0, step)
    merged = np.concatenate(parts)
    if merged.size == 0:
        return (0.0, step)
    return _limits_from_data_multiples(merged, step)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_set_range(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _read_csv_skip_hash(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_lines = [ln for ln in lines if not ln.strip().startswith("#")]
    from io import StringIO

    return pd.read_csv(StringIO("\n".join(data_lines)))


def _apply_font_scale() -> None:
    plt.rcParams.update(
        {
            "font.size": 10 * FONT_SCALE,
            "axes.labelsize": 11 * FONT_SCALE,
            "axes.titlesize": 12 * FONT_SCALE,
            "xtick.labelsize": 9 * FONT_SCALE,
            "ytick.labelsize": 9 * FONT_SCALE,
            "legend.fontsize": 9 * FONT_SCALE,
        }
    )


def _specimen_color_map(catalog: pd.DataFrame) -> dict[str, tuple]:
    """Catalog ID order, colors from matplotlib default ``axes.prop_cycle`` (tab10).

    Matches the marker colors in ``b_vs_geometry`` PNGs (verified vs
    ``bn_vs_geometry_all_mean.png``): default C0…C9 then repeat for specimens 11–16.
    """
    ordered = catalog.sort_values("ID")["Name"].astype(str).tolist()
    cycle_hex = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        name: mcolors.to_rgba(cycle_hex[i % len(cycle_hex)])
        for i, name in enumerate(ordered)
    }


def _legend_name_order(catalog: pd.DataFrame, present_names: set[str]) -> list[str]:
    return [
        n
        for n in catalog.sort_values("ID")["Name"].astype(str).tolist()
        if n in present_names
    ]


def _pick_optimal_parameter_rows(
    metrics: pd.DataFrame, optimized: pd.DataFrame, set_ids: list[int]
) -> pd.DataFrame:
    m = metrics[
        metrics["set_id"].isin(set_ids)
        & metrics["success"].astype(bool)
        & np.isfinite(metrics["final_J_feat_raw"])
    ].copy()
    if m.empty:
        raise ValueError("No successful metrics rows in the given set range.")

    need_cols = ["Name", "set_id", *METRIC_PARAM_CHECK]
    missing = [c for c in need_cols if c not in optimized.columns]
    if missing:
        raise KeyError(f"optimized_brb_parameters missing columns: {missing}")
    opt = optimized[need_cols].copy()

    merged = m.merge(opt, on=["Name", "set_id"], how="inner")
    for c in METRIC_PARAM_CHECK:
        merged = merged[np.isfinite(pd.to_numeric(merged[c], errors="coerce"))]

    merged = merged.sort_values(["Name", "final_J_feat_raw"])
    best = merged.groupby("Name", as_index=False).first()
    return best


def _resolve_Q(catalog_row: pd.Series, apparent_row: pd.Series | None) -> float:
    if apparent_row is not None:
        q = apparent_row.get("Q")
        if q is not None and pd.notna(q):
            return float(q)
    Asc = float(catalog_row["A_c_in2"])
    At = float(catalog_row["A_t_in2"])
    return 1.0 + At / Asc


def _resolve_E_kpsi(
    catalog_row: pd.Series,
    apparent_row: pd.Series | None,
    E_from_opt: float | None,
) -> float:
    if E_from_opt is not None and np.isfinite(E_from_opt):
        return float(E_from_opt)
    fy = float(catalog_row["f_yc_ksi"])
    if apparent_row is not None:
        fe = apparent_row.get("fy_over_E")
        if fe is not None and pd.notna(fe) and float(fe) > 0:
            return fy / float(fe)
    return 29000.0


def _geometry_features(
    catalog_row: pd.Series,
    E_kpsi: float,
    Q: float,
) -> dict[str, float]:
    Ly = float(catalog_row["L_y_in"])
    LT = float(catalog_row["L_T_in"])
    Asc = float(catalog_row["A_c_in2"])
    fy = float(catalog_row["f_yc_ksi"])
    if Asc <= 0:
        raise ValueError(f"Non-positive A_c_in2 for {catalog_row.get('Name')!r}")
    Ly2_A = Ly**2 / Asc
    LT2_A = LT**2 / Asc
    E_over_fy = E_kpsi / fy
    QE_over_fy = Q * E_over_fy
    Ly2 = Ly * Ly
    LT2 = LT * LT
    E_Asc_over_fy_Ly2 = (E_kpsi * Asc) / (fy * Ly2) if Ly2 else np.nan
    E_Asc_over_fy_LT2 = (E_kpsi * Asc) / (fy * LT2) if LT2 else np.nan
    QE_Asc_over_fy_Ly2 = Q * E_Asc_over_fy_Ly2 if np.isfinite(E_Asc_over_fy_Ly2) else np.nan
    QE_Asc_over_fy_LT2 = Q * E_Asc_over_fy_LT2 if np.isfinite(E_Asc_over_fy_LT2) else np.nan
    return {
        "L_y": Ly,
        "L_T": LT,
        "A_sc": Asc,
        # Length squared over core area (same units as L²/A when L [in], A [in²]).
        "Ly2_over_A_sc": Ly2_A,
        "LT2_over_A_sc": LT2_A,
        "E_div_fy": E_over_fy,
        "Q": Q,
        "QE_div_fy": QE_over_fy,
        "E_Asc_over_fy_Ly2": E_Asc_over_fy_Ly2,
        "E_Asc_over_fy_LT2": E_Asc_over_fy_LT2,
        "QE_Asc_over_fy_Ly2": QE_Asc_over_fy_Ly2,
        "QE_Asc_over_fy_LT2": QE_Asc_over_fy_LT2,
    }


# 12 panels → 3×4 grid. Axis text: [in], [in$^2$], [-] like apparent_b / b_vs_geometry.
GEOMETRY_LABELS: list[tuple[str, str]] = [
    (r"$L_y$ [in]", "L_y"),
    (r"$L_T$ [in]", "L_T"),
    (r"$A_{sc}$ [in$^2$]", "A_sc"),
    (r"$L_y^2/A_{sc}$ [-]", "Ly2_over_A_sc"),
    (r"$L_T^2/A_{sc}$ [-]", "LT2_over_A_sc"),
    (r"$E/f_y$ [-]", "E_div_fy"),
    (r"$Q$ [-]", "Q"),
    (r"$QE/f_y$ [-]", "QE_div_fy"),
    (r"$\frac{E A_{sc}}{f_y L_y^2}$ [-]", "E_Asc_over_fy_Ly2"),
    (r"$\frac{E A_{sc}}{f_y L_T^2}$ [-]", "E_Asc_over_fy_LT2"),
    (r"$\frac{Q E A_{sc}}{f_y L_y^2}$ [-]", "QE_Asc_over_fy_Ly2"),
    (r"$\frac{Q E A_{sc}}{f_y L_T^2}$ [-]", "QE_Asc_over_fy_LT2"),
]


def _build_frame_for_names(
    catalog: pd.DataFrame,
    best: pd.DataFrame,
    names: list[str],
    apparent: pd.DataFrame,
) -> pd.DataFrame:
    cat = catalog.set_index("Name")
    b = best.set_index("Name")
    app = apparent.set_index("Name")
    rows = []
    for name in names:
        if name not in cat.index or name not in b.index:
            continue
        crow = cat.loc[name]
        opt = b.loc[name]
        arow = app.loc[name] if name in app.index else None
        E = float(opt["E"])
        Q = _resolve_Q(crow, arow)
        g = _geometry_features(crow, E, Q)
        row = {"Name": name, **g}
        for p in PARAM_COLS_STEEL:
            row[p] = float(opt[p])
        row["b_p"] = float(opt["b_p"])
        row["b_n"] = float(opt["b_n"])
        row["R0(1−cR1)"] = float(opt["R0"]) * (1.0 - float(opt["cR1"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _montage_scatter(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    out_path: Path,
    *,
    name_to_color: dict[str, tuple],
    legend_names: list[str],
    square_marker_names: set[str] | None = None,
    y_limits: tuple[float, float] | None = None,
    train_overlay_df: pd.DataFrame | None = None,
) -> None:
    if df.empty:
        raise ValueError(f"No rows to plot for {y_col!r}")

    df = df.copy()
    df["_name"] = df["Name"].astype(str)
    y_all = pd.to_numeric(df[y_col], errors="coerce")
    mean_y = float(np.nanmean(y_all))

    y_step = _Y_STEP_BY_YCOL.get(y_col)
    if y_step is None:
        raise KeyError(f"No y-axis step for column {y_col!r}; add to _Y_STEP_BY_YCOL")
    if y_limits is not None:
        panel_ylim = y_limits
    else:
        panel_ylim = _limits_from_data_multiples(
            y_all.to_numpy(dtype=float), y_step
        )

    if len(GEOMETRY_LABELS) != 12:
        raise ValueError(f"Expected 12 geometry columns, got {len(GEOMETRY_LABELS)}")

    _nrows, _ncols = 3, 4
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(14, 9.0), layout="constrained")
    axes_flat = axes.ravel()

    mean_color = "0.35"
    fit_color = "crimson"
    train_mean_color = "mediumblue"
    train_fit_color = "steelblue"

    mean_train: float | None = None
    if train_overlay_df is not None:
        yt = pd.to_numeric(train_overlay_df[y_col], errors="coerce")
        m_tr = float(np.nanmean(yt))
        if np.isfinite(m_tr):
            mean_train = m_tr

    for idx, (ax, (xlab, xkey)) in enumerate(zip(axes_flat, GEOMETRY_LABELS)):
        x_all = pd.to_numeric(df[xkey], errors="coerce")
        y_all = pd.to_numeric(df[y_col], errors="coerce")
        m_all = np.isfinite(x_all.to_numpy()) & np.isfinite(y_all.to_numpy())
        xstep = _X_STEP_BY_XKEY.get(xkey)
        if xstep is None:
            raise KeyError(f"No x-axis step for geometry key {xkey!r}")
        xv_lim = x_all[m_all].to_numpy(dtype=float)
        x_lim_panel: tuple[float, float] | None = None
        if xv_lim.size:
            x_lim_panel = _limits_from_data_multiples(xv_lim, xstep)

        for name in legend_names:
            sub = df.loc[df["_name"] == name]
            if sub.empty:
                continue
            x = pd.to_numeric(sub[xkey], errors="coerce")
            yy = pd.to_numeric(sub[y_col], errors="coerce")
            m = np.isfinite(x.to_numpy()) & np.isfinite(yy.to_numpy())
            if not np.any(m):
                continue
            c = name_to_color.get(name, (0.5, 0.5, 0.5, 1.0))
            use_sq = square_marker_names is not None and name in square_marker_names
            ax.scatter(
                x[m],
                yy[m],
                s=SCATTER_S,
                c=[c],
                marker="s" if use_sq else "o",
                edgecolors="k",
                linewidths=0.5,
                alpha=0.85,
                zorder=2,
            )

        ax.axhline(
            mean_y,
            color=mean_color,
            linestyle="--",
            linewidth=0.9 * FONT_SCALE,
            zorder=0,
        )

        if m_all.sum() >= 2:
            xv = x_all[m_all].to_numpy(dtype=float)
            yv = y_all[m_all].to_numpy(dtype=float)
            if float(np.ptp(xv)) > 1e-15:
                slope, intercept = np.polyfit(xv, yv, 1)
                x_fit = np.linspace(float(np.min(xv)), float(np.max(xv)), 100)
                ax.plot(
                    x_fit,
                    slope * x_fit + intercept,
                    color=fit_color,
                    linestyle="-",
                    linewidth=1.05 * FONT_SCALE,
                    alpha=0.9,
                    zorder=1,
                )

        if train_overlay_df is not None and mean_train is not None:
            ax.axhline(
                mean_train,
                color=train_mean_color,
                linestyle="--",
                linewidth=0.9 * FONT_SCALE,
                zorder=0,
                alpha=0.95,
            )
            tx_all = pd.to_numeric(train_overlay_df[xkey], errors="coerce")
            ty_all = pd.to_numeric(train_overlay_df[y_col], errors="coerce")
            mt = np.isfinite(tx_all.to_numpy()) & np.isfinite(ty_all.to_numpy())
            if mt.sum() >= 2:
                txv = tx_all[mt].to_numpy(dtype=float)
                tyv = ty_all[mt].to_numpy(dtype=float)
                if float(np.ptp(txv)) > 1e-15:
                    tslope, tintercept = np.polyfit(txv, tyv, 1)
                    if x_lim_panel is not None:
                        xa, xb = x_lim_panel
                    else:
                        xa, xb = float(np.min(txv)), float(np.max(txv))
                    x_train_fit = np.linspace(xa, xb, 100)
                    ax.plot(
                        x_train_fit,
                        tslope * x_train_fit + tintercept,
                        color=train_fit_color,
                        linestyle="-",
                        linewidth=1.05 * FONT_SCALE,
                        alpha=0.9,
                        zorder=1,
                    )

        ax.set_xlabel(xlab)
        ax.grid(True, alpha=0.25)
        # Y tick numbers only on the left column (next to supylabel).
        ax.tick_params(axis="y", labelleft=(idx % _ncols == 0))
        y_lo, y_hi = panel_ylim
        ax.set_ylim(panel_ylim)
        ax.yaxis.set_major_locator(FixedLocator(_three_ticks_inclusive(y_lo, y_hi)))
        if x_lim_panel is not None:
            x_lo, x_hi = x_lim_panel
            ax.set_xlim((x_lo, x_hi))
            ax.xaxis.set_major_locator(FixedLocator(_three_ticks_inclusive(x_lo, x_hi)))

    fig.supylabel(y_label, fontsize=plt.rcParams["axes.labelsize"])

    leg_fs = plt.rcParams["legend.fontsize"]
    ms = float(np.sqrt(SCATTER_S))
    handles = [
        Line2D(
            [0],
            [0],
            marker="s"
            if square_marker_names is not None and n in square_marker_names
            else "o",
            linestyle="none",
            color="none",
            markerfacecolor=name_to_color.get(n, (0.5, 0.5, 0.5, 1.0)),
            markeredgecolor="k",
            markeredgewidth=0.5,
            markersize=ms,
            alpha=0.85,
            label=n,
        )
        for n in legend_names
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=mean_color,
                linestyle="--",
                linewidth=1.2 * FONT_SCALE,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                color=fit_color,
                linestyle="-",
                linewidth=1.2 * FONT_SCALE,
                label="Linear fit",
            ),
        ]
    )
    if train_overlay_df is not None:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=train_mean_color,
                    linestyle="--",
                    linewidth=1.2 * FONT_SCALE,
                    label="Mean (train)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=train_fit_color,
                    linestyle="-",
                    linewidth=1.2 * FONT_SCALE,
                    label="Linear fit (train)",
                ),
            ]
        )
    ncol = min(8, max(len(handles), 1))
    # Outside upper center: sits just above the axes; constrained layout reserves space.
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        fontsize=leg_fs,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


def _as_bool(val: object) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


def _digitized_unordered_names(catalog: pd.DataFrame) -> set[str]:
    """Specimens with digitized layout and path_ordered false (cloud / unordered path)."""
    names: set[str] = set()
    for _, row in catalog.iterrows():
        el = str(row.get("experimental_layout", "")).strip().lower()
        if el != "digitized":
            continue
        if _as_bool(row.get("path_ordered", True)):
            continue
        names.add(str(row["Name"]))
    return names


def _extended_bp_bn_frame(
    catalog: pd.DataFrame,
    best_by_name: pd.DataFrame,
    apparent: pd.DataFrame,
) -> pd.DataFrame:
    """All catalog specimens: optimal b for train + individually optimized; else apparent means."""
    cat = catalog.set_index("Name")
    best = best_by_name.set_index("Name")
    app = apparent.set_index("Name")

    rows = []
    for name in cat.index:
        row_cat = cat.loc[name]
        gw = int(float(row_cat["generalized_weight"]))
        io = _as_bool(row_cat["individual_optimize"])
        arow = app.loc[name] if name in app.index else None
        E_opt = float(best.loc[name]["E"]) if name in best.index else None
        E = _resolve_E_kpsi(row_cat, arow, E_opt)
        Q = _resolve_Q(row_cat, arow)
        g = _geometry_features(row_cat, E, Q)
        rec = {"Name": name, **g}

        if gw > 0 or io:
            if name not in best.index:
                continue
            rec["b_p"] = float(best.loc[name]["b_p"])
            rec["b_n"] = float(best.loc[name]["b_n"])
        else:
            if name not in app.index:
                continue
            bp = app.loc[name].get("b_p_mean")
            bn = app.loc[name].get("b_n_mean")
            if pd.isna(bp) or pd.isna(bn):
                continue
            rec["b_p"] = float(bp)
            rec["b_n"] = float(bn)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _repo_root()
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "config" / "calibration" / "BRB-Specimens.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=root
        / "results"
        / "calibration"
        / "individual_optimize"
        / "optimized_brb_parameters_metrics.csv",
    )
    parser.add_argument(
        "--optimized-params",
        type=Path,
        default=root
        / "results"
        / "calibration"
        / "individual_optimize"
        / "optimized_brb_parameters.csv",
    )
    parser.add_argument(
        "--apparent-bn-bp",
        type=Path,
        default=root / "results" / "calibration" / "specimen_apparent_bn_bp.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root
        / "results"
        / "plots"
        / "calibration"
        / "individual_optimize"
        / "optimal_params_vs_geometry",
    )
    parser.add_argument("--sets", type=str, default="1-10")
    args = parser.parse_args()

    _apply_font_scale()

    set_ids = _parse_set_range(args.sets)
    catalog = _read_csv_skip_hash(args.catalog)
    metrics = pd.read_csv(args.metrics)
    optimized = pd.read_csv(args.optimized_params)
    apparent = pd.read_csv(args.apparent_bn_bp)

    best = _pick_optimal_parameter_rows(metrics, optimized, set_ids)

    name_to_color = _specimen_color_map(catalog)

    gw = pd.to_numeric(catalog["generalized_weight"], errors="coerce").fillna(0)
    gw_pos = catalog.loc[gw > 0, "Name"].tolist()
    train_df = _build_frame_for_names(catalog, best, gw_pos, apparent)
    train_legend = _legend_name_order(
        catalog, set(train_df["Name"].astype(str).tolist())
    )

    ylim_a1_a3 = _shared_y_limits_from_columns(train_df, ["a1", "a3"], 0.01)
    ylim_bp_bn_train = _shared_y_limits_from_columns(
        train_df, ["b_p", "b_n"], 0.01
    )

    steel_specs: list[tuple[str, str, str]] = [
        ("R0", r"$R_0$ [-]", "individual_optimal_R0_vs_geometry.png"),
        ("cR1", r"$c_{R1}$ [-]", "individual_optimal_cR1_vs_geometry.png"),
        ("cR2", r"$c_{R2}$ [-]", "individual_optimal_cR2_vs_geometry.png"),
        ("R0(1−cR1)", r"$R_0(1-c_{R1})$ [-]", "individual_optimal_R0_1_minus_cR1_vs_geometry.png"),
        ("a1", r"$a_1$ [-]", "individual_optimal_a1_vs_geometry.png"),
        ("a3", r"$a_3$ [-]", "individual_optimal_a3_vs_geometry.png"),
    ]

    for col, ylab, fname in steel_specs:
        _montage_scatter(
            train_df,
            col,
            ylab,
            args.out_dir / fname,
            name_to_color=name_to_color,
            legend_names=train_legend,
            y_limits=ylim_a1_a3 if col in ("a1", "a3") else None,
        )

    # b_p / b_n: train-only (same cohort as steel plots)
    _montage_scatter(
        train_df,
        "b_p",
        r"$b_p$ [-]",
        args.out_dir / "individual_optimal_bp_vs_geometry.png",
        name_to_color=name_to_color,
        legend_names=train_legend,
        y_limits=ylim_bp_bn_train,
    )
    _montage_scatter(
        train_df,
        "b_n",
        r"$b_n$ [-]",
        args.out_dir / "individual_optimal_bn_vs_geometry.png",
        name_to_color=name_to_color,
        legend_names=train_legend,
        y_limits=ylim_bp_bn_train,
    )

    ext_df = _extended_bp_bn_frame(catalog, best, apparent)
    ext_legend = _legend_name_order(
        catalog, set(ext_df["Name"].astype(str).tolist())
    )
    ylim_bp_bn_ext = _shared_y_limits_from_columns(ext_df, ["b_p", "b_n"], 0.01)
    digitized_unordered = _digitized_unordered_names(catalog)
    _montage_scatter(
        ext_df,
        "b_p",
        r"$b_p$ [-]",
        args.out_dir / "individual_optimal_bp_vs_geometry_extended.png",
        name_to_color=name_to_color,
        legend_names=ext_legend,
        square_marker_names=digitized_unordered,
        y_limits=ylim_bp_bn_ext,
        train_overlay_df=train_df,
    )
    _montage_scatter(
        ext_df,
        "b_n",
        r"$b_n$ [-]",
        args.out_dir / "individual_optimal_bn_vs_geometry_extended.png",
        name_to_color=name_to_color,
        legend_names=ext_legend,
        square_marker_names=digitized_unordered,
        y_limits=ylim_bp_bn_ext,
        train_overlay_df=train_df,
    )


if __name__ == "__main__":
    main()
