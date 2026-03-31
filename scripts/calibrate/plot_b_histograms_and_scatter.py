"""
Plot histograms of b_n and b_p per specimen (with mean/median highlighted) and
scatter plots of resampled segment statistics vs geometry (fifth panel: ``Q = \\hat{E}/E``; ordered data
include ``Q`` from ``extract_bn_bp``).

**B vs geometry** (``results/plots/apparent_b/b_vs_geometry/``): **ordered** and **all** (resampled +
envelope) are written for each segment statistic (**mean**, **median**, **weighted_mean**, **Q1**).
**Digitized envelope** scatter uses a single $b$ per specimen (no segment mean/median/weighted_mean/Q1),
so only ``bn_vs_geometry_scatter.png`` and ``bp_vs_geometry_scatter.png`` (cohort mean + median baselines on
those figures). **All** files: ``bn_vs_geometry_all_{mean,median,weighted_mean,q1}.png`` (and ``bp_vs_geometry_*``).
Box plots: ``box_bn_vs_geometry.png`` / ``box_bp_vs_geometry.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from math import log10, floor
from matplotlib.ticker import MaxNLocator, FuncFormatter

sns.set_theme()
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS / "postprocess"))
sys.path.insert(0, str(_SCRIPTS))

from plot_dimensions import (  # noqa: E402
    SAVE_DPI,
    b_vs_geometry_rcparams,
    configure_matplotlib_style,
    figsize_for_grid,
    grid_montage_rcparams,
    style_axes_spines_and_ticks,
)
from calibrate.calibration_paths import SPECIMEN_APPARENT_BN_BP_PATH  # noqa: E402

configure_matplotlib_style()
from calibrate.digitized_unordered_bn import compute_envelope_bn_unordered  # noqa: E402
from extract_bn_bp import get_b_lists_one_specimen, get_specimens_with_resampled
from model.corotruss import compute_Q  # noqa: E402
from specimen_catalog import (  # noqa: E402
    force_deformation_unordered_csv_path,
    list_names_digitized_unordered,
    read_catalog,
)

E_ksi = 29000.0
PARAMS_CSV = SPECIMEN_APPARENT_BN_BP_PATH
_APPARENT = _PROJECT_ROOT / "results" / "plots" / "apparent_b"
HIST_DIR = _APPARENT / "b_histograms"
B_VS_GEOMETRY_DIR = _APPARENT / "b_vs_geometry"
# Outside legends: fewer columns so the figure does not need excessive width (specimen names wrap to more rows).
B_VS_GEOMETRY_LEGEND_NCOL = 7

SegmentStatLit = Literal["mean", "median", "weighted_mean", "q1"]
SEGMENT_STATS_ORDER: tuple[SegmentStatLit, ...] = ("mean", "median", "weighted_mean", "q1")

DEF_COL = "Deformation[in]"
FORCE_COL = "Force[kip]"


def _cohort_ref_value(y: np.ndarray, stat: SegmentStatLit) -> float:
    """Single reference level for horizontal guide (cohort aggregate of the same kind as ``stat``)."""
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if stat in ("mean", "weighted_mean"):
        return float(np.mean(arr))
    if stat == "median":
        return float(np.median(arr))
    return float(np.percentile(arr, 25))


def _stat_label_word(stat: SegmentStatLit) -> str:
    return {"mean": "Mean", "median": "Median", "weighted_mean": "Weighted mean", "q1": "Q1"}[stat]


def _cohort_legend_label(stat: SegmentStatLit) -> str:
    return {
        "mean": "Cohort mean",
        "median": "Cohort median",
        "weighted_mean": "Cohort weighted mean",
        "q1": "Cohort Q1",
    }[stat]


def _format_x_3sig(x, pos):
    """Format x with 3 significant digits, plain decimal (no scientific notation). Shared by scatter and box x-axes."""
    if x == 0:
        return "0"
    r = round(x, -int(floor(log10(abs(x)))) + 2)
    if r == int(r) and abs(r) >= 1:
        return str(int(r))
    s = f"{r:.6f}".rstrip("0").rstrip(".")
    return s or "0"


def plot_histogram_one_specimen(
    specimen_id: str,
    b_n_list: list[float],
    b_p_list: list[float],
    out_dir: Path,
) -> None:
    """One figure per specimen: two histograms (b_n, b_p) with mean and median lines."""
    x_min, x_max = 0.0, 0.06
    y_min, y_max = 0.0, 1.0
    bins = np.linspace(x_min, x_max, 21)
    fig, (ax_n, ax_p) = plt.subplots(1, 2, figsize=figsize_for_grid(1, 2), layout="constrained")
    for ax, values, label, color in [
        (ax_n, b_n_list, r"$b_n$ (compression)", "#0077BB"),
        (ax_p, b_p_list, r"$b_p$ (tension)", "#CC3311"),
    ]:
        if not values:
            ax.set_title(f"{label}\nNo data")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            style_axes_spines_and_ticks(ax)
            continue
        weights = np.ones_like(values, dtype=float) / len(values)
        ax.hist(values, bins=bins, weights=weights, color=color, alpha=0.6, edgecolor="white")
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        ax.axvline(mean_val, color="k", linestyle="-", linewidth=1.5, label=f"Mean = {mean_val:.4f}")
        ax.axvline(median_val, color="k", linestyle="--", linewidth=1.2, label=f"Median = {median_val:.4f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Relative frequency")
        ax.set_title(f"{specimen_id}: {label}")
        ax.legend()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        style_axes_spines_and_ticks(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{specimen_id}.png", dpi=SAVE_DPI)
    plt.close(fig)


def plot_histograms_all(
    specimens: list[str],
    b_data: dict[str, tuple[list[float], list[float]]],
    out_dir: Path,
) -> None:
    """One figure with a grid of subplots: one per specimen, each with b_n and b_p histograms.
    Shared x and y limits and fixed bin edges so bar widths are consistent.
    """
    n = len(specimens)
    if n == 0:
        return
    # Fixed axis limits and bins (same for all subplots)
    all_b: list[float] = []
    for b_n_list, b_p_list in b_data.values():
        all_b.extend(b_n_list)
        all_b.extend(b_p_list)
    if not all_b:
        return
    b_min, b_max = 0.0, 0.06
    n_bins = 20
    bins = np.linspace(b_min, b_max, n_bins + 1)
    y_max = 1.0

    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(grid_montage_rcparams()):
        fig, axs = plt.subplots(
            nrow, ncol, figsize=figsize_for_grid(nrow, ncol), layout="constrained", sharex=True, sharey=True
        )
        if n == 1:
            axs = np.array([[axs]])
        elif axs.ndim == 1:
            axs = axs.reshape(1, -1)
        for idx, specimen_id in enumerate(specimens):
            row, col = idx // ncol, idx % ncol
            ax = axs[row, col]
            b_n_list, b_p_list = b_data.get(specimen_id, ([], []))
            all_b_spec = b_n_list + b_p_list
            if not all_b_spec:
                ax.set_title(specimen_id)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(b_min, b_max)
                ax.set_ylim(0, y_max)
                style_axes_spines_and_ticks(ax)
                continue
            w_n = np.ones_like(b_n_list, dtype=float) / len(b_n_list) if b_n_list else None
            w_p = np.ones_like(b_p_list, dtype=float) / len(b_p_list) if b_p_list else None
            if b_n_list:
                ax.hist(b_n_list, bins=bins, weights=w_n, color="#0077BB", alpha=0.6)
            if b_p_list:
                ax.hist(b_p_list, bins=bins, weights=w_p, color="#CC3311", alpha=0.6)
            mean_n = np.mean(b_n_list) if b_n_list else np.nan
            median_n = np.median(b_n_list) if b_n_list else np.nan
            mean_p = np.mean(b_p_list) if b_p_list else np.nan
            median_p = np.median(b_p_list) if b_p_list else np.nan
            for val, ls in [(mean_n, "-"), (median_n, "--")]:
                if not np.isnan(val):
                    ax.axvline(val, color="#0077BB", linestyle=ls, linewidth=0.8)
            for val, ls in [(mean_p, "-"), (median_p, "--")]:
                if not np.isnan(val):
                    ax.axvline(val, color="#CC3311", linestyle=ls, linewidth=0.8)
            ax.set_title(specimen_id)
            ax.set_xlabel(r"$b$")
            ax.set_ylabel("Rel. freq.")
            ax.set_xlim(b_min, b_max)
            ax.set_ylim(0, y_max)
            ax.grid(True, alpha=0.3)
            style_axes_spines_and_ticks(ax)
        for j in range(n, nrow * ncol):
            r, c = j // ncol, j % ncol
            axs[r, c].set_visible(False)
        legend_handles = [
            Patch(facecolor="#0077BB", alpha=0.6, label=r"$b_n$ (compression)"),
            Patch(facecolor="#CC3311", alpha=0.6, label=r"$b_p$ (tension)"),
            Line2D([0], [0], color="k", linestyle="-", linewidth=1.5, label="Mean"),
            Line2D([0], [0], color="k", linestyle="--", linewidth=1.2, label="Median"),
        ]
        fig.legend(handles=legend_handles, loc="outside lower center", ncol=4)
        fig.savefig(out_dir / "all_specimens_b_histograms.png", dpi=SAVE_DPI)
        plt.close(fig)


def _geometry_augment_for_scatter(df: pd.DataFrame) -> pd.DataFrame:
    """Add sqrt ratios and combined metrics for the shared b-vs-geometry panel layout."""
    df = df.copy()
    if "L_T_in" in df.columns and "A_c_in2" in df.columns:
        df["L_T_sqrt_A_sc"] = df["L_T_in"] / np.sqrt(df["A_c_in2"])
    else:
        df["L_T_sqrt_A_sc"] = np.nan
    if "L_y_in" in df.columns and "A_c_in2" in df.columns:
        df["L_y_sqrt_A_sc"] = df["L_y_in"] / np.sqrt(df["A_c_in2"])
    else:
        df["L_y_sqrt_A_sc"] = np.nan
    df["E_over_fy"] = np.where(df["fy_over_E"].notna() & (df["fy_over_E"] != 0), 1.0 / df["fy_over_E"], np.nan)
    df["E_hat_over_fy"] = np.where(
        df["fy_over_E_hat"].notna() & (df["fy_over_E_hat"] != 0),
        1.0 / df["fy_over_E_hat"],
        np.nan,
    )
    denom = df["f_yc_ksi"] * (df["L_T_in"] ** 2)
    df["E_hat_A2_over_fy_LT"] = np.where(
        denom.notna() & (denom != 0),
        df["E_hat"] * df["A_c_in2"] / denom,
        np.nan,
    )
    denom_y = df["f_yc_ksi"] * (df.get("L_y_in", np.nan) ** 2)
    df["E_A2_over_fy_Ly"] = np.where(
        denom_y.notna() & (denom_y != 0),
        E_ksi * df["A_c_in2"] / denom_y,
        np.nan,
    )
    return df


# Panel layout shared by resampled (median/mean) and digitized (envelope) scatter figures.
SCATTER_PANELS_ORDERED: list[tuple[str, str, tuple[float, float] | None]] = [
    ("L_T_in", r"$L_T$ [in]", None),
    ("L_y_in", r"$L_y$ [in]", None),
    ("A_c_in2", r"$A_{sc}$ [in^2]", None),
    ("f_yc_ksi", r"$f_y$ [ksi]", None),
    ("Q", r"$Q=\hat{E}/E$", None),
    ("L_T_sqrt_A_sc", r"$L_T / \sqrt{A_{sc}}$", (30.0, 180.0)),
    ("L_y_sqrt_A_sc", r"$L_y / \sqrt{A_{sc}}$", (0.0, 150.0)),
    ("E_hat_over_fy", r"$\hat{E} / f_y$", (750.0, 1050.0)),
    ("E_over_fy", r"$E / f_y$", (600.0, 800.0)),
    ("E_hat_A2_over_fy_LT", r"$\hat{E} A_{sc} / (f_y L_T^2)$", (0.0, 0.5)),
    ("E_A2_over_fy_Ly", r"$E A_{sc} / (f_y L_y^2)$", (0.0, 0.8)),
]


def build_digitized_envelope_bn_table(project_root: Path | None = None) -> pd.DataFrame:
    """
    One row per scatter-cloud digitized specimen (both cloud CSVs present): catalog geometry plus
    ``Q``, ``E_hat``, ``fy_over_E``, ``fy_over_E_hat``, and envelope ``b_p`` / ``b_n`` from the
    F–u unordered samples (same definition as ``plot_b_slopes.plot_one_digitized_unordered`` / averaged envelope merge).
    """
    root = project_root if project_root is not None else _PROJECT_ROOT
    catalog = read_catalog()
    names = list_names_digitized_unordered(catalog)
    if not names:
        return pd.DataFrame()
    catalog_by_name = catalog.set_index("Name")
    rows: list[dict] = []
    for sid in names:
        fd_path = force_deformation_unordered_csv_path(sid, root)
        if not fd_path.is_file():
            continue
        df_fd = pd.read_csv(fd_path)
        if DEF_COL not in df_fd.columns or FORCE_COL not in df_fd.columns:
            continue
        u = df_fd[DEF_COL].to_numpy(dtype=float)
        F = df_fd[FORCE_COL].to_numpy(dtype=float)
        m = np.isfinite(u) & np.isfinite(F)
        u, F = u[m], F[m]
        if len(u) == 0:
            continue
        row_cat = catalog_by_name.loc[sid].to_dict()
        out_row = {k: v for k, v in row_cat.items()}
        if "Name" not in out_row:
            out_row["Name"] = sid
        L_T = float(row_cat["L_T_in"])
        L_y = float(row_cat["L_y_in"])
        A_sc = float(row_cat["A_c_in2"])
        A_t = float(row_cat["A_t_in2"])
        fy = float(row_cat["f_yc_ksi"])
        Q = float(compute_Q(L_T, L_y, A_sc, A_t))
        E_hat = Q * E_ksi
        fy_over_E = fy / E_ksi if E_ksi != 0 else float("nan")
        fy_over_E_hat = fy / E_hat if E_hat != 0 else float("nan")
        diag = compute_envelope_bn_unordered(
            u, F, L_T=L_T, L_y=L_y, A_sc=A_sc, A_t=A_t, f_yc=fy, E_ksi_val=E_ksi
        )
        out_row["Q"] = Q
        out_row["E_hat"] = E_hat
        out_row["fy_over_E"] = fy_over_E
        out_row["fy_over_E_hat"] = fy_over_E_hat
        out_row["b_n_envelope"] = diag.b_n
        out_row["b_p_envelope"] = diag.b_p
        rows.append(out_row)
    if not rows:
        return pd.DataFrame()
    out_df = pd.DataFrame(rows)
    catalog_cols = [c for c in catalog.columns if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in catalog_cols]
    return out_df[catalog_cols + rest]


def plot_scatter_bn_bp(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    segment_stat: SegmentStatLit = "median",
) -> None:
    """Scatter: one 3x4 figure per b_n and b_p; fifth panel is ``Q=\\hat E/E``; one segment stat per figure."""
    df = _geometry_augment_for_scatter(df)
    stat_label = _stat_label_word(segment_stat)

    # Color-code by specimen for scatter (consistent across panels)
    if "Name" in df.columns:
        specimens = sorted(df["Name"].astype(str).unique())
        cmap = plt.colormaps["tab10"]
        specimen_color = {s: cmap((i % 10) / 9.0) for i, s in enumerate(specimens)}
        specimen_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=specimen_color[s], markersize=8, markeredgewidth=0, label=s)
            for s in specimens
        ]
    else:
        specimen_color = {}
        specimen_handles = []

    for y_label, y_primary, _, stem in [
        ("b_n", f"b_n_{segment_stat}", r"$b_n$ (compression) vs geometry", "bn_vs_geometry_ordered"),
        ("b_p", f"b_p_{segment_stat}", r"$b_p$ (tension) vs geometry", "bp_vs_geometry_ordered"),
    ]:
        out_name = f"{stem}_{segment_stat}.png"
        color_median = "#0077BB" if y_label == "b_n" else "#CC3311"
        y_lim = (0.02, 0.05) if y_label == "b_n" else (0.0, 0.03)
        out_dir.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(b_vs_geometry_rcparams()):
            fig, axs = plt.subplots(3, 4, figsize=figsize_for_grid(3, 4), layout="constrained")
            for idx, (x_col, x_latex, x_lim) in enumerate(SCATTER_PANELS_ORDERED):
                row, col = idx // 4, idx % 4
                ax = axs[row, col]
                if x_col not in df.columns or y_primary not in df.columns:
                    style_axes_spines_and_ticks(ax)
                    continue
                valid = df[[x_col, y_primary]].notna().all(axis=1)
                if "Name" in df.columns:
                    valid = valid & df["Name"].notna()
                x = df.loc[valid, x_col]
                y_vals = df.loc[valid, y_primary]
                if len(y_vals) == 0:
                    style_axes_spines_and_ticks(ax)
                    continue
                ref_b = _cohort_ref_value(y_vals.to_numpy(dtype=float), segment_stat)
                if specimen_color:
                    names = df.loc[valid, "Name"].astype(str)
                    pt_colors = [specimen_color.get(n, color_median) for n in names]
                    ax.scatter(x, y_vals, c=pt_colors, s=60, marker="o", zorder=2, edgecolors="white", linewidths=0.5)
                else:
                    ax.scatter(x, y_vals, c=color_median, s=60, marker="o", zorder=2)
                ax.set_xlabel(x_latex)
                ax.set_ylabel(r"$b_n$" if y_label == "b_n" else r"$b_p$")
                if x_lim is not None:
                    ax.set_xlim(*x_lim)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FuncFormatter(_format_x_3sig))
                if np.isfinite(ref_b):
                    ax.axhline(ref_b, color="0.45", linestyle="-", linewidth=1.2)
                ax.set_ylim(*y_lim)
                ax.grid(True, alpha=0.3)
                style_axes_spines_and_ticks(ax)
            axs[2, 3].set_visible(False)
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_median,
                    markersize=8,
                    markeredgewidth=0,
                    label=f"Resampled {stat_label}",
                ),
                Line2D(
                    [0],
                    [0],
                    color="0.45",
                    linestyle="-",
                    linewidth=1.2,
                    label=f"{_cohort_legend_label(segment_stat)} baseline",
                ),
            ]
            if specimen_handles:
                legend_handles.extend(specimen_handles)
            fig.legend(
                handles=legend_handles,
                loc="outside lower center",
                ncol=min(B_VS_GEOMETRY_LEGEND_NCOL, len(legend_handles)),
            )
            fig.savefig(out_dir / out_name, dpi=SAVE_DPI)
            plt.close(fig)


def plot_scatter_bn_bp_digitized(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Same geometry panels as ``plot_scatter_bn_bp``, but one **envelope** $b$ per specimen
    (``b_n_envelope``, ``b_p_envelope`` from the digitized cloud)—no separate mean/median/weighted_mean/Q1 series.
    Writes ``bn_vs_geometry_scatter.png`` and ``bp_vs_geometry_scatter.png`` only. Cohort mean and
    median of envelope $b$ in each panel (two baselines, same plotted quantity).
    """
    if df.empty or "Name" not in df.columns:
        return
    df = _geometry_augment_for_scatter(df)
    if "Name" in df.columns:
        specimens = sorted(df["Name"].astype(str).unique())
        cmap = plt.colormaps["tab10"]
        specimen_color = {s: cmap((i % 10) / 9.0) for i, s in enumerate(specimens)}
        specimen_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=specimen_color[s],
                markersize=8,
                markeredgewidth=0,
                label=s,
            )
            for s in specimens
        ]
    else:
        specimen_color = {}
        specimen_handles = []

    for y_col, y_tex, _, out_name in [
        ("b_n_envelope", r"$b_n$", r"$b_n$ (compression, digitized envelope) vs geometry", "bn_vs_geometry_scatter.png"),
        ("b_p_envelope", r"$b_p$", r"$b_p$ (tension, digitized envelope) vs geometry", "bp_vs_geometry_scatter.png"),
    ]:
        if y_col not in df.columns:
            continue
        color_pt = "#0077BB" if y_col == "b_n_envelope" else "#CC3311"
        y_lim = (0.02, 0.05) if y_col == "b_n_envelope" else (0.0, 0.03)
        out_dir.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(b_vs_geometry_rcparams()):
            fig, axs = plt.subplots(3, 4, figsize=figsize_for_grid(3, 4), layout="constrained")
            for idx, (x_col, x_latex, x_lim) in enumerate(SCATTER_PANELS_ORDERED):
                row, col = idx // 4, idx % 4
                ax = axs[row, col]
                if x_col not in df.columns:
                    style_axes_spines_and_ticks(ax)
                    continue
                valid = df[x_col].notna() & df[y_col].notna() & df["Name"].notna()
                x = df.loc[valid, x_col]
                y_vals = df.loc[valid, y_col]
                y_arr = y_vals.to_numpy(dtype=float)
                if y_arr.size == 0:
                    mean_b = float("nan")
                    median_b = float("nan")
                else:
                    mean_b = float(np.mean(y_arr))
                    median_b = float(np.median(y_arr))
                names = df.loc[valid, "Name"].astype(str)
                pt_colors = [specimen_color.get(n, color_pt) for n in names]
                ax.scatter(
                    x,
                    y_vals,
                    c=pt_colors,
                    s=68,
                    marker="s",
                    zorder=2,
                    edgecolors="white",
                    linewidths=0.5,
                )
                ax.set_xlabel(x_latex)
                ax.set_ylabel(y_tex)
                if x_lim is not None:
                    ax.set_xlim(*x_lim)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FuncFormatter(_format_x_3sig))
                if np.isfinite(mean_b):
                    ax.axhline(mean_b, color="0.4", linestyle=":", linewidth=1.2)
                if np.isfinite(median_b):
                    ax.axhline(median_b, color="0.5", linestyle="-", linewidth=1)
                ax.set_ylim(*y_lim)
                ax.grid(True, alpha=0.3)
                style_axes_spines_and_ticks(ax)
            axs[2, 3].set_visible(False)
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=color_pt,
                    markersize=8,
                    markeredgewidth=0,
                    label="Envelope $b$",
                ),
                Line2D([0], [0], color="0.4", linestyle=":", linewidth=1.5, label="Cohort mean"),
                Line2D([0], [0], color="0.5", linestyle="-", linewidth=1.2, label="Cohort median"),
            ]
            legend_handles.extend(specimen_handles)
            fig.legend(
                handles=legend_handles,
                loc="outside lower center",
                ncol=min(B_VS_GEOMETRY_LEGEND_NCOL, len(legend_handles)),
            )
            fig.savefig(out_dir / out_name, dpi=SAVE_DPI)
            plt.close(fig)


def plot_scatter_bn_bp_combined(
    df_seg: pd.DataFrame,
    df_env: pd.DataFrame,
    out_dir: Path,
    *,
    segment_stat: SegmentStatLit = "median",
) -> None:
    """
    One 3x4 figure per b_n and b_p: resampled segment stat (circles) and digitized envelope b (squares).
    One cohort baseline per figure, matching ``segment_stat`` over all plotted y values in each panel.
    """
    if df_env.empty:
        return
    df_seg = _geometry_augment_for_scatter(df_seg.copy())
    df_env = _geometry_augment_for_scatter(df_env.copy())
    names_seg = set(df_seg["Name"].dropna().astype(str)) if "Name" in df_seg.columns else set()
    names_env = set(df_env["Name"].dropna().astype(str)) if "Name" in df_env.columns else set()
    specimens = sorted(names_seg | names_env)
    if not specimens:
        return
    cmap = plt.colormaps["tab10"]
    specimen_color = {s: cmap((i % 10) / 9.0) for i, s in enumerate(specimens)}
    specimen_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=specimen_color[s], markersize=8, markeredgewidth=0, label=s)
        for s in specimens
    ]
    seg_label = f"Resampled {_stat_label_word(segment_stat)}"

    for y_label, y_seg_col, y_env_col, _, stem in [
        (
            "b_n",
            f"b_n_{segment_stat}",
            "b_n_envelope",
            r"$b_n$ (compression): ordered",
            "bn_vs_geometry_all",
        ),
        (
            "b_p",
            f"b_p_{segment_stat}",
            "b_p_envelope",
            r"$b_p$ (tension): ordered",
            "bp_vs_geometry_all",
        ),
    ]:
        if y_seg_col not in df_seg.columns or y_env_col not in df_env.columns:
            continue
        out_name = f"{stem}_{segment_stat}.png"
        color_ord = "#0077BB" if y_label == "b_n" else "#CC3311"
        y_lim = (0.02, 0.05) if y_label == "b_n" else (0.0, 0.03)
        out_dir.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(b_vs_geometry_rcparams()):
            fig, axs = plt.subplots(3, 4, figsize=figsize_for_grid(3, 4), layout="constrained")
            for idx, (x_col, x_latex, x_lim) in enumerate(SCATTER_PANELS_ORDERED):
                row, col = idx // 4, idx % 4
                ax = axs[row, col]
                if x_col not in df_seg.columns or x_col not in df_env.columns:
                    style_axes_spines_and_ticks(ax)
                    continue
                ys_all: list[float] = []

                v_seg = df_seg[[x_col, y_seg_col]].notna().all(axis=1)
                if "Name" in df_seg.columns:
                    v_seg = v_seg & df_seg["Name"].notna()
                if v_seg.any():
                    x_s = df_seg.loc[v_seg, x_col]
                    y_s = df_seg.loc[v_seg, y_seg_col]
                    ns = df_seg.loc[v_seg, "Name"].astype(str)
                    c_s = [specimen_color.get(n, color_ord) for n in ns]
                    ax.scatter(
                        x_s,
                        y_s,
                        c=c_s,
                        s=60,
                        marker="o",
                        zorder=2,
                        edgecolors="white",
                        linewidths=0.5,
                    )
                    ys_all.extend(float(v) for v in y_s)

                v_env = df_env[[x_col, y_env_col]].notna().all(axis=1)
                if "Name" in df_env.columns:
                    v_env = v_env & df_env["Name"].notna()
                if v_env.any():
                    x_e = df_env.loc[v_env, x_col]
                    y_e = df_env.loc[v_env, y_env_col]
                    ne = df_env.loc[v_env, "Name"].astype(str)
                    c_e = [specimen_color.get(n, color_ord) for n in ne]
                    ax.scatter(
                        x_e,
                        y_e,
                        c=c_e,
                        s=68,
                        marker="s",
                        zorder=3,
                        edgecolors="white",
                        linewidths=0.5,
                    )
                    ys_all.extend(float(v) for v in y_e)

                ax.set_xlabel(x_latex)
                ax.set_ylabel(r"$b_n$" if y_label == "b_n" else r"$b_p$")
                if x_lim is not None:
                    ax.set_xlim(*x_lim)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FuncFormatter(_format_x_3sig))
                if ys_all:
                    mean_b = float(np.mean(ys_all))
                    median_b = float(np.median(ys_all))
                    ax.axhline(mean_b, color="0.4", linestyle=":", linewidth=1.2)
                    ax.axhline(median_b, color="0.5", linestyle="-", linewidth=1)
                ax.set_ylim(*y_lim)
                ax.grid(True, alpha=0.3)
                style_axes_spines_and_ticks(ax)

            axs[2, 3].set_visible(False)
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_ord,
                    markersize=8,
                    markeredgewidth=0,
                    label=seg_label,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=color_ord,
                    markersize=8,
                    markeredgewidth=0,
                    label="Envelope (digitized)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="0.45",
                    linestyle="-",
                    linewidth=1.2,
                    label=f"{_stat_label_word(segment_stat)} (plotted pts)",
                ),
            ]
            legend_handles.extend(specimen_handles)
            fig.legend(
                handles=legend_handles,
                loc="outside lower center",
                ncol=min(B_VS_GEOMETRY_LEGEND_NCOL, len(legend_handles)),
            )
            fig.savefig(out_dir / out_name, dpi=SAVE_DPI)
            plt.close(fig)


def plot_box_bn_bp(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Box-and-whisker version: same 3 geometry variables; one box per specimen, per geometry variable."""
    df = df.copy()
    if "L_T_in" in df.columns and "A_c_in2" in df.columns:
        df["L_T_sqrt_A_sc"] = df["L_T_in"] / np.sqrt(df["A_c_in2"])
    else:
        df["L_T_sqrt_A_sc"] = np.nan
    # Alternate geometry based on L_y
    if "L_y_in" in df.columns and "A_c_in2" in df.columns:
        df["L_y_sqrt_A_sc"] = df["L_y_in"] / np.sqrt(df["A_c_in2"])
    else:
        df["L_y_sqrt_A_sc"] = np.nan
    df["E_hat_over_fy"] = np.where(
        df["fy_over_E_hat"].notna() & (df["fy_over_E_hat"] != 0),
        1.0 / df["fy_over_E_hat"],
        np.nan,
    )
    denom = df["f_yc_ksi"] * (df["L_T_in"] ** 2)
    df["E_hat_A2_over_fy_LT"] = np.where(
        denom.notna() & (denom != 0),
        df["E_hat"] * df["A_c_in2"] / denom,
        np.nan,
    )
    # Alternate metric with E and L_y: E A / (f_y L_y^2) using constant E_ksi
    denom_y = df["f_yc_ksi"] * (df.get("L_y_in", np.nan) ** 2)
    df["E_A2_over_fy_Ly"] = np.where(
        denom_y.notna() & (denom_y != 0),
        E_ksi * df["A_c_in2"] / denom_y,
        np.nan,
    )

    if "fy_over_E" in df.columns:
        df["E_over_fy"] = np.where(
            df["fy_over_E"].notna() & (df["fy_over_E"] != 0),
            1.0 / df["fy_over_E"],
            np.nan,
        )
    else:
        df["E_over_fy"] = np.nan
    # One box per specimen: load full b_n/b_p lists
    if "Name" not in df.columns:
        return
    specimen_ids = df["Name"].astype(str).tolist()
    from extract_bn_bp import get_b_lists_one_specimen as _get_b_lists_spec
    b_data: dict[str, tuple[list[float], list[float]]] = {
        sid: _get_b_lists_spec(sid) for sid in specimen_ids
    }
    # Same specimen color scheme as scatter (tab10)
    specimens = sorted(df["Name"].astype(str).unique())
    cmap = plt.colormaps["tab10"]
    specimen_color = {s: cmap((i % 10) / 9.0) for i, s in enumerate(specimens)}
    fallback_color = "#0077BB"  # used only if a specimen is missing from map
    specimen_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=specimen_color[s], markersize=8, markeredgewidth=0, label=s)
        for s in specimens
    ]

    def _draw_box_panel(ax, df, b_data, x_col, x_latex, y_label, specimen_color_map, x_lim=None):
        """Draw one box/violin panel for apparent-b statistics."""
        if x_col not in df.columns:
            style_axes_spines_and_ticks(ax)
            return
        xs = []
        vals_per_spec = []
        sids = []
        for _, row in df[[x_col, "Name"]].iterrows():
            x_val = float(row[x_col])
            if not np.isfinite(x_val):
                continue
            sid = str(row["Name"])
            b_n_list, b_p_list = b_data.get(sid, ([], []))
            values = b_n_list if y_label == "b_n" else b_p_list
            if not values:
                continue
            xs.append(x_val)
            vals_per_spec.append(values)
            sids.append(sid)
        if not xs:
            ax.set_ylabel(r"$b_n$" if y_label == "b_n" else r"$b_p$")
            style_axes_spines_and_ticks(ax)
            return
        xs_arr = np.asarray(xs, dtype=float)
        x_min, x_max = float(xs_arr.min()), float(xs_arr.max())
        width = 0.02 * (x_max - x_min) if x_max > x_min else 0.02 * max(abs(x_max), 1.0)
        bp = ax.boxplot(vals_per_spec, positions=xs_arr, patch_artist=True, widths=width)
        for i, patch in enumerate(bp["boxes"]):
            c = specimen_color_map.get(sids[i], fallback_color)
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for i, patch in enumerate(bp["boxes"]):
            c = specimen_color_map.get(sids[i], fallback_color)
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
            if i < len(bp["medians"]):
                bp["medians"][i].set_color(c)
            for k in (2 * i, 2 * i + 1):
                if k < len(bp["whiskers"]):
                    bp["whiskers"][k].set_color(c)
                if k < len(bp["caps"]):
                    bp["caps"][k].set_color(c)
            if i < len(bp["fliers"]):
                bp["fliers"][i].set_color(c)
        ax.set_xlabel(x_latex)
        ax.set_ylabel(r"$b_n$" if y_label == "b_n" else r"$b_p$")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_x_3sig))
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        ax.set_ylim(0.0, 0.05)
        ax.grid(True, alpha=0.3, axis="y")
        style_axes_spines_and_ticks(ax)

    for y_label, y_col, _, out_name in [
        ("b_n", "b_n_median", r"$b_n$ (compression) vs geometry", "box_bn_vs_geometry.png"),
        ("b_p", "b_p_median", r"$b_p$ (tension) vs geometry", "box_bp_vs_geometry.png"),
    ]:
        if y_col not in df.columns:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(b_vs_geometry_rcparams()):
            fig, axs = plt.subplots(3, 4, figsize=figsize_for_grid(3, 4), layout="constrained")
            for idx, (x_col, x_latex, x_lim) in enumerate(SCATTER_PANELS_ORDERED):
                row, col = idx // 4, idx % 4
                _draw_box_panel(axs[row, col], df, b_data, x_col, x_latex, y_label, specimen_color, x_lim=x_lim)
            axs[2, 3].set_visible(False)
            fig.legend(
                handles=specimen_handles,
                loc="outside lower center",
                ncol=min(B_VS_GEOMETRY_LEGEND_NCOL, len(specimen_handles)),
            )
            fig.savefig(out_dir / out_name, dpi=SAVE_DPI)
            plt.close(fig)


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Plot b histograms and b vs geometry scatter.")
    parser.add_argument("--hist-only", action="store_true", help="Only plot histograms")
    parser.add_argument("--scatter-only", action="store_true", help="Only plot scatter")
    parser.add_argument(
        "--skip-digitized-scatter",
        action="store_true",
        help="Skip digitized envelope b vs geometry (same folder as segment-based scatters)",
    )
    parser.add_argument("--specimen", type=str, default=None, help="Single specimen for per-specimen histogram")
    args = parser.parse_args()

    specimens = get_specimens_with_resampled()
    if not specimens:
        print("No specimens with resampled data.")
        return

    do_hist = not args.scatter_only
    do_scatter = not args.hist_only

    if do_hist:
        b_data = {sid: get_b_lists_one_specimen(sid) for sid in specimens}
        for specimen_id in specimens:
            b_n_list, b_p_list = b_data[specimen_id]
            if args.specimen and specimen_id != args.specimen:
                continue
            plot_histogram_one_specimen(specimen_id, b_n_list, b_p_list, HIST_DIR)
        if not args.specimen:
            plot_histograms_all(specimens, b_data, HIST_DIR)
        print(f"Histograms saved to {HIST_DIR}")

    if do_scatter:
        if not PARAMS_CSV.exists():
            print(f"Run extract_bn_bp first to create {PARAMS_CSV}")
            return
        df = pd.read_csv(PARAMS_CSV)
        for stat in SEGMENT_STATS_ORDER:
            plot_scatter_bn_bp(df, B_VS_GEOMETRY_DIR, segment_stat=stat)
        plot_box_bn_bp(df, B_VS_GEOMETRY_DIR)
        print(f"Scatter and box plots saved to {B_VS_GEOMETRY_DIR}")
        if not args.skip_digitized_scatter:
            ddf = build_digitized_envelope_bn_table(_PROJECT_ROOT)
            if ddf.empty:
                print("No scatter-cloud digitized specimens with valid F-u CSVs; skip digitized scatter.")
            else:
                plot_scatter_bn_bp_digitized(ddf, B_VS_GEOMETRY_DIR)
                for stat in SEGMENT_STATS_ORDER:
                    plot_scatter_bn_bp_combined(df, ddf, B_VS_GEOMETRY_DIR, segment_stat=stat)
                print(
                    f"Digitized + combined b vs geometry figures saved under {B_VS_GEOMETRY_DIR} "
                    f"(one envelope scatter pair; ordered/all × {', '.join(SEGMENT_STATS_ORDER)})."
                )


if __name__ == "__main__":
    main()
