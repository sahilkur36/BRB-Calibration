r"""
Generate simulated vs experimental (resampled) plots using a parameters CSV.

For each specimen with:
- **individual_optimize=true** in BRB-Specimens.csv (digitized unordered rows are skipped here)
- resampled CSV in data/resampled/{Name}/force_deformation.csv
- geometry name in BRB-Specimens.csv (for discovery); model inputs from parameters CSV
- parameter row in the given parameters CSV (e.g. initial_brb_parameters.csv or
  optimized_brb_parameters.csv): columns follow ``run_simulation`` / SteelMPF order after
  ID/Name/set_id -- L_T, L_y, A_sc, A_t, fyp, fyn, E, b_p, b_n, R0, cR1, cR2, a1-a4

this script runs the corotruss model with SteelMPF and plots both:
- **Physical** overlays ($P$ [kip] vs deformation [in]) — per-specimen companion only; exception to the repo-wide normalized hysteresis default.
- **Normalized** overlays ($P/(f_y A_{sc})$ vs $\delta/L_y$ as percent strain; limits can match
  ``results/plots/postprocess/force_deformation/raw_and_filtered/all_specimens_raw_filtered.png``).
  Normalized single-specimen PNGs use figure legends (**Experimental** vs **Numerical**; digitized
  unordered scatter uses the same Experimental label as path-ordered lines). ``plot_compare_calibration_overlays.py`` builds the per-``set_id`` combined grid from
  ``*_simulated.csv`` with **one** shared legend for the whole figure.

Plots are saved under ``results/plots/calibration/individual_optimize/<output_dir>`` (default
``overlays``). Use --params and --output-dir to compare different parameter sets.

Optional ``--override-bp`` / ``--override-bn`` replace ``b_p`` / ``b_n`` for every simulation
(e.g. preset overlays before L-BFGS).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

# Project root and path setup for imports from scripts/model
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.calibration_paths import (  # noqa: E402
    BRB_SPECIMENS_CSV,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    PLOTS_INDIVIDUAL_OPTIMIZE,
)
from calibrate.digitized_unordered_eval_lib import (  # noqa: E402
    DEFAULT_BINENV_N_BINS,
    num_to_exp_nearest_indices,
)
from model.corotruss import run_simulation  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    COLOR_EXPERIMENTAL,
    COLOR_NUMERICAL_COHORT,
    LEGEND_FONT_SIZE_SMALL_PT,
    SAVE_DPI,
    SINGLE_FIGSIZE_IN,
    configure_matplotlib_style,
    style_axes_spines_and_ticks,
)
from postprocess.plot_specimens import (  # noqa: E402
    NORM_FORCE_LABEL,
    NORM_STRAIN_LABEL,
    PHYS_FORCE_KIP_LABEL,
    apply_normalized_fu_axes,
    compute_raw_filtered_global_norm_limits,
)
from specimen_catalog import (  # noqa: E402
    get_specimen_record,
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_resampled_force_deformation_csv,
)


CATALOG_PATH = BRB_SPECIMENS_CSV
DEFAULT_PARAMS_PATH = OPTIMIZED_BRB_PARAMETERS_PATH
PLOTS_BASE = PLOTS_INDIVIDUAL_OPTIMIZE

# Plot styling (numerical model curve); experimental color from ``plot_dimensions``.
# Individual L-BFGS overlays: always **train** color. (Generalized/averaged per-specimen PNGs set train vs
# validation in ``eval_averaged_params`` / ``optimize_generalized_brb_mse``.)
COLOR_SIMULATED = COLOR_NUMERICAL_COHORT
LINEWIDTH_SIMULATED = 0.9
LINEWIDTH_EXPERIMENTAL = LINEWIDTH_SIMULATED  # same thickness as numerical on all overlays
# Digitized unordered experimental clouds (scatter ``s`` is marker area in points²).
DIGITIZED_UNORDERED_OVERLAY_SCATTER_S = 5  # half of former 10
DIGITIZED_UNORDERED_LEGEND_MARKERSIZE_PT = 3.5  # half of former 7 for combined-montage legend proxy
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
configure_matplotlib_style()


def _binned_deformation_max_min(
    u: np.ndarray,
    f: np.ndarray,
    *,
    u_min: float,
    u_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Equal-width bins on [u_min, u_max] (same edge rule as ``J_binenv`` in
    ``digitized_unordered_eval_lib``). Returns bin centers and per-bin max/min ``F``; NaN if bin empty.
    """
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    if u.shape[0] == 0 or int(n_bins) < 1:
        return (
            np.full(int(n_bins), np.nan),
            np.full(int(n_bins), np.nan),
            np.full(int(n_bins), np.nan),
        )
    edges = np.linspace(u_min, u_max, int(n_bins) + 1)
    u_c = 0.5 * (edges[:-1] + edges[1:])
    upper = np.full(int(n_bins), np.nan, dtype=float)
    lower = np.full(int(n_bins), np.nan, dtype=float)
    nb = int(n_bins)
    for b in range(nb):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == nb - 1:
            m = (u >= lo) & (u <= hi)
        else:
            m = (u >= lo) & (u < hi)
        if not np.any(m):
            continue
        fb = f[m]
        upper[b] = float(np.nanmax(fb))
        lower[b] = float(np.nanmin(fb))
    return u_c, upper, lower


def set_symmetric_axes(ax, x_data: np.ndarray, y_data: np.ndarray, margin: float = 1.05) -> None:
    """Set x and y limits symmetric about zero (no equal aspect)."""
    x_max = float(np.nanmax(np.abs(x_data))) * margin
    y_max = float(np.nanmax(np.abs(y_data))) * margin
    if x_max == 0:
        x_max = 1.0
    if y_max == 0:
        y_max = 1.0
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)


def plot_force_def_overlays(
    specimen_id: str,
    displacement_in: np.ndarray,
    F_exp_kip: np.ndarray,
    F_sim_kip: np.ndarray,
    *,
    fy_ksi: float,
    A_c_in2: float,
    L_y_in: float,
    set_id: int | str,
    out_dir: Path,
    norm_xy_half: tuple[float, float] | None = None,
    numerical_color: str = COLOR_NUMERICAL_COHORT,
    show_numerical_curve: bool = True,
) -> None:
    """Create physical and normalized force–deformation overlays."""
    y_phys_components = [F_exp_kip]
    if show_numerical_curve:
        y_phys_components.append(F_sim_kip)
    # Physical units: P [kip] vs deformation [in]
    fig1, ax1 = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
    ax1.plot(
        displacement_in,
        F_exp_kip,
        color=COLOR_EXPERIMENTAL,
        alpha=0.95,
        linewidth=LINEWIDTH_EXPERIMENTAL,
        linestyle="-",
        label="Experimental",
    )
    if show_numerical_curve:
        ax1.plot(
            displacement_in,
            F_sim_kip,
            color=numerical_color,
            alpha=0.95,
            linewidth=LINEWIDTH_SIMULATED,
            linestyle="--",
            label="Numerical",
        )
    set_symmetric_axes(ax1, displacement_in, np.concatenate(y_phys_components))
    ax1.set_xlabel("Deformation [in]")
    ax1.set_ylabel(PHYS_FORCE_KIP_LABEL)
    h1, lab1 = ax1.get_legend_handles_labels()
    fig1.legend(
        h1,
        lab1,
        loc="outside upper center",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE_SMALL_PT,
        frameon=False,
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axvline(0, color="k", linewidth=0.5)
    style_axes_spines_and_ticks(ax1)
    fig1.savefig(out_dir / f"{specimen_id}_set{set_id}_force_def.png", dpi=SAVE_DPI)
    plt.close(fig1)

    # Normalized: F/(f_yc * A_c) vs delta/L_y
    fyA = fy_ksi * A_c_in2
    if fyA <= 0 or L_y_in <= 0:
        return
    disp_norm = displacement_in / L_y_in
    F_exp_norm = F_exp_kip / fyA
    F_sim_norm = F_sim_kip / fyA

    fig2, ax2 = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
    ax2.plot(
        disp_norm,
        F_exp_norm,
        color=COLOR_EXPERIMENTAL,
        alpha=0.95,
        linewidth=LINEWIDTH_EXPERIMENTAL,
        linestyle="-",
        label="Experimental",
    )
    y_norm_components = [F_exp_norm]
    if show_numerical_curve:
        ax2.plot(
            disp_norm,
            F_sim_norm,
            color=numerical_color,
            alpha=0.95,
            linewidth=LINEWIDTH_SIMULATED,
            linestyle="--",
            label="Numerical",
        )
        y_norm_components.append(F_sim_norm)
    if norm_xy_half is not None:
        hx, hy = norm_xy_half
        ax2.set_xlim(-hx, hx)
        ax2.set_ylim(-hy, hy)
    else:
        set_symmetric_axes(ax2, disp_norm, np.concatenate(y_norm_components))
    ax2.set_xlabel(NORM_STRAIN_LABEL)
    ax2.set_ylabel(NORM_FORCE_LABEL)
    apply_normalized_fu_axes(ax2)
    h2, lab2 = ax2.get_legend_handles_labels()
    fig2.legend(
        h2,
        lab2,
        loc="outside upper center",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE_SMALL_PT,
        frameon=False,
    )
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.axvline(0, color="k", linewidth=0.5)
    style_axes_spines_and_ticks(ax2)
    fig2.savefig(out_dir / f"{specimen_id}_set{set_id}_force_def_norm.png", dpi=SAVE_DPI)
    plt.close(fig2)


def plot_force_def_digitized_unordered_overlays(
    specimen_id: str,
    D_drive: np.ndarray,
    F_sim_kip: np.ndarray,
    u_unordered: np.ndarray,
    F_unordered: np.ndarray,
    *,
    fy_ksi: float,
    A_c_in2: float,
    L_y_in: float,
    set_id: int | str,
    out_dir: Path,
    norm_xy_half: tuple[float, float] | None = None,
    numerical_color: str = COLOR_NUMERICAL_COHORT,
    show_numerical_curve: bool = True,
) -> None:
    """
    Simulated hysteresis driven by ``D_drive`` vs digitized unordered samples ``(u_unordered, F_unordered)``.
    Same output filenames as ``plot_force_def_overlays`` (for averaged/generalized ``overlays/``).
    """
    safe = str(set_id).replace(" ", "_")

    x_phys = np.concatenate([np.asarray(D_drive, dtype=float), np.asarray(u_unordered, dtype=float)])
    y_phys_list: list[np.ndarray] = [np.asarray(F_unordered, dtype=float)]
    if show_numerical_curve:
        y_phys_list.insert(0, np.asarray(F_sim_kip, dtype=float))
    y_phys = np.concatenate(y_phys_list)

    fig1, ax1 = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
    ax1.scatter(
        u_unordered,
        F_unordered,
        s=DIGITIZED_UNORDERED_OVERLAY_SCATTER_S,
        alpha=0.35,
        c=COLOR_EXPERIMENTAL,
        edgecolors="none",
        rasterized=True,
        label="Experimental",
    )
    if show_numerical_curve:
        ax1.plot(
            D_drive,
            F_sim_kip,
            color=numerical_color,
            alpha=0.95,
            linewidth=LINEWIDTH_SIMULATED,
            linestyle="--",
            label="Numerical",
        )
    set_symmetric_axes(ax1, x_phys, y_phys)
    ax1.set_xlabel("Deformation [in]")
    ax1.set_ylabel(PHYS_FORCE_KIP_LABEL)
    h1, lab1 = ax1.get_legend_handles_labels()
    fig1.legend(
        h1,
        lab1,
        loc="outside upper center",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE_SMALL_PT,
        frameon=False,
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axvline(0, color="k", linewidth=0.5)
    style_axes_spines_and_ticks(ax1)
    fig1.savefig(out_dir / f"{specimen_id}_set{safe}_force_def.png", dpi=SAVE_DPI)
    plt.close(fig1)

    fyA = fy_ksi * A_c_in2
    if fyA <= 0 or L_y_in <= 0:
        return
    disp_drive_n = np.asarray(D_drive, dtype=float) / L_y_in
    F_sim_n = np.asarray(F_sim_kip, dtype=float) / fyA
    u_unordered_n = np.asarray(u_unordered, dtype=float) / L_y_in
    F_unordered_n = np.asarray(F_unordered, dtype=float) / fyA
    x_n = np.concatenate([disp_drive_n, u_unordered_n])
    y_n_list: list[np.ndarray] = [F_unordered_n]
    if show_numerical_curve:
        y_n_list.insert(0, F_sim_n)
    y_n = np.concatenate(y_n_list)

    fig2, ax2 = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
    ax2.scatter(
        u_unordered_n,
        F_unordered_n,
        s=DIGITIZED_UNORDERED_OVERLAY_SCATTER_S,
        alpha=0.35,
        c=COLOR_EXPERIMENTAL,
        edgecolors="none",
        rasterized=True,
        label="Experimental",
    )
    if show_numerical_curve:
        ax2.plot(
            disp_drive_n,
            F_sim_n,
            color=numerical_color,
            alpha=0.95,
            linewidth=LINEWIDTH_SIMULATED,
            linestyle="--",
            label="Numerical",
        )
    if norm_xy_half is not None:
        hx, hy = norm_xy_half
        ax2.set_xlim(-hx, hx)
        ax2.set_ylim(-hy, hy)
    else:
        set_symmetric_axes(ax2, x_n, y_n)
    ax2.set_xlabel(NORM_STRAIN_LABEL)
    ax2.set_ylabel(NORM_FORCE_LABEL)
    apply_normalized_fu_axes(ax2)
    h2, lab2 = ax2.get_legend_handles_labels()
    fig2.legend(
        h2,
        lab2,
        loc="outside upper center",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE_SMALL_PT,
        frameon=False,
    )
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.axvline(0, color="k", linewidth=0.5)
    style_axes_spines_and_ticks(ax2)
    fig2.savefig(out_dir / f"{specimen_id}_set{safe}_force_def_norm.png", dpi=SAVE_DPI)
    plt.close(fig2)


def plot_unordered_binned_cloud_envelopes(
    specimen_id: str,
    set_id: int | str,
    exp_points_plot: np.ndarray,
    num_points_plot: np.ndarray,
    *,
    out_dir: Path,
    n_bins: int = DEFAULT_BINENV_N_BINS,
) -> None:
    """
    Experimental and numerical scatter in normalized (δ/L_y, P/(f_y A_sc)) plus deformation-binned
    upper/lower envelope curves (max/min force per bin in ``u``). No interior shading—lines only.
    Bin edges match ``J_binenv`` (same as the plotted upper/lower curves): [min(u_exp), max(u_exp)]
    with equal width; last bin closed on the right.
    """
    safe = str(set_id).replace(" ", "_")
    exp_xy = np.asarray(exp_points_plot, dtype=float)
    num_xy = np.asarray(num_points_plot, dtype=float)
    if exp_xy.ndim != 2 or num_xy.ndim != 2 or exp_xy.shape[1] != 2 or num_xy.shape[1] != 2:
        return
    if exp_xy.shape[0] == 0 or num_xy.shape[0] == 0:
        return

    u_e = exp_xy[:, 0]
    f_e = exp_xy[:, 1]
    u_n = num_xy[:, 0]
    f_n = num_xy[:, 1]
    u_min = float(np.min(u_e))
    u_max = float(np.max(u_e))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return

    u_c, e_up, e_lo = _binned_deformation_max_min(
        u_e, f_e, u_min=u_min, u_max=u_max, n_bins=n_bins
    )
    _u_c2, n_up, n_lo = _binned_deformation_max_min(
        u_n, f_n, u_min=u_min, u_max=u_max, n_bins=n_bins
    )
    del _u_c2

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")

    # Light scatter so binned upper/lower curves remain the visual focus.
    ax.scatter(
        u_e,
        f_e,
        s=1.125,
        alpha=0.10,
        c=COLOR_EXPERIMENTAL,
        edgecolors="none",
        rasterized=True,
        label="Exp. cloud",
        zorder=1,
    )
    ax.scatter(
        u_n,
        f_n,
        s=2.0,
        alpha=0.12,
        c=COLOR_NUMERICAL_COHORT,
        edgecolors="none",
        rasterized=True,
        label="Num. cloud",
        zorder=1,
    )

    # No fill_between: shaded bands between upper/lower obscured the cloud vs envelope comparison.

    _env_lw = 1.65
    ax.plot(
        u_c,
        e_up,
        color=COLOR_EXPERIMENTAL,
        linewidth=_env_lw,
        linestyle="-",
        label="Exp. upper",
        zorder=4,
    )
    ax.plot(
        u_c,
        e_lo,
        color=COLOR_EXPERIMENTAL,
        linewidth=_env_lw,
        linestyle="--",
        label="Exp. lower",
        zorder=4,
    )
    ax.plot(
        u_c,
        n_up,
        color=COLOR_NUMERICAL_COHORT,
        linewidth=_env_lw,
        linestyle="-",
        label="Num. upper",
        zorder=4,
    )
    ax.plot(
        u_c,
        n_lo,
        color=COLOR_NUMERICAL_COHORT,
        linewidth=_env_lw,
        linestyle="--",
        label="Num. lower",
        zorder=4,
    )

    x_all = np.concatenate([u_e, u_n])
    y_all = np.concatenate([f_e, f_n, e_up[np.isfinite(e_up)], e_lo[np.isfinite(e_lo)], n_up[np.isfinite(n_up)], n_lo[np.isfinite(n_lo)]])
    set_symmetric_axes(ax, x_all, y_all)
    ax.set_xlabel(NORM_STRAIN_LABEL)
    ax.set_ylabel(NORM_FORCE_LABEL)
    apply_normalized_fu_axes(ax)
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    style_axes_spines_and_ticks(ax)

    h, l = ax.get_legend_handles_labels()
    _binenv_legend_pt = max(5.0, LEGEND_FONT_SIZE_SMALL_PT - 1.0)
    fig.legend(
        h,
        l,
        loc="outside upper center",
        ncol=3,
        fontsize=_binenv_legend_pt,
        frameon=False,
    )
    fig.savefig(out_dir / f"{specimen_id}_set{safe}_binned_cloud_envelope_norm.png", dpi=SAVE_DPI)
    plt.close(fig)


def run_one_specimen(
    specimen_id: str,
    params_rows: pd.DataFrame,
    catalog_row: pd.Series,
    out_dir: Path,
    *,
    norm_xy_half: tuple[float, float] | None = None,
    override_bp: float | None = None,
    override_bn: float | None = None,
) -> None:
    """Run simulation and plot overlays for one specimen (possibly multiple parameter sets)."""
    csv_path = resolve_resampled_force_deformation_csv(specimen_id, _PROJECT_ROOT)
    if csv_path is None or not csv_path.is_file():
        print(f"  Skipping {specimen_id}: resampled force_deformation.csv not found")
        return
    df = pd.read_csv(csv_path)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        print(f"  Skipping {specimen_id}: resampled CSV missing Force[kip] or Deformation[in]")
        return

    displacement = df["Deformation[in]"].to_numpy(dtype=float)
    F_exp = df["Force[kip]"].to_numpy(dtype=float)

    for _, prow in params_rows.iterrows():
        set_id = prow.get("set_id", 1)
        fy = float(prow["fyp"])
        if (
            "A_sc" not in prow
            or "A_t" not in prow
            or pd.isna(prow["A_sc"])
            or pd.isna(prow["A_t"])
        ):
            print(f"  Skipping {specimen_id}, set {set_id}: need A_sc and A_t in parameters CSV")
            continue
        A_sc = float(prow["A_sc"])
        A_t = float(prow["A_t"])
        L_T_param = float(prow["L_T"])
        if "L_y" in prow and pd.notna(prow.get("L_y")):
            L_y = float(prow["L_y"])
        else:
            L_y = float(catalog_row["L_y_in"])
        E_val = float(prow["E"])
        b_n = float(override_bn) if override_bn is not None else float(prow["b_n"])
        b_p = float(override_bp) if override_bp is not None else float(prow["b_p"])
        a1 = float(prow["a1"])
        a2 = float(prow["a2"])
        a3 = float(prow["a3"])
        a4 = float(prow["a4"])
        R0 = float(prow["R0"])
        cR1 = float(prow["cR1"])
        cR2 = float(prow["cR2"])

        try:
            F_sim = run_simulation(
                displacement,
                L_T=L_T_param,
                L_y=L_y,
                A_sc=A_sc,
                A_t=A_t,
                fyp=fy,
                fyn=fy,
                E=E_val,
                b_p=b_p,
                b_n=b_n,
                R0=R0,
                cR1=cR1,
                cR2=cR2,
                a1=a1,
                a2=a2,
                a3=a3,
                a4=a4,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  Simulation failed for {specimen_id}, set {set_id}: {exc}")
            continue

        F_sim = np.asarray(F_sim, dtype=float)
        if F_sim.shape != F_exp.shape:
            print(
                f"  Simulation length mismatch for {specimen_id}, set {set_id}: "
                f"exp={F_exp.shape}, sim={F_sim.shape}"
            )
            continue

        plot_force_def_overlays(
            specimen_id,
            displacement,
            F_exp,
            F_sim,
            fy_ksi=fy,
            A_c_in2=A_sc,
            L_y_in=L_y,
            set_id=set_id,
            out_dir=out_dir,
            norm_xy_half=norm_xy_half,
        )
        print(f"  Saved plots for {specimen_id}, set {set_id}")


def write_one_specimen_simulated_csvs(
    specimen_id: str,
    params_rows: pd.DataFrame,
    catalog_row: pd.Series,
    sim_dir: Path,
    *,
    override_bp: float | None = None,
    override_bn: float | None = None,
) -> int:
    """Run simulations and write ``{{Name}}_set{{k}}_simulated.csv`` only (no PNGs). Returns write count."""
    from calibrate.optimize_brb_mse import save_simulated_force_history_csv

    csv_path = resolve_resampled_force_deformation_csv(specimen_id, _PROJECT_ROOT)
    if csv_path is None or not csv_path.is_file():
        print(f"  Skipping {specimen_id}: resampled force_deformation.csv not found")
        return 0
    df = pd.read_csv(csv_path)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        print(f"  Skipping {specimen_id}: resampled CSV missing Force[kip] or Deformation[in]")
        return 0

    displacement = df["Deformation[in]"].to_numpy(dtype=float)
    F_exp = df["Force[kip]"].to_numpy(dtype=float)
    n_ok = 0

    for _, prow in params_rows.iterrows():
        set_id = prow.get("set_id", 1)
        fy = float(prow["fyp"])
        if (
            "A_sc" not in prow
            or "A_t" not in prow
            or pd.isna(prow["A_sc"])
            or pd.isna(prow["A_t"])
        ):
            print(f"  Skipping {specimen_id}, set {set_id}: need A_sc and A_t in parameters CSV")
            continue
        A_sc = float(prow["A_sc"])
        A_t = float(prow["A_t"])
        L_T_param = float(prow["L_T"])
        if "L_y" in prow and pd.notna(prow.get("L_y")):
            L_y = float(prow["L_y"])
        else:
            L_y = float(catalog_row["L_y_in"])
        E_val = float(prow["E"])
        b_n = float(override_bn) if override_bn is not None else float(prow["b_n"])
        b_p = float(override_bp) if override_bp is not None else float(prow["b_p"])
        a1 = float(prow["a1"])
        a2 = float(prow["a2"])
        a3 = float(prow["a3"])
        a4 = float(prow["a4"])
        R0 = float(prow["R0"])
        cR1 = float(prow["cR1"])
        cR2 = float(prow["cR2"])

        try:
            F_sim = run_simulation(
                displacement,
                L_T=L_T_param,
                L_y=L_y,
                A_sc=A_sc,
                A_t=A_t,
                fyp=fy,
                fyn=fy,
                E=E_val,
                b_p=b_p,
                b_n=b_n,
                R0=R0,
                cR1=cR1,
                cR2=cR2,
                a1=a1,
                a2=a2,
                a3=a3,
                a4=a4,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  Simulation failed for {specimen_id}, set {set_id}: {exc}")
            continue

        F_sim = np.asarray(F_sim, dtype=float)
        if F_sim.shape != F_exp.shape:
            print(
                f"  Simulation length mismatch for {specimen_id}, set {set_id}: "
                f"exp={F_exp.shape}, sim={F_sim.shape}"
            )
            continue

        saved = save_simulated_force_history_csv(
            sim_dir, specimen_id, set_id, displacement, F_exp, F_sim
        )
        if saved is not None:
            n_ok += 1
            print(f"  Wrote {saved.name}")

    return n_ok


def run_multi_specimen_simulated_csvs(
    params_df: pd.DataFrame,
    sim_dir: Path,
    *,
    params_path_label: str | Path = "parameters",
    specimen: str | None = None,
    override_bp: float | None = None,
    override_bn: float | None = None,
) -> None:
    """Write ``*_simulated.csv`` for path-ordered individual_optimize specimens (same filters as overlays)."""
    if "Name" not in params_df.columns:
        raise RuntimeError(f"'Name' column not found in parameters ({params_path_label})")

    sim_dir.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog(CATALOG_PATH)
    catalog_by_name = catalog.set_index("Name")

    params_grouped = params_df.groupby("Name")

    resampled_stems = path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT)
    param_names = set(params_df["Name"].astype(str))
    catalog_names = set(catalog["Name"].astype(str))
    with_data = sorted(param_names & catalog_names & resampled_stems)
    available = [
        s for s in with_data if get_specimen_record(str(s).strip(), catalog).individual_optimize
    ]
    if not available:
        print(
            "No specimens found with individual_optimize=true, parameters, catalog entry, "
            "and resampled data."
        )
        return

    if specimen:
        sid = str(specimen).strip()
        if sid not in catalog_names:
            print(f"Specimen {sid} not found in {CATALOG_PATH}")
            return
        if not get_specimen_record(sid, catalog).individual_optimize:
            print(
                f"Specimen {sid} has individual_optimize=false (e.g. digitized unordered). "
                "Preset combined overlays only cover path-ordered specimens."
            )
            return
        if sid not in available:
            print(
                f"Specimen {sid} not available. "
                "It must exist in the parameters table and data/resampled/{Name}/force_deformation.csv with "
                "individual_optimize=true."
            )
            return
        specimens = [sid]
    else:
        specimens = available

    if override_bp is not None or override_bn is not None:
        print(f"  Overriding b_p={override_bp}, b_n={override_bn} (None = use CSV per row)")
    print(f"Writing simulated CSVs into {sim_dir} for specimens: {', '.join(specimens)}")
    for sid in specimens:
        if sid not in params_grouped.groups:
            print(f"Skipping {sid}: no parameter rows found in {params_path_label}")
            continue
        if sid not in catalog_by_name.index:
            print(f"Skipping {sid}: not found in {CATALOG_PATH}")
            continue
        write_one_specimen_simulated_csvs(
            sid,
            params_grouped.get_group(sid),
            catalog_by_name.loc[sid],
            sim_dir,
            override_bp=override_bp,
            override_bn=override_bn,
        )


def run_multi_specimen_overlays(
    params_df: pd.DataFrame,
    *,
    plots_dir: Path,
    params_path_label: str | Path = "parameters",
    specimen: str | None = None,
    override_bp: float | None = None,
    override_bn: float | None = None,
) -> None:
    """
    Load catalog, resolve specimens, and write overlays for each row in ``params_df``.

    ``params_path_label`` is only used in log messages (path to CSV or a short label).
    """
    if "Name" not in params_df.columns:
        raise RuntimeError(f"'Name' column not found in parameters ({params_path_label})")

    plots_dir.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog(CATALOG_PATH)
    catalog_by_name = catalog.set_index("Name")
    norm_xy_half = compute_raw_filtered_global_norm_limits(catalog, project_root=_PROJECT_ROOT)

    params_grouped = params_df.groupby("Name")

    resampled_stems = path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT)
    param_names = set(params_df["Name"].astype(str))
    catalog_names = set(catalog["Name"].astype(str))
    with_data = sorted(param_names & catalog_names & resampled_stems)
    available = [
        s for s in with_data if get_specimen_record(str(s).strip(), catalog).individual_optimize
    ]
    if not available:
        print(
            "No specimens found with individual_optimize=true, parameters, catalog entry, "
            "and resampled data."
        )
        return

    if specimen:
        sid = str(specimen).strip()
        if sid not in catalog_names:
            print(f"Specimen {sid} not found in {CATALOG_PATH}")
            return
        if not get_specimen_record(sid, catalog).individual_optimize:
            print(
                f"Specimen {sid} has individual_optimize=false (e.g. digitized unordered). "
                "Use averaged or generalized evaluation overlays instead: "
                "results/plots/calibration/averaged_optimize/overlays/ or "
                "results/plots/calibration/generalized_optimize/overlays/."
            )
            return
        if sid not in available:
            print(
                f"Specimen {sid} not available. "
                "It must exist in the parameters table and data/resampled/{Name}/force_deformation.csv with "
                "individual_optimize=true. "
                "(L_y may come from the parameters row or the catalog.)"
            )
            return
        specimens = [sid]
    else:
        specimens = available

    if override_bp is not None or override_bn is not None:
        print(f"  Overriding b_p={override_bp}, b_n={override_bn} (None = use CSV per row)")
    print(f"Generating plots into {plots_dir} for specimens: {', '.join(specimens)}")
    for sid in specimens:
        if sid not in params_grouped.groups:
            print(f"Skipping {sid}: no parameter rows found in {params_path_label}")
            continue
        if sid not in catalog_by_name.index:
            print(f"Skipping {sid}: not found in {CATALOG_PATH}")
            continue
        run_one_specimen(
            sid,
            params_grouped.get_group(sid),
            catalog_by_name.loc[sid],
            plots_dir,
            norm_xy_half=norm_xy_half,
            override_bp=override_bp,
            override_bn=override_bn,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate simulated vs experimental (resampled) plots using a parameters CSV.",
    )
    parser.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="Single specimen Name (e.g. PC250). Default: all specimens with individual_optimize=true, parameters, and resampled data.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to parameters CSV (default: results/calibration/individual_optimize/optimized_brb_parameters.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="overlays",
        help="Subfolder under results/plots/calibration/individual_optimize (default: overlays).",
    )
    parser.add_argument(
        "--override-bp",
        type=float,
        default=None,
        metavar="VAL",
        help="If set, use this b_p for every row instead of the CSV column.",
    )
    parser.add_argument(
        "--override-bn",
        type=float,
        default=None,
        metavar="VAL",
        help="If set, use this b_n for every row instead of the CSV column.",
    )
    args = parser.parse_args()

    params_path = Path(args.params) if args.params else DEFAULT_PARAMS_PATH
    plots_dir = PLOTS_BASE / args.output_dir

    params_df = pd.read_csv(params_path)
    obp = float(args.override_bp) if args.override_bp is not None else None
    obn = float(args.override_bn) if args.override_bn is not None else None
    run_multi_specimen_overlays(
        params_df,
        plots_dir=plots_dir,
        params_path_label=params_path,
        specimen=args.specimen,
        override_bp=obp,
        override_bn=obn,
    )


if __name__ == "__main__":
    main()
