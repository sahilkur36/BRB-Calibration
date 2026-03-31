"""
Plot resampled force-deformation hysteresis with fitted b (hardening) slopes
overlaid per cycle, and digitized scatter-cloud envelope figures -- **same output directory**
(``results/plots/apparent_b/b_slopes/`` by default). Resampled and envelope-cloud figures both use
``{Name}.png`` (path-ordered PC* names and digitized cloud names are disjoint in practice).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS / "postprocess"))
sys.path.insert(0, str(_SCRIPTS))
from plot_dimensions import (
    COLOR_EXPERIMENTAL,
    LEGEND_FONT_SIZE_SMALL_PT,
    SAVE_DPI,
    SINGLE_FIGSIZE_IN,
    configure_matplotlib_style,
    style_axes_spines_and_ticks,
)
from model.corotruss import compute_Q

configure_matplotlib_style()
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled
from postprocess.plot_specimens import (
    NORM_FORCE_LABEL,
    NORM_STRAIN_LABEL,
    apply_normalized_fu_axes,
    normalize,
    set_symmetric_axes,
)
from load_raw import load_raw_valid

from calibrate.digitized_unordered_bn import compute_envelope_bn_unordered  # noqa: E402
from extract_bn_bp import (
    E_ksi,
    CATALOG_PATH,
    _segment_line_data,
    _segment_peak_ok,
    _segments_zero_to_peak,
    get_specimens_with_resampled,
)
from specimen_catalog import (  # noqa: E402
    force_deformation_unordered_csv_path,
    get_specimen_record,
    list_names_digitized_unordered,
    read_catalog,
    resolve_resampled_force_deformation_csv,
    uses_unordered_inputs,
)

_APPARENT_B = _PROJECT_ROOT / "results" / "plots" / "apparent_b"
PLOTS_B_SLOPES = _APPARENT_B / "b_slopes"

DEF_COL = "Deformation[in]"
FORCE_COL = "Force[kip]"

# Match resampled overlays and scatter-cloud envelope diagnostics (b_slopes)
COLOR_TENSION = "#CC3311"
COLOR_COMPRESSION = "#0077BB"


def plot_one_specimen(specimen_id: str, catalog_row: pd.Series, out_dir: Path) -> None:
    """Plot resampled hysteresis with fitted hardening segments overlaid."""
    rpath = resolve_resampled_force_deformation_csv(specimen_id, _PROJECT_ROOT)
    if rpath is None or not rpath.is_file():
        return
    df = pd.read_csv(rpath)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        return
    u = df["Deformation[in]"].values
    F = df["Force[kip]"].values
    n = len(u)
    if n == 0:
        return

    loaded = load_cycle_points_resampled(specimen_id)
    if loaded is None:
        points, _ = find_cycle_points(df)
    else:
        points, _ = loaded

    L_T = float(catalog_row["L_T_in"])
    L_y = float(catalog_row["L_y_in"])
    A_sc = float(catalog_row["A_c_in2"])
    A_t = float(catalog_row["A_t_in2"])
    fy = float(catalog_row["f_yc_ksi"])
    fy_A = fy * A_sc
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi

    # Normalized axes (project convention: delta = deformation, delta/L_y and F/(f_y A_sc) as in plot_specimens)
    u_norm = u / L_y if L_y != 0 else u
    F_norm = F / fy_A if fy_A != 0 else F

    segments = _segments_zero_to_peak(points)
    segment_lines: list[tuple[bool, float, np.ndarray, np.ndarray]] = []  # (is_tension, b, u_line, F_line)
    last_tension_peak: float | None = None
    last_compression_peak: float | None = None
    for start_idx, end_idx, _end_type in segments:
        if end_idx >= n or start_idx < 0:
            continue
        result = _segment_line_data(u, F, start_idx, end_idx, E_hat, A_sc, L_T, fy, L_y)
        if result is None:
            continue
        b, u_line, F_line, is_tension, _amp = result
        peak_def = float(u_line[-1])
        keep, last_tension_peak, last_compression_peak = _segment_peak_ok(
            peak_def, is_tension, last_tension_peak, last_compression_peak
        )
        if not keep:
            continue
        segment_lines.append((is_tension, b, u_line, F_line))

    # Colors: experimental hysteresis from ``plot_dimensions``; model fits in red/blue (color-blind friendly)
    COLOR_TENSION = "#CC3311"    # red tone (fitted b_p; distinguishable from blue for protan/deutan)
    COLOR_COMPRESSION = "#0077BB" # blue tone (fitted b_n; Tol/colorblind-safe)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
    ax.plot(u_norm, F_norm, color=COLOR_EXPERIMENTAL, linewidth=1.2, label="Experimental", zorder=1)
    for is_tension, b, u_line, F_line in segment_lines:
        color = COLOR_TENSION if is_tension else COLOR_COMPRESSION
        u_line_norm = u_line / L_y if L_y != 0 else u_line
        F_line_norm = F_line / fy_A if fy_A != 0 else F_line
        ax.plot(u_line_norm, F_line_norm, color=color, linewidth=1.5, alpha=0.9, linestyle="--", zorder=2)
    ax.plot([], [], color=COLOR_TENSION, linewidth=2, linestyle="--", label=r"Fitted $b_p$")
    ax.plot([], [], color=COLOR_COMPRESSION, linewidth=2, linestyle="--", label=r"Fitted $b_n$")
    ax.set_xlabel(NORM_STRAIN_LABEL)
    ax.set_ylabel(NORM_FORCE_LABEL)
    # Same axis limits as raw_and_filtered (trimmed raw + filtered data)
    raw_df = load_raw_valid(specimen_id)
    if raw_df is not None and "Force[kip]" in raw_df.columns and "Deformation[in]" in raw_df.columns:
        raw_n = normalize(raw_df, fy, A_sc, L_y)
        filtered_n = normalize(df, fy, A_sc, L_y)
        x_all = np.concatenate([raw_n["Deformation_norm"].values, filtered_n["Deformation_norm"].values])
        y_all = np.concatenate([raw_n["Force_norm"].values, filtered_n["Force_norm"].values])
        set_symmetric_axes(ax, x_all, y_all)
    else:
        set_symmetric_axes(ax, u_norm, F_norm)
    apply_normalized_fu_axes(ax, pct_decimals=2)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="outside upper center",
        ncol=3,
        fontsize=LEGEND_FONT_SIZE_SMALL_PT,
        frameon=False,
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    style_axes_spines_and_ticks(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{specimen_id}.png", dpi=SAVE_DPI)
    plt.close(fig)


def _plot_digitized_unordered_envelope_normalized(
    ax,
    u: np.ndarray,
    F: np.ndarray,
    diag,
    *,
    b_p: float,
    b_n: float,
    L_y: float,
    fy_A: float,
) -> None:
    """Scatter + envelope vertices and linear fits in delta/L_y vs F/(f_y A_sc) space."""
    if L_y <= 0 or fy_A <= 0:
        return
    u_n = u / L_y
    F_n = F / fy_A
    pl = diag.plastic_mask
    t_env = diag.tension_envelope_mask
    c_env = diag.compression_envelope_mask
    neither_plastic = ~pl
    plastic_only = pl & ~t_env & ~c_env

    if np.any(neither_plastic):
        ax.scatter(
            u_n[neither_plastic],
            F_n[neither_plastic],
            s=6,
            alpha=0.22,
            c=COLOR_EXPERIMENTAL,
            label=r"$\sigma / f_y \leq 1.1$",
            rasterized=True,
            zorder=1,
        )
    if np.any(plastic_only):
        ax.scatter(
            u_n[plastic_only],
            F_n[plastic_only],
            s=10,
            alpha=0.35,
            c=COLOR_EXPERIMENTAL,
            label=r"$\sigma / f_y > 1.1$",
            rasterized=True,
            zorder=2,
        )
    # Envelope vertices: ~same scale as grey cloud markers (s=6 / 10); edge for visibility.
    _s_env = 12
    if np.any(t_env):
        ax.scatter(
            u_n[t_env],
            F_n[t_env],
            s=_s_env,
            alpha=0.95,
            c=COLOR_TENSION,
            edgecolors="0.15",
            linewidths=0.35,
            label=r"$b_p$ envelope",
            zorder=5,
        )
    if np.any(c_env):
        ax.scatter(
            u_n[c_env],
            F_n[c_env],
            s=_s_env,
            alpha=0.95,
            c=COLOR_COMPRESSION,
            edgecolors="0.15",
            linewidths=0.35,
            label=r"$b_n$ envelope",
            zorder=5,
        )

    ut, Ft = diag.u_tension_env, diag.F_tension_env
    if len(ut) >= 2 and np.isfinite(b_p):
        utn = ut / L_y
        Ftn = Ft / fy_A
        coef = np.polyfit(utn, Ftn, 1)
        x_line = np.array([float(np.min(utn)), float(np.max(utn))])
        ax.plot(
            x_line,
            coef[0] * x_line + coef[1],
            color=COLOR_TENSION,
            linestyle="--",
            linewidth=1.35,
            alpha=0.85,
            zorder=4,
            label=rf"$b_p = {b_p:.4g}$",
        )
    uc, Fc = diag.u_compression_env, diag.F_compression_env
    if len(uc) >= 2 and np.isfinite(b_n):
        ucn = uc / L_y
        Fcn = Fc / fy_A
        coef = np.polyfit(ucn, Fcn, 1)
        x_line = np.array([float(np.min(ucn)), float(np.max(ucn))])
        ax.plot(
            x_line,
            coef[0] * x_line + coef[1],
            color=COLOR_COMPRESSION,
            linestyle="--",
            linewidth=1.35,
            alpha=0.85,
            zorder=4,
            label=rf"$b_n = {b_n:.4g}$",
        )

    ax.set_xlabel(NORM_STRAIN_LABEL)
    ax.set_ylabel(NORM_FORCE_LABEL)
    apply_normalized_fu_axes(ax, pct_decimals=2)
    ax.axhline(0.0, color="k", linewidth=0.4, alpha=0.4)
    ax.axvline(0.0, color="k", linewidth=0.4, alpha=0.4)
    ax.grid(True, alpha=0.3)


def plot_one_digitized_unordered(specimen_id: str, catalog_row: pd.Series, out_dir: Path) -> bool:
    """
    Digitized unordered P–u with envelope-based apparent b_p / b_n (normalized axes, same colors as resampled b_slopes).
    """
    fd_path = force_deformation_unordered_csv_path(specimen_id, _PROJECT_ROOT)
    if not fd_path.is_file():
        return False
    df = pd.read_csv(fd_path)
    if DEF_COL not in df.columns or FORCE_COL not in df.columns:
        return False
    u = df[DEF_COL].to_numpy(dtype=float)
    F = df[FORCE_COL].to_numpy(dtype=float)
    m = np.isfinite(u) & np.isfinite(F)
    u, F = u[m], F[m]
    if len(u) == 0:
        return False

    L_T = float(catalog_row["L_T_in"])
    L_y = float(catalog_row["L_y_in"])
    A_sc = float(catalog_row["A_c_in2"])
    A_t = float(catalog_row["A_t_in2"])
    fy = float(catalog_row["f_yc_ksi"])
    fy_A = fy * A_sc
    if L_y <= 0 or fy_A <= 0:
        return False

    diag = compute_envelope_bn_unordered(
        u,
        F,
        L_T=L_T,
        L_y=L_y,
        A_sc=A_sc,
        A_t=A_t,
        f_yc=fy,
        E_ksi_val=E_ksi,
    )
    b_p, b_n = diag.b_p, diag.b_n

    u_norm = u / L_y
    F_norm = F / fy_A

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
    _plot_digitized_unordered_envelope_normalized(
        ax, u, F, diag, b_p=b_p, b_n=b_n, L_y=L_y, fy_A=fy_A
    )
    set_symmetric_axes(ax, u_norm, F_norm)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="outside upper center",
        ncol=3,
        fontsize=LEGEND_FONT_SIZE_SMALL_PT,
        frameon=False,
    )
    style_axes_spines_and_ticks(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{specimen_id}.png", dpi=SAVE_DPI)
    plt.close(fig)
    return True


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Plot filtered hysteresis with fitted b slopes overlaid.")
    parser.add_argument("--specimen", type=str, default=None, help="Single specimen ID; default: all")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory for resampled and envelope-unordered b_slopes; default: results/plots/apparent_b/b_slopes",
    )
    parser.add_argument(
        "--digitized-output",
        type=str,
        default=None,
        help="Override directory for envelope-unordered figures only (default: same as --output)",
    )
    parser.add_argument(
        "--skip-digitized",
        action="store_true",
        help="Do not write digitized unordered envelope figures",
    )
    parser.add_argument(
        "--digitized-only",
        action="store_true",
        help="Only plot digitized unordered specimens (skip resampled hysteresis figures)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PLOTS_B_SLOPES
    out_dir = out_dir.resolve()
    digitized_dir = Path(args.digitized_output).resolve() if args.digitized_output else out_dir

    catalog = read_catalog(CATALOG_PATH)
    catalog_by_name = catalog.set_index("Name")
    unordered_names_all = list_names_digitized_unordered(catalog)

    if args.specimen:
        sid = args.specimen
        if sid not in catalog_by_name.index:
            print(f"Specimen {sid!r} not in catalog.")
            return
        row = catalog_by_name.loc[sid]
        rec = get_specimen_record(sid, catalog)
        is_unordered = uses_unordered_inputs(rec) or sid in unordered_names_all

        if args.digitized_only:
            if not is_unordered:
                print(f"Specimen {sid!r} is not digitized unordered; nothing to plot.")
                return
            if plot_one_digitized_unordered(sid, row, digitized_dir):
                print(f"  {sid} (digitized)")
                print(f"Wrote {digitized_dir}")
            else:
                print(f"Specimen {sid!r}: missing or invalid digitized force_deformation CSV.")
            return

        specimens = get_specimens_with_resampled()
        if sid in specimens:
            print(f"  {sid} (resampled)")
            plot_one_specimen(sid, row, out_dir)
            print(f"Wrote {out_dir}")
        elif is_unordered and not args.skip_digitized:
            if plot_one_digitized_unordered(sid, row, digitized_dir):
                print(f"  {sid} (digitized)")
                print(f"Wrote {digitized_dir}")
            else:
                print(f"Specimen {sid!r}: missing or invalid digitized force_deformation CSV.")
        else:
            print(f"Specimen {sid!r} not found in resampled data and not plotted as digitized.")
        return

    if not args.digitized_only:
        specimens = get_specimens_with_resampled()
        if specimens:
            for sid in specimens:
                print(f"  {sid} (resampled)")
                plot_one_specimen(sid, catalog_by_name.loc[sid], out_dir)
            print(f"Wrote {out_dir}")
        else:
            print("No specimens with resampled data found.")

    if args.skip_digitized:
        return

    digitized_written = 0
    for sid in unordered_names_all:
        if sid not in catalog_by_name.index:
            continue
        if plot_one_digitized_unordered(sid, catalog_by_name.loc[sid], digitized_dir):
            print(f"  {sid} (digitized)")
            digitized_written += 1
    if digitized_written:
        print(f"Wrote {digitized_dir} ({digitized_written} specimen(s))")
    elif not args.digitized_only:
        pass
    else:
        print("No digitized unordered specimens with valid force_deformation CSVs.")


if __name__ == "__main__":
    main()
