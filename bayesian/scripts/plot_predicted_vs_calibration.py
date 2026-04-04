"""
Overlay experimental (calibration) vs simulated force–deformation.

Runs ``model.run_analysis()`` with the displacement CSV path each time (OpenSees), writes ``predicted_force.csv``
(same layout as ``target_force.csv``: one comma-separated row, no header), then plots.
``--displacement`` is passed through as the path (default: ``target_displacement.csv``).

Primary axes: normalized δ/L_y and P/(f_y A_sc), matching repo overlay style.
Secondary (top) axis: deformation [in]; secondary (right) axis: force [kip].

  python scripts/plot_predicted_vs_calibration.py
  python scripts/plot_predicted_vs_calibration.py -o predicted_vs_calibration.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import run_analysis

from lib.specimen_config import load_specimen_config, resolve_path

# Match ``postprocess/plot_dimensions.py`` / individual-optimize overlays.
COLOR_EXPERIMENTAL = "#B0B0B0"
COLOR_SIMULATED = "#001F3F"
LINEWIDTH = 0.9
SAVE_DPI = 300
FIGSIZE_IN = (2.5, 2.25)

# Typography: 60% of prior overlay-style label size; ticks, legend, and axis labels match.
TEXT_PT = 6

# One sans-serif family for all text; mathtext uses matching DejaVu sans glyphs.
_RC_PLOT = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": TEXT_PT,
    "axes.titlesize": TEXT_PT,
    "xtick.labelsize": TEXT_PT,
    "ytick.labelsize": TEXT_PT,
    "legend.fontsize": TEXT_PT,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

NORM_STRAIN_LABEL = r"Axial strain, $\delta/L_y$ (%)"
NORM_FORCE_LABEL = r"Axial force, $P/\left(f_y A_{sc}\right)$"


def _fraction_to_percent_formatter(*, decimals: int = 2) -> mticker.FuncFormatter:
    def _fmt(x: float, _pos: int) -> str:
        if not np.isfinite(x):
            return ""
        p = 100.0 * x
        if decimals <= 0:
            return f"{p:.0f}"
        return f"{p:.{decimals}f}"

    return mticker.FuncFormatter(_fmt)


def _symmetric_limits(u: np.ndarray, v: np.ndarray, *, pad: float = 0.05) -> tuple[float, float]:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    m = float(np.nanmax(np.abs(np.concatenate([u[np.isfinite(u)], v[np.isfinite(v)]]))))
    if not np.isfinite(m) or m <= 0.0:
        return -1.0, 1.0
    w = m * (1.0 + pad)
    return -w, w


def main() -> None:
    default_cfg = _ROOT / "config" / "specimen_config.yaml"
    p = argparse.ArgumentParser(description="Plot calibration vs predicted F–u with dual physical axes.")
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Experimental CSV (default: paths.force_deformation from config)",
    )
    p.add_argument(
        "--displacement",
        type=Path,
        default=_ROOT / "target_displacement.csv",
        help="Deformation history CSV [in]; path passed to model.run_analysis()",
    )
    p.add_argument(
        "--predicted",
        type=Path,
        default=_ROOT / "predicted_force.csv",
        help="Write simulated forces here (one row, comma-separated)",
    )
    p.add_argument("-o", "--output", type=Path, default=_ROOT / "predicted_vs_calibration.png")
    args = p.parse_args()

    cfg = load_specimen_config(args.config)
    base = args.config.parent
    cal_path = args.calibration or resolve_path(cfg, "force_deformation", base)

    fy = float(cfg["fy_ksi"])
    A_sc = float(cfg["A_sc_in2"])
    L_y = float(cfg["L_y_in"])
    fy_a = fy * A_sc
    if fy_a <= 0.0 or L_y <= 0.0:
        raise ValueError("fy * A_sc and L_y must be positive")

    df_e = pd.read_csv(cal_path)
    for col in ("Deformation[in]", "Force[kip]"):
        if col not in df_e.columns:
            raise ValueError(f"Calibration CSV needs column {col!r}")

    D_e = df_e["Deformation[in]"].to_numpy(dtype=float)
    F_e = df_e["Force[kip]"].to_numpy(dtype=float)

    disp_path = args.displacement.expanduser().resolve()
    D_p, F_p = run_analysis(disp_path)
    D_p = np.asarray(D_p, dtype=float).reshape(-1)
    F_p = np.asarray(F_p, dtype=float).reshape(-1)
    if D_p.shape != F_p.shape:
        raise ValueError(
            f"model force length {F_p.size} != displacement length {D_p.size} (from {disp_path})"
        )

    pred_path = args.predicted.expanduser().resolve()
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(pred_path, F_p[np.newaxis, :], delimiter=",", fmt="%.17g")

    x_e = D_e / L_y
    y_e = F_e / fy_a
    x_p = D_p / L_y
    y_p = F_p / fy_a

    x_lo, x_hi = _symmetric_limits(
        np.concatenate([x_e[np.isfinite(x_e)], x_p[np.isfinite(x_p)]]),
        np.array([0.0]),
    )
    y_lo, y_hi = _symmetric_limits(
        np.concatenate([y_e[np.isfinite(y_e)], y_p[np.isfinite(y_p)]]),
        np.array([0.0]),
    )

    with plt.rc_context(_RC_PLOT):
        fig, ax = plt.subplots(figsize=FIGSIZE_IN, layout="constrained")
        ax.plot(
            x_e,
            y_e,
            color=COLOR_EXPERIMENTAL,
            alpha=0.95,
            linewidth=LINEWIDTH,
            linestyle="-",
            label="Experimental",
        )
        ax.plot(
            x_p,
            y_p,
            color=COLOR_SIMULATED,
            alpha=0.95,
            linewidth=LINEWIDTH,
            linestyle="--",
            label="Numerical",
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(NORM_STRAIN_LABEL)
        ax.set_ylabel(NORM_FORCE_LABEL)
        ax.xaxis.set_major_formatter(_fraction_to_percent_formatter(decimals=2))
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)

        ax_top = ax.twiny()
        ax_top.set_xlim(x_lo * L_y, x_hi * L_y)
        ax_top.set_xlabel(r"Deformation, $\delta$ [in]")
        ax_top.tick_params(axis="x", which="both", direction="in", labelsize=TEXT_PT)

        ax_right = ax.twinx()
        ax_right.set_ylim(y_lo * fy_a, y_hi * fy_a)
        ax_right.set_ylabel(r"Axial force, $P$ [kip]")
        ax_right.tick_params(axis="y", which="both", direction="in", labelsize=TEXT_PT)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=TEXT_PT)

        h, lab = ax.get_legend_handles_labels()
        fig.legend(
            h,
            lab,
            loc="outside upper center",
            ncol=2,
            fontsize=TEXT_PT,
            frameon=False,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        for spine in ax_top.spines.values():
            spine.set_linewidth(0.6)
        for spine in ax_right.spines.values():
            spine.set_linewidth(0.6)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=SAVE_DPI)
        plt.close(fig)
    print(f"Wrote {pred_path}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
