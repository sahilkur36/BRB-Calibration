"""
Optimize BRB parameters by minimizing cycle landmark loss plus optional energy loss.

Overview
--------
Given experimental deformation D_exp (resampled CSV) and force F_exp, the optimizer
adjusts a subset of SteelMPF / BRB parameters (default ``PARAMS_TO_OPTIMIZE`` in
``params_to_optimize.py``, optionally overridden per ``set_id`` by
``config/calibration/set_id_settings.csv``). For each
trial vector, the corotruss model runs with the same D_exp and returns simulated
force F_sim (same length as F_exp).

Landmark loss J_feat
--------------------
Per **weight cycle** (same partition as ``build_cycle_weight_ranges``), build up to
twelve experimental landmarks on ``(D_exp, F_exp)``: tension/compression first yield
(``F > f_y A_{sc}`` / ``F < -f_y A_{sc}``), global max/min force, ``F`` at ``D=0`` on each
yield-to-peak subpath, two mid-``D`` points per side, then ``D`` at ``F=0`` after each peak.
Slots 1--10 compare **force** at the experimental ``D`` on sim vs exp, normalized by ``S_F``.
Slots 11--12 compare **displacement** at unload (``F=0`` after each peak), normalized by ``S_D``.
Average squared error over contributing slots within each cycle, then cycle-weight by ``w_c``
(from per-``set_id`` settings in ``config/calibration/set_id_settings.csv`` by default;
override with ``--amplitude-weights`` / ``--no-amplitude-weights``).

    S_D = max(D_exp) - min(D_exp),  S_F = max(F_exp) - min(F_exp)   (fallbacks 1)

    J_feat = (sum over c of w_c * mean_p e_{c,p}) / (sum over c of w_c)   over cycles with >=1 matched slot

Energy loss J_E (optional when ``w_energy_l2`` or ``w_energy_l1`` is nonzero)
-------------------------------------------
Per cycle, E_c = |int F du| (trapezoidal rule on the resampled path).

    S_E = (max F_exp - min F_exp) * (max u - min u)   (fallback 1)
    J_E = mean over cycles of (E_c^sim - E_c^exp)^2, divided by S_E^2 (uniform over cycles; w_c is not used here)

Weight cycles
-------------
Cycle weights ``w_c`` for ``J_feat`` default to 1 unless you pass ``--amplitude-weights``; energy uses an unweighted per-cycle mean.

Total
-----
    J = weighted sum of active raw terms (``J_feat`` L2/L1, ``J_E`` L2/L1,
    ``J_binenv`` L2/L1) using per-``set_id`` weights in ``config/calibration/set_id_settings.csv``.

Run metrics (initial/final **raw** ``J_feat`` / ``J_E`` / binenv L2 and L1,
``J_total``, scalings ``S_F``/``S_D``/``S_E``, and active ``W_*`` weights) are
written to ``*_metrics.csv`` next to the optimized parameters CSV.

After each successful row, the **final** simulated force history is saved under
``results/calibration/individual_optimize/{output_stem}_simulated_force/{Name}_set{set_id}_force_history.npz``
(next to ``--output``; default ``--output`` is ``results/calibration/individual_optimize/optimized_brb_parameters.csv``), with arrays ``Deformation_in``, ``Force_kip_exp``, ``Force_kip_sim``
in NPZ and a companion CSV ``{Name}_set{set_id}_simulated.csv`` with ``Deformation[in],Force[kip],Force_sim[kip]``.

Seeding ``b_p`` / ``b_n``
-----------------------
After ``extract_bn_bp.py``, ``results/calibration/specimen_apparent_bn_bp.csv``
lists **apparent** segment slopes (median/mean per specimen). Those values are a practical
first guess for the SteelMPF ``b_p`` and ``b_n`` columns in the initial parameters CSV; they
are not the same mathematical object as the model parameters, but they often land in a
sensible range.

Optimizer and failures
----------------------
``scipy.optimize.minimize`` (L-BFGS-B). On simulation failure, J = FAILURE_PENALTY.

Parameterization for L-BFGS-B
-----------------------------
Finite ``L, U`` with ``U > L`` optimize in z in [0,1]; see ``lbfgsb_reparam.py``.
Output CSV values are always physical parameters.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.amplitude_mse_partition import (  # noqa: E402
    build_amplitude_weights,
    energy_mae_cycles,
    energy_mse_cycles,
    energy_scale_s_e,
    meta_to_dataframe,
)
from calibrate.cycle_feature_loss import (  # noqa: E402
    deformation_scale_s_d,
    feature_mae_cycles,
    feature_mse_cycles,
    load_p_y_kip_catalog,
)
from model.brace_geometry import compute_Q  # noqa: E402
from calibrate.calibration_io import metrics_dataframe  # noqa: E402
from calibrate.calibration_loss_settings import (  # noqa: E402
    DEFAULT_CALIBRATION_LOSS_SETTINGS,
    CalibrationLossSettings,
)
from calibrate.calibration_paths import (  # noqa: E402
    INITIAL_BRB_PARAMETERS_PATH,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    PARAM_LIMITS_CSV,
    PLOTS_INDIVIDUAL_OPTIMIZE,
    SET_ID_SETTINGS_CSV,
)
from calibrate.param_limits import bounds_dict_for  # noqa: E402
from calibrate.params_to_optimize import PARAMS_TO_OPTIMIZE  # noqa: E402
from calibrate.set_id_optimize_params import (  # noqa: E402
    resolve_loss_settings_for_set_id,
    resolve_optimize_params_for_set_id,
)
from calibrate.set_id_settings import load_set_id_optimize_and_loss  # noqa: E402
from calibrate.specimen_weights import catalog_metrics_fields, names_for_individual_optimize  # noqa: E402
from calibrate.digitized_unordered_eval_lib import compute_unordered_binenv_metrics  # noqa: E402
from calibrate.lbfgsb_reparam import (  # noqa: E402
    prepare_lbfgsb_parameterization,
    variable_params_from_optimizer_x,
)
from model.corotruss import run_simulation  # noqa: E402
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    AXES_SPINE_COLOR,
    AXES_SPINE_LINEWIDTH,
    LEGEND_FONT_SIZE_SMALL_PT,
    SAVE_DPI,
    SINGLE_FIGSIZE_IN,
    configure_matplotlib_style,
    style_major_tick_lines,
)
from postprocess.plot_specimens import (  # noqa: E402
    NORM_FORCE_LABEL,
    NORM_STRAIN_LABEL,
    _set_axes_frame,
    apply_normalized_fu_axes,
    set_symmetric_axes,
)
from specimen_catalog import (  # noqa: E402
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_resampled_force_deformation_csv,
)

configure_matplotlib_style()
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# ----- Config (adjust as needed); see params_to_optimize.py for PARAMS_TO_OPTIMIZE -----

# --- Cycle weights w_c for J_feat: per-set defaults from `set_id_settings.csv`;
# override with `--amplitude-weights` / `--no-amplitude-weights`.
AMPLITUDE_WEIGHTS_ARG_HELP = (
    "Override per-set loss settings: use amplitude-based cycle weights w_c for J_feat "
    "(omit both --amplitude-weights and --no-amplitude-weights to use per-set CSV values)."
)
AMPLITUDE_WEIGHT_POWER = DEFAULT_CALIBRATION_LOSS_SETTINGS.amplitude_weight_power
AMPLITUDE_WEIGHT_EPS = DEFAULT_CALIBRATION_LOSS_SETTINGS.amplitude_weight_eps
DEBUG_PARTITION = False  # if True, assert disjoint cover 0..n-1 for cycle partition

# Box limits for L-BFGS-B: ``config/calibration/params_limits.csv`` (``calibration_paths.PARAM_LIMITS_CSV``).
# Omitted parameters in that file are unbounded. Notes on SteelMPF R-degradation (cR1, cR2, R0)
# live in comments inside ``params_limits.csv``.

FAILURE_PENALTY = 1e6


@dataclass(frozen=True)
class LossBreakdown:
    """Raw metrics for one (D_exp, F_exp, F_sim) path-ordered evaluation and weighted ``j_total``."""

    j_feat_l2: float
    j_feat_l1: float
    j_e_l2: float
    j_e_l1: float
    j_total: float
    binenv_l2: float
    binenv_l1: float


def _weighted_objective_from_raw(loss: CalibrationLossSettings, raw: dict[str, float]) -> float | None:
    """Return J_total or None if any active weight pairs with a non-finite raw value."""
    keys_weights = [
        ("j_feat_l2", loss.w_feat_l2),
        ("j_feat_l1", loss.w_feat_l1),
        ("j_e_l2", loss.w_energy_l2),
        ("j_e_l1", loss.w_energy_l1),
        ("binenv_l2", loss.w_unordered_binenv_l2),
        ("binenv_l1", loss.w_unordered_binenv_l1),
    ]
    total = 0.0
    for k, w in keys_weights:
        if abs(w) < 1e-300:
            continue
        x = raw[k]
        if not math.isfinite(x):
            return None
        total += w * x
    return float(total)


def _loss_weight_active(w: float) -> bool:
    """True if this weight is included in ``_weighted_objective_from_raw`` (nonzero coefficient)."""
    return abs(float(w)) >= 1e-300

INITIAL_PARAMS_PATH = INITIAL_BRB_PARAMETERS_PATH
OUTPUT_CSV = OPTIMIZED_BRB_PARAMETERS_PATH
CYCLE_WEIGHT_PLOTS_DIR = PLOTS_INDIVIDUAL_OPTIMIZE / "cycle_weights"


def simulated_force_history_dir(out_csv: Path) -> Path:
    """Directory for per-row optimized sim force histories: next to ``out_csv``, named from its stem."""
    return out_csv.parent / f"{out_csv.stem}_simulated_force"


def save_simulated_force_history(
    out_dir: Path,
    specimen_id: str,
    set_id: int | str,
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray | None,
) -> Path | None:
    """
    Write compressed NPZ with resampled experimental deformation/force and final simulated force.

    Arrays align index-wise with ``data/resampled/{specimen_id}/force_deformation.csv``. Returns path or None if skipped.
    """
    if F_sim is None or F_sim.size == 0:
        return None
    D_exp = np.asarray(D_exp, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if F_sim.shape != F_exp.shape or D_exp.shape != F_exp.shape:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_set = str(set_id).replace(" ", "_")
    path = out_dir / f"{specimen_id}_set{safe_set}_force_history.npz"
    np.savez_compressed(
        path,
        Deformation_in=D_exp,
        Force_kip_exp=F_exp,
        Force_kip_sim=F_sim,
    )
    return path


def save_simulated_force_history_csv(
    out_dir: Path,
    specimen_id: str,
    set_id: int | str,
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray | None,
) -> Path | None:
    """
    Write CSV with ``Deformation[in],Force[kip],Force_sim[kip]`` (same deformation/exp headers
    as ``data/resampled/{Name}/force_deformation.csv``). Returns path or None if skipped.
    """
    if F_sim is None or F_sim.size == 0:
        return None
    D_exp = np.asarray(D_exp, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if F_sim.shape != F_exp.shape or D_exp.shape != F_exp.shape:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_set = str(set_id).replace(" ", "_")
    path = out_dir / f"{specimen_id}_set{safe_set}_simulated.csv"
    pd.DataFrame(
        {
            "Deformation[in]": D_exp,
            "Force[kip]": F_exp,
            "Force_sim[kip]": F_sim,
        }
    ).to_csv(path, index=False)
    return path


# Significant figures for numeric columns in optimized_brb_parameters.csv (not decimal places).
OUTPUT_CSV_SIGFIGS = 6

def force_scale_s_f(F_exp: np.ndarray) -> float:
    """S_F = max(F_exp) - min(F_exp); fallback 1.0."""
    f = np.asarray(F_exp, dtype=float)
    r = float(np.nanmax(f) - np.nanmin(f))
    if not np.isfinite(r) or r <= 0.0:
        return 1.0
    return r


def _float_round_sigfigs(x: float, n: int) -> float:
    """Round ``x`` to ``n`` significant figures (not ``n`` decimal places)."""
    if not math.isfinite(x):
        return x
    if x == 0.0:
        return 0.0
    exp = math.floor(math.log10(abs(x)))
    return round(x, n - 1 - exp)


def _dataframe_for_param_csv(df: pd.DataFrame, *, sigfigs: int) -> pd.DataFrame:
    """Format numeric columns for human-readable CSV: ``sigfigs`` significant figures."""
    out = df.copy()
    int_cols = {"ID", "set_id"}
    for col in out.columns:
        if col == "Name":
            continue
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if col in int_cols:
            out[col] = pd.to_numeric(s, errors="coerce").round(0).astype("Int64")
            if not out[col].isna().any():
                out[col] = out[col].astype(np.int64)
            continue

        def _cell(v: object) -> float:
            """Format one Markdown table cell (numeric or placeholder)."""
            if pd.isna(v):
                return np.nan
            return _float_round_sigfigs(float(v), sigfigs)

        out[col] = s.map(_cell)
    return out


def _row_to_sim_params(prow: pd.Series) -> dict:
    """Build kwargs for run_simulation. Keys match CSV columns after ID, Name, set_id (SteelMPF order)."""
    keys = ("L_T", "L_y", "A_sc", "A_t", "fyp", "fyn", "E", "b_p", "b_n", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4")
    return {k: float(prow[k]) for k in keys}


def simulate_and_loss_breakdown(
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    amp_meta: list[dict],
    *,
    s_d: float,
    loss: CalibrationLossSettings,
    fy_ksi: float,
    a_sc: float,
    L_T: float,
    L_y: float,
    A_t: float,
    E_ksi: float,
    exp_landmark_cache: dict | None = None,
    full_metrics: bool = True,
) -> LossBreakdown | None:
    """
    Compute raw L1/L2 metric family members and ``j_total`` from ``loss`` weights.

    ``exp_landmark_cache``: optional dict reused across calls (e.g. L-BFGS iterations) to avoid
    recomputing experimental cycle landmarks per iteration; keys use array ``id()`` so
    experimental ``D_exp``/``F_exp`` must keep stable object identity while caching.

    When ``full_metrics`` is False, only metrics with **nonzero** loss weights are computed (plus any
    co-computed pair from the same helper, e.g. binenv L1/L2). Omitted entries are left as NaN in
    the breakdown; ``j_total`` still matches the weighted sum of active terms. Use ``full_metrics=True``
    for reporting (initial/final CSV rows, eval scripts).

    Returns None if ``F_sim`` is invalid, or if energy is FAILURE_PENALTY while that energy term is
    computed and its weight is active, or if any **active** weight multiplies a non-finite raw value.
    """
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if F_sim.shape != F_exp.shape or not np.all(np.isfinite(F_sim)):
        return None
    s_f = force_scale_s_f(F_exp)
    nan = float("nan")

    dy_in: float | None = None
    try:
        Q = float(compute_Q(float(L_T), float(L_y), float(a_sc), float(A_t)))
        E_hat = Q * float(E_ksi)
        if np.isfinite(E_hat) and E_hat > 0.0:
            dy = (float(fy_ksi) / E_hat) * float(L_T)
            if np.isfinite(dy) and dy > 0.0:
                dy_in = float(dy)
    except Exception:
        dy_in = None

    if full_metrics or _loss_weight_active(loss.w_feat_l2):
        j_feat_l2 = feature_mse_cycles(
            D_exp,
            F_exp,
            F_sim,
            amp_meta,
            s_d=float(s_d),
            s_f=float(s_f),
            fy_ksi=float(fy_ksi),
            A_sc=float(a_sc),
            dy_in=dy_in,
            exp_landmark_cache=exp_landmark_cache,
        )
    else:
        j_feat_l2 = nan

    if full_metrics or _loss_weight_active(loss.w_feat_l1):
        j_feat_l1 = feature_mae_cycles(
            D_exp,
            F_exp,
            F_sim,
            amp_meta,
            s_d=float(s_d),
            s_f=float(s_f),
            fy_ksi=float(fy_ksi),
            A_sc=float(a_sc),
            dy_in=dy_in,
            exp_landmark_cache=exp_landmark_cache,
        )
    else:
        j_feat_l1 = nan

    compute_e_l2 = full_metrics or _loss_weight_active(loss.w_energy_l2)
    compute_e_l1 = full_metrics or _loss_weight_active(loss.w_energy_l1)
    if compute_e_l2:
        j_e_l2 = energy_mse_cycles(
            D_exp,
            F_exp,
            F_sim,
            amp_meta,
            failure_penalty=FAILURE_PENALTY,
        )
    else:
        j_e_l2 = nan
    if compute_e_l1:
        j_e_l1 = energy_mae_cycles(
            D_exp,
            F_exp,
            F_sim,
            amp_meta,
            failure_penalty=FAILURE_PENALTY,
        )
    else:
        j_e_l1 = nan
    if compute_e_l2 and j_e_l2 == FAILURE_PENALTY:
        return None
    if compute_e_l1 and j_e_l1 == FAILURE_PENALTY:
        return None

    if full_metrics or _loss_weight_active(loss.w_unordered_binenv_l2) or _loss_weight_active(
        loss.w_unordered_binenv_l1
    ):
        binenv_l2, binenv_l1 = compute_unordered_binenv_metrics(D_exp, F_exp, D_exp, F_sim)
    else:
        binenv_l2, binenv_l1 = nan, nan

    raw = {
        "j_feat_l2": float(j_feat_l2),
        "j_feat_l1": float(j_feat_l1),
        "j_e_l2": float(j_e_l2),
        "j_e_l1": float(j_e_l1),
        "binenv_l2": float(binenv_l2),
        "binenv_l1": float(binenv_l1),
    }
    j_total = _weighted_objective_from_raw(loss, raw)
    if j_total is None:
        return None
    return LossBreakdown(
        j_feat_l2=raw["j_feat_l2"],
        j_feat_l1=raw["j_feat_l1"],
        j_e_l2=raw["j_e_l2"],
        j_e_l1=raw["j_e_l1"],
        j_total=j_total,
        binenv_l2=raw["binenv_l2"],
        binenv_l1=raw["binenv_l1"],
    )


def _loss_weight_snapshot(loss: CalibrationLossSettings) -> dict[str, float]:
    return {
        "W_FEAT_L2": float(loss.w_feat_l2),
        "W_FEAT_L1": float(loss.w_feat_l1),
        "W_ENERGY_L2": float(loss.w_energy_l2),
        "W_ENERGY_L1": float(loss.w_energy_l1),
        "W_UNORDERED_BINENV_L2": float(loss.w_unordered_binenv_l2),
        "W_UNORDERED_BINENV_L1": float(loss.w_unordered_binenv_l1),
    }


def _metrics_dict_nan_prefix(prefix: str) -> dict[str, float]:
    """All-NaN block for a specimen row that was never evaluated."""
    p = prefix
    z = float("nan")
    return {
        f"{p}_J_feat_raw": z,
        f"{p}_J_feat_l1_raw": z,
        f"{p}_J_E_raw": z,
        f"{p}_J_E_l1_raw": z,
        f"{p}_J_total": z,
        f"{p}_unordered_J_binenv": z,
        f"{p}_unordered_J_binenv_l1": z,
    }


def _metrics_dict_for_breakdown(
    bd: LossBreakdown | None,
    loss: CalibrationLossSettings,
    prefix: str,
) -> dict[str, float]:
    """Map ``LossBreakdown`` fields to ``initial_*`` / ``final_*`` metrics CSV columns."""
    p = prefix
    if bd is None:
        return {
            f"{p}_J_feat_raw": float(FAILURE_PENALTY),
            f"{p}_J_feat_l1_raw": float("nan"),
            f"{p}_J_E_raw": float(FAILURE_PENALTY),
            f"{p}_J_E_l1_raw": float("nan"),
            f"{p}_J_total": float(FAILURE_PENALTY),
            f"{p}_unordered_J_binenv": float("nan"),
            f"{p}_unordered_J_binenv_l1": float("nan"),
        }
    return {
        f"{p}_J_feat_raw": float(bd.j_feat_l2),
        f"{p}_J_feat_l1_raw": float(bd.j_feat_l1),
        f"{p}_J_E_raw": float(bd.j_e_l2),
        f"{p}_J_E_l1_raw": float(bd.j_e_l1),
        f"{p}_J_total": float(bd.j_total),
        f"{p}_unordered_J_binenv": float(bd.binenv_l2),
        f"{p}_unordered_J_binenv_l1": float(bd.binenv_l1),
    }


def optimize_one_specimen(
    specimen_id: str,
    prow: pd.Series,
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    amp_meta: list[dict],
    params_to_optimize: list[str],
    bounds_dict: dict[str, tuple[float, float]],
    *,
    p_y_ref: float,
    s_d: float,
    loss: CalibrationLossSettings,
) -> tuple[pd.Series, LossBreakdown | None, LossBreakdown | None, np.ndarray | None, np.ndarray | None]:
    """Return optimized row, initial/final ``LossBreakdown`` (or None), and simulated forces."""
    fixed_params = _row_to_sim_params(prow)
    fy_ksi = float(prow["fyp"])
    a_sc = float(prow["A_sc"])
    use_norm, Ls, Us, x0, scipy_bounds = prepare_lbfgsb_parameterization(
        params_to_optimize,
        bounds_dict,
        prow,
        specimen_hint=specimen_id,
    )
    exp_landmark_cache: dict = {}

    def run_breakdown(
        x: np.ndarray, *, full_metrics: bool
    ) -> tuple[LossBreakdown, np.ndarray] | None:
        params = {**fixed_params, **variable_params_from_optimizer_x(x, params_to_optimize, use_norm, Ls, Us)}
        try:
            F_sim = run_simulation(D_exp, **params)
        except Exception:
            return None
        F_sim = np.asarray(F_sim, dtype=float)
        bd = simulate_and_loss_breakdown(
            D_exp,
            F_exp,
            F_sim,
            amp_meta,
            s_d=s_d,
            loss=loss,
            fy_ksi=fy_ksi,
            a_sc=a_sc,
            L_T=float(prow["L_T"]),
            L_y=float(prow["L_y"]),
            A_t=float(prow["A_t"]),
            E_ksi=float(prow["E"]),
            exp_landmark_cache=exp_landmark_cache,
            full_metrics=full_metrics,
        )
        if bd is None:
            return None
        return bd, F_sim

    bd0_pair = run_breakdown(x0, full_metrics=True)
    F_sim_initial: np.ndarray | None
    bd_initial: LossBreakdown | None
    if bd0_pair is None:
        bd_initial = None
        F_sim_initial = None
    else:
        bd_initial, F_sim_initial = bd0_pair

    def fun(x: np.ndarray) -> float:
        """Scalar objective for L-BFGS-B."""
        pair = run_breakdown(x, full_metrics=False)
        if pair is None:
            return FAILURE_PENALTY
        return pair[0].j_total

    res = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        bounds=scipy_bounds,
        options={"ftol": 1e-8, "gtol": 1e-6},
    )

    physical = variable_params_from_optimizer_x(
        res.x, params_to_optimize, use_norm, Ls, Us
    )
    out_row = prow.copy()
    for name in params_to_optimize:
        out_row[name] = physical[name]

    F_sim_final: np.ndarray | None
    bd_final: LossBreakdown | None
    bd1_pair = run_breakdown(res.x, full_metrics=True)
    if bd1_pair is None:
        bd_final = None
        F_sim_final = None
    else:
        bd_final, F_sim_final = bd1_pair

    return (out_row, bd_initial, bd_final, F_sim_initial, F_sim_final)


def _style_twin_y_axes_frame(
    ax: plt.Axes,
    ax2: plt.Axes,
    *,
    linewidth: float = AXES_SPINE_LINEWIDTH,
) -> None:
    """Rectangle on the data plane: left/bottom/top on host, right spine for twin y (no doubled edges)."""
    col = AXES_SPINE_COLOR
    ax.set_facecolor("white")
    ax2.set_facecolor("none")
    for a in (ax, ax2):
        a.patch.set_linewidth(0.0)
        a.patch.set_edgecolor("none")
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom", "top"):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_color(col)
        sp.set_linewidth(linewidth)
    for side in ("left", "bottom", "top"):
        ax2.spines[side].set_visible(False)
    sp_r = ax2.spines["right"]
    sp_r.set_visible(True)
    sp_r.set_color(col)
    sp_r.set_linewidth(linewidth)
    ax.tick_params(
        axis="x",
        which="major",
        direction="in",
        bottom=True,
        top=True,
        left=False,
        right=False,
        labelbottom=True,
        labeltop=False,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
    )
    ax2.tick_params(
        axis="y",
        which="major",
        direction="in",
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
    )
    ax2.tick_params(
        axis="x",
        which="major",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    style_major_tick_lines(ax, color=col, linewidth=linewidth)
    style_major_tick_lines(ax2, color=col, linewidth=linewidth)


def plot_amplitude_diagnostics(
    meta: list[dict],
    specimen_id: str,
    set_id: int | str,
    out_dir: Path,
) -> None:
    """Cycle index vs amplitude and cycle weight w_c."""
    if not meta:
        return
    idx = np.arange(len(meta))
    amps = [float(m.get("amp", 0.0)) for m in meta]
    w_cs = [float(m.get("w_c", 0.0)) for m in meta]
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
    fig.patch.set_facecolor("white")
    ax.plot(idx, amps, "o-", label=r"$A_c$", markersize=3)
    ax.set_xlabel("Cycle index")
    ax.set_ylabel("Amplitude |u| [in]")
    ax2 = ax.twinx()
    ax2.plot(idx, w_cs, "s--", color="C1", label=r"$w_c$", markersize=3)
    ax2.set_ylabel(r"$w_c$")
    ax.grid(True, alpha=0.3)
    _style_twin_y_axes_frame(ax, ax2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="outside upper center",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE_SMALL_PT,
        frameon=False,
    )
    fig.savefig(out_dir / f"{specimen_id}_set{set_id}_amplitude_diagnostics.png", dpi=SAVE_DPI)
    plt.close(fig)


def plot_cycle_weight_hysteresis(
    specimen_id: str,
    set_id: int | str,
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    pointwise_weights: np.ndarray,
    meta: list[dict],
    out_dir: Path,
    *,
    f_yc: float,
    A_c: float,
    L_y: float,
) -> None:
    """Plot hysteresis colored by cycle weight w_i (normalized P–δ); mark cycle starts; save cycle CSV."""
    w = np.asarray(pointwise_weights, dtype=float)
    fyA = float(f_yc) * float(A_c)
    L_yf = float(L_y)
    if fyA <= 0 or L_yf <= 0 or not np.isfinite(fyA) or not np.isfinite(L_yf):
        fyA, L_yf = 1.0, 1.0
    D_n = np.asarray(D_exp, dtype=float) / L_yf
    F_n = np.asarray(F_exp, dtype=float) / fyA
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    sc = ax.scatter(
        D_n,
        F_n,
        c=w,
        cmap="viridis",
        s=4,
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label=r"Cycle weight $w_i$")
    for m in meta:
        s = int(m["start"])
        if 0 <= s < len(D_exp):
            ax.axvline(
                float(D_n[s]),
                color="red",
                alpha=0.25,
                linewidth=0.8,
            )
    set_symmetric_axes(ax, D_n, F_n)
    ax.set_xlabel(NORM_STRAIN_LABEL)
    ax.set_ylabel(NORM_FORCE_LABEL)
    apply_normalized_fu_axes(ax)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.4)
    ax.axvline(0, color="k", linewidth=0.4)
    _set_axes_frame(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{specimen_id}_set{set_id}_cycle_weights.png", dpi=SAVE_DPI)
    plt.close(fig)

    df_meta = meta_to_dataframe(meta)
    df_meta.to_csv(
        out_dir / f"{specimen_id}_set{set_id}_amplitude_cycles.csv",
        index=False,
    )
    plot_amplitude_diagnostics(meta, specimen_id, set_id, out_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Optimize BRB parameters: weighted sum of raw J_feat / J_E / J_binenv (L2/L1) "
            "per-set via config/calibration/set_id_settings.csv unless overridden."
        ),
    )
    parser.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="Single specimen Name (e.g. CB225). Default: all specimens with resampled data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output CSV path. Default: {OUTPUT_CSV}",
    )
    _init_rel = INITIAL_PARAMS_PATH
    try:
        _init_rel = INITIAL_PARAMS_PATH.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    parser.add_argument(
        "--initial-params",
        type=Path,
        default=INITIAL_PARAMS_PATH,
        help=(
            f"Initial BRB parameters CSV (Name, SteelMPF columns). Default: {_init_rel}. "
            "Regenerate with scripts/calibrate/build_initial_brb_parameters.py after extract_bn_bp.py "
            "(calibration set_ids and b_p/b_n from config/calibration/set_id_settings.csv; see build_initial_brb_parameters.py)."
        ),
    )
    _pl_rel = PARAM_LIMITS_CSV
    try:
        _pl_rel = PARAM_LIMITS_CSV.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    parser.add_argument(
        "--param-limits",
        type=Path,
        default=None,
        help=(
            "Parameter box limits CSV (parameter, lower, upper). "
            f"Default: {_pl_rel}."
        ),
    )
    _sip_rel = SET_ID_SETTINGS_CSV
    try:
        _sip_rel = SET_ID_SETTINGS_CSV.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    parser.add_argument(
        "--set-id-settings",
        type=Path,
        default=None,
        help=(
            "Per-set_id settings CSV (steel seeds + optimize_params + loss weights). "
            f"Default path if omitted: {_sip_rel}."
        ),
    )
    parser.add_argument(
        "--amplitude-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=AMPLITUDE_WEIGHTS_ARG_HELP,
    )
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else OUTPUT_CSV
    initial_path = Path(args.initial_params).expanduser().resolve()

    opt_csv_path = (
        Path(args.set_id_settings).expanduser().resolve()
        if args.set_id_settings is not None
        else None
    )
    optimize_by_set_id, loss_by_set_id = load_set_id_optimize_and_loss(opt_csv_path)
    if optimize_by_set_id or loss_by_set_id:
        _sip_show = opt_csv_path if opt_csv_path is not None else SET_ID_SETTINGS_CSV
        print(f"Per-set_id settings CSV: {_sip_show}")
    else:
        print(
            "Per-set_id settings: (file missing or empty) — using defaults: "
            f"PARAMS_TO_OPTIMIZE={', '.join(PARAMS_TO_OPTIMIZE)}; "
            "loss settings = DEFAULT_CALIBRATION_LOSS_SETTINGS"
        )
    print(f"Initial parameters CSV: {initial_path}")
    _limits_p = Path(args.param_limits).expanduser().resolve() if args.param_limits else PARAM_LIMITS_CSV
    print(f"Parameter limits CSV: {_limits_p}")
    print("Loss reporting: raw J_feat (L2/L1), J_E (L2/L1), J_binenv (L2/L1); J_total = weighted sum.")
    if args.amplitude_weights is not None:
        print(
            "J_feat cycle weights: "
            + (
                "amplitude w_c (CLI override)"
                if bool(args.amplitude_weights)
                else "uniform w_c=1 (CLI override)"
            )
        )

    params_df = pd.read_csv(initial_path)
    if "Name" not in params_df.columns:
        raise RuntimeError(f"'Name' column not found in {initial_path}")
    if params_df.empty:
        raise RuntimeError(f"No rows in {initial_path}")

    catalog_df = read_catalog()
    _catalog_by_name = catalog_df.set_index("Name")

    allowed = names_for_individual_optimize()
    resampled_stems = path_ordered_resampled_force_csv_stems(
        catalog_df, project_root=_PROJECT_ROOT
    )
    available = sorted(
        set(params_df["Name"].astype(str)) & resampled_stems & set(allowed)
    )
    if not available:
        print(
            "No specimens found with both initial parameters, resampled data, and "
            "individual_optimize=true (optimized columns will be NaN for the full specimen set)."
        )

    if args.specimen:
        if args.specimen not in available:
            print(
                f"Specimen {args.specimen} not available. "
                f"Must be in {initial_path.name} and have data/resampled/{{Name}}/force_deformation.csv"
            )
            return
        specimens = [args.specimen]
    else:
        specimens = available

    bounds_cache: dict[tuple[str, ...], dict[str, tuple[float, float]]] = {}

    def _bounds_for_active(active: list[str]) -> dict[str, tuple[float, float]]:
        key = tuple(active)
        if key not in bounds_cache:
            bounds_cache[key] = bounds_dict_for(list(active), limits_path=args.param_limits)
        return bounds_cache[key]

    out_rows = []
    metrics_rows: list[dict] = []
    for sid in specimens:
        csv_path = resolve_resampled_force_deformation_csv(sid, _PROJECT_ROOT)
        if csv_path is None or not csv_path.is_file():
            print(f"  Skipping {sid}: missing resampled force_deformation.csv")
            continue
        df = pd.read_csv(csv_path)
        if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
            print(f"  Skipping {sid}: missing Force[kip] or Deformation[in]")
            continue

        D_exp = df["Deformation[in]"].to_numpy(dtype=float)
        F_exp = df["Force[kip]"].to_numpy(dtype=float)

        loaded = load_cycle_points_resampled(sid)
        if loaded is not None:
            points, _segments = loaded
        else:
            points, _segments = find_cycle_points(df)
        s_f_ref = force_scale_s_f(F_exp)
        s_d_ref = deformation_scale_s_d(D_exp)
        s_e_ref = energy_scale_s_e(D_exp, F_exp)
        rows_for_specimen = params_df[params_df["Name"].astype(str) == sid]
        if rows_for_specimen.empty:
            print(f"  Skipping {sid}: no row in {initial_path.name}")
            continue

        p_y_specimen = load_p_y_kip_catalog(
            _PROJECT_ROOT,
            sid,
            float(rows_for_specimen.iloc[0]["fyp"]),
            float(rows_for_specimen.iloc[0]["A_sc"]),
        )

        CYCLE_WEIGHT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        for _, prow in rows_for_specimen.iterrows():
            try:
                active = resolve_optimize_params_for_set_id(
                    optimize_by_set_id,
                    prow.get("set_id"),
                    list(PARAMS_TO_OPTIMIZE),
                )
                loss = resolve_loss_settings_for_set_id(loss_by_set_id, prow.get("set_id"))
                use_amp_w = (
                    bool(args.amplitude_weights)
                    if args.amplitude_weights is not None
                    else loss.use_amplitude_weights
                )
                mse_weights, amp_meta = build_amplitude_weights(
                    D_exp,
                    points,
                    p=loss.amplitude_weight_power,
                    eps=loss.amplitude_weight_eps,
                    debug_partition=DEBUG_PARTITION,
                    use_amplitude_weights=use_amp_w,
                )
                n_cycles = len(amp_meta)
                bd_local = _bounds_for_active(active)
                (
                    out_row,
                    bd_initial,
                    bd_final,
                    _F_sim_initial,
                    F_sim_final,
                ) = optimize_one_specimen(
                    sid,
                    prow,
                    D_exp,
                    F_exp,
                    amp_meta,
                    active,
                    bd_local,
                    p_y_ref=p_y_specimen,
                    s_d=s_d_ref,
                    loss=loss,
                )
                out_rows.append(out_row)
                set_id = prow.get("set_id", "?")
                sim_hist_dir = simulated_force_history_dir(out_path)
                saved_npz = save_simulated_force_history(
                    sim_hist_dir,
                    sid,
                    set_id,
                    D_exp,
                    F_exp,
                    F_sim_final,
                )
                saved_csv = save_simulated_force_history_csv(
                    sim_hist_dir,
                    sid,
                    set_id,
                    D_exp,
                    F_exp,
                    F_sim_final,
                )
                if saved_npz is not None:
                    print(f"    saved {saved_npz.name}")
                if saved_csv is not None:
                    print(f"    saved {saved_csv.name}")
                mi = _metrics_dict_for_breakdown(bd_initial, loss, "initial")
                mf = _metrics_dict_for_breakdown(bd_final, loss, "final")
                j1 = mf["final_J_total"]
                cw = 14
                print(f"  {sid} (set {set_id})  optimize: {', '.join(active)}")
                print(
                    f"    {'':10} {'J_feat_L2':>{cw}} {'J_feat_L1':>{cw}} {'J_E_L2':>{cw}} {'J_E_L1':>{cw}} "
                    f"{'J_total':>{cw}}"
                )
                print(
                    f"    {'initial':10} {mi['initial_J_feat_raw']:>{cw}.6g} {mi['initial_J_feat_l1_raw']:>{cw}.6g} "
                    f"{mi['initial_J_E_raw']:>{cw}.6g} {mi['initial_J_E_l1_raw']:>{cw}.6g} {mi['initial_J_total']:>{cw}.6g}"
                )
                print(
                    f"    {'final':10} {mf['final_J_feat_raw']:>{cw}.6g} {mf['final_J_feat_l1_raw']:>{cw}.6g} "
                    f"{mf['final_J_E_raw']:>{cw}.6g} {mf['final_J_E_l1_raw']:>{cw}.6g} {mf['final_J_total']:>{cw}.6g}"
                )
                print(
                    f"    {'report':10} J_binenv_L2={mf['final_unordered_J_binenv']:.6g}  "
                    f"J_binenv_L1={mf['final_unordered_J_binenv_l1']:.6g}"
                )
                wsnap = _loss_weight_snapshot(loss)
                metrics_rows.append(
                    {
                        "Name": sid,
                        "set_id": set_id,
                        "specimen_weight": 1.0,
                        "contributes_to_aggregate": True,
                        **catalog_metrics_fields(sid, _catalog_by_name),
                        "weight_config": "per_specimen",
                        "calibration_stage": "optimize",
                        "aggregate_by_set_id": False,
                        **mi,
                        **mf,
                        **wsnap,
                        "S_F": s_f_ref,
                        "S_D": s_d_ref,
                        "S_E": s_e_ref,
                        "P_y_ref": p_y_specimen,
                        "n_cycles": n_cycles,
                        "success": j1 < FAILURE_PENALTY * 0.5,
                    }
                )
                cr = _catalog_by_name.loc[sid]
                plot_cycle_weight_hysteresis(
                    sid,
                    set_id,
                    D_exp,
                    F_exp,
                    mse_weights,
                    amp_meta,
                    CYCLE_WEIGHT_PLOTS_DIR,
                    f_yc=float(cr["f_yc_ksi"]),
                    A_c=float(cr["A_c_in2"]),
                    L_y=float(cr["L_y_in"]),
                )
            except Exception as e:
                print(f"  {sid} failed: {e}")
                raise

    opt_map: dict[tuple[str, int], pd.Series] = {}
    for item in out_rows:
        s = pd.Series(item) if not isinstance(item, pd.Series) else item
        opt_map[(str(s["Name"]), int(s["set_id"]))] = s

    full_rows: list[pd.Series] = []
    for _, prow in params_df.iterrows():
        k = (str(prow["Name"]), int(prow["set_id"]))
        if k in opt_map:
            full_rows.append(opt_map[k])
        else:
            row = prow.copy()
            active_skip = resolve_optimize_params_for_set_id(
                optimize_by_set_id,
                prow.get("set_id"),
                list(PARAMS_TO_OPTIMIZE),
            )
            for p in active_skip:
                row[p] = np.nan
            full_rows.append(row)

    out_df = pd.DataFrame(full_rows)
    out_df = out_df[params_df.columns.tolist()]
    out_df = _dataframe_for_param_csv(out_df, sigfigs=OUTPUT_CSV_SIGFIGS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows, one per initial parameter row)")

    metrics_keys = {(str(m["Name"]), int(m["set_id"])) for m in metrics_rows}
    for _, prow in params_df.iterrows():
        k = (str(prow["Name"]), int(prow["set_id"]))
        if k in metrics_keys:
            continue
        sid = str(prow["Name"])
        set_id = int(prow["set_id"])
        nan_bd: dict[str, float] = {}
        for px in ("initial", "final"):
            nan_bd.update(_metrics_dict_nan_prefix(px))
        metrics_rows.append(
            {
                "Name": sid,
                "set_id": set_id,
                "specimen_weight": 0.0,
                "contributes_to_aggregate": False,
                **catalog_metrics_fields(sid, _catalog_by_name),
                "weight_config": "per_specimen",
                "calibration_stage": "optimize",
                "aggregate_by_set_id": False,
                **nan_bd,
                **_loss_weight_snapshot(loss),
                "S_F": np.nan,
                "S_D": np.nan,
                "S_E": np.nan,
                "P_y_ref": np.nan,
                "n_cycles": 0,
                "success": False,
            }
        )

    mpath = out_path.parent / f"{out_path.stem}_metrics.csv"
    metrics_dataframe(metrics_rows).to_csv(mpath, index=False)
    print(f"Wrote {mpath}")


if __name__ == "__main__":
    main()
