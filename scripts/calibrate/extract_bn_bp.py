"""
Extract b_n and b_p per specimen from resampled force-deformation data using
cycle points. For each segment from zero_def to max_def (or min_def), find
the first sample where ``|F| >= PLASTIC_STRESS_RATIO_FY * f_y A_sc``, fit hardening
slope ksh from there to peak, and b = ksh / kinit (k_init = E_hat*A_sc/L_T). Aggregate
per specimen as mean and median. Save catalog-like CSV with Q, E_hat, and
computed parameters (no optimization). Default output:
``results/calibration/specimen_apparent_bn_bp.csv`` (path-ordered rows from ``data/resampled``;
digitized scatter-cloud rows from envelope fit, **mean** only--median/quartiles **NaN**).
Apparent plastic onset on each zero-to-peak segment uses ``|F| >= PLASTIC_STRESS_RATIO_FY * f_y A_sc``
(landmark loss uses ``f_y A_sc`` without this ratio; apparent-b extraction keeps 1.1 here.)
Input resampled: ``data/resampled/{Name}/force_deformation.csv``. Scatter: filtered cloud if present else raw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))  # calibration_paths, digitized_unordered_bn
sys.path.insert(0, str(_SCRIPTS / "postprocess"))  # so cycle_points can "from load_raw import ..."
sys.path.insert(0, str(_SCRIPTS))
from calibration_paths import BRB_SPECIMENS_CSV, SPECIMEN_APPARENT_BN_BP_PATH  # noqa: E402
from cycle_feature_loss import PLASTIC_STRESS_RATIO_FY  # noqa: E402
from digitized_unordered_bn import compute_envelope_bn_unordered  # noqa: E402
from model.corotruss import compute_Q
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled
from specimen_catalog import (  # noqa: E402
    force_deformation_unordered_csv_path,
    list_names_digitized_unordered,
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_filtered_force_deformation_csv,
    resolve_resampled_force_deformation_csv,
)

E_ksi = 29000.0
CATALOG_PATH = BRB_SPECIMENS_CSV
DEF_COL = "Deformation[in]"
FORCE_COL = "Force[kip]"
# Minimum deformation range (first to last point of fit) to use a segment for b; fraction of L_y
MIN_DEF_RANGE_FRAC = 0.005
MIN_DEF_RANGE_ABS_IN = 0.001  # fallback when L_y is 0
# Discard segment if its peak def is < this fraction of the previous same-kind peak (max_def/min_def)
MIN_PEAK_FRAC = 0.75

# Apparent-b amplitude weighting (for weighted_mean outputs).
# Uses the same functional form as cycle amplitude weights:
#   w = (A/Amax)^p + eps
APPARENT_B_WEIGHT_POWER = 2.0
APPARENT_B_WEIGHT_EPS = 0.05


def _segments_zero_to_peak(points: list[dict]) -> list[tuple[int, int, str]]:
    """
    From points (with idx, type), build segments from zero_def to max_def or min_def.
    Returns list of (start_idx, end_idx, end_type) with end_type in ('max_def', 'min_def').
    Only includes segments with end_idx > start_idx.
    """
    sorted_pts = sorted(points, key=lambda p: p["idx"])
    out: list[tuple[int, int, str]] = []
    last_zero_def: int | None = None
    for pt in sorted_pts:
        idx, t = pt["idx"], pt.get("type", "")
        if "zero_def" in t:
            last_zero_def = idx
        if "max_def" in t and last_zero_def is not None and idx > last_zero_def:
            out.append((last_zero_def, idx, "max_def"))
            last_zero_def = None
        if "min_def" in t and last_zero_def is not None and idx > last_zero_def:
            out.append((last_zero_def, idx, "min_def"))
            last_zero_def = None
    return out


def _segment_peak_ok(
    peak_def: float,
    is_tension: bool,
    last_tension_peak: float | None,
    last_compression_peak: float | None,
) -> tuple[bool, float | None, float | None]:
    """
    Return whether to keep this segment given peak deformation and previous same-kind peak.
    max_def should be increasing or stay constant; min_def magnitude same.
    Returns (keep, new_last_tension_peak, new_last_compression_peak).
    """
    if is_tension:
        if last_tension_peak is not None and peak_def < MIN_PEAK_FRAC * last_tension_peak:
            return (False, last_tension_peak, last_compression_peak)
        return (True, peak_def, last_compression_peak)
    else:
        abs_peak = abs(peak_def)
        if last_compression_peak is not None and abs_peak < MIN_PEAK_FRAC * abs(last_compression_peak):
            return (False, last_tension_peak, last_compression_peak)
        return (True, last_tension_peak, peak_def)


def _b_from_segment(
    u: np.ndarray,
    F: np.ndarray,
    start: int,
    end: int,
    E_hat: float,
    A_sc: float,
    L_T: float,
    f_yc: float,
    L_y: float,
) -> float | None:
    """
    For segment [start, end] (inclusive), local tangent stiffness on **experimental** ``F``;
    first index where ``|k| <= LOW_STIFFNESS_K_FRAC * k_init`` with ``k_init = E_hat*A_sc/L_T``;
    fit F vs u from that index to end. Return b = ksh / kinit, or None if not enough points.
    """
    if end <= start + 1:
        return None
    fy_A = f_yc * A_sc
    if fy_A <= 0:
        return None
    kinit = E_hat * A_sc / L_T if L_T != 0 else 0.0
    if kinit <= 0:
        return None
    u_seg = u[start : end + 1]
    F_seg = F[start : end + 1]
    n_seg = len(u_seg)
    e_half = end + 1
    sl = local_tangent_stiffness_exp(
        u, F, start, e_half, cd_step=LANDMARK_SLOPE_CD_STEP
    )
    mask_seg = low_stiffness_mask_slopes(sl, kinit, frac=LOW_STIFFNESS_K_FRAC)
    i_plastic = None
    for i in range(len(mask_seg)):
        if mask_seg[i]:
            i_plastic = i
            break
    if i_plastic is None or i_plastic >= n_seg - 1:
        return None
    # Fit F = ksh * u + c from i_plastic to end
    u_fit = np.asarray(u_seg[i_plastic:], dtype=float)
    F_fit = np.asarray(F_seg[i_plastic:], dtype=float)
    if len(u_fit) < 2:
        return None
    min_def_range = MIN_DEF_RANGE_FRAC * L_y if L_y > 0 else MIN_DEF_RANGE_ABS_IN
    if np.abs(u_fit[-1] - u_fit[0]) < min_def_range:
        return None
    # Linear regression: slope = cov(u,F) / var(u)
    cov = np.cov(u_fit, F_fit)[0, 1]
    var_u = np.var(u_fit)
    if var_u <= 0:
        return None
    ksh = cov / var_u
    b = ksh / kinit
    # Clamp to reasonable range (strain hardening ratio typically in [0, 0.05] or so)
    b = float(np.clip(b, 0.0, 0.2))
    return b


def _segment_line_data(
    u: np.ndarray,
    F: np.ndarray,
    start: int,
    end: int,
    E_hat: float,
    A_sc: float,
    L_T: float,
    f_yc: float,
    L_y: float,
) -> tuple[float, np.ndarray, np.ndarray, bool, float] | None:
    """
    Same as _b_from_segment but also return (u_fit, F_fitted) for the fitted line
    from the first ``|F| >= 1.1 f_y A_sc`` index to peak.
    Returns (b, u_fit, F_fitted, is_tension, amp) or None.
    is_tension is True when mean force in plastic part >= 0.
    amp is max(|u|) within the fitted hardening region (u_fit).
    """
    if end <= start + 1:
        return None
    fy_A = f_yc * A_sc
    if fy_A <= 0:
        return None
    kinit = E_hat * A_sc / L_T if L_T != 0 else 0.0
    if kinit <= 0:
        return None
    u_seg = u[start : end + 1]
    F_seg = F[start : end + 1]
    n_seg = len(u_seg)
    thr = PLASTIC_STRESS_RATIO_FY * fy_A
    i_plastic = None
    for i in range(n_seg):
        fv = float(F_seg[i])
        if np.isfinite(fv) and abs(fv) >= thr:
            i_plastic = i
            break
    if i_plastic is None or i_plastic >= n_seg - 1:
        return None
    u_fit = np.asarray(u_seg[i_plastic:], dtype=float)
    F_fit = np.asarray(F_seg[i_plastic:], dtype=float)
    if len(u_fit) < 2:
        return None
    min_def_range = MIN_DEF_RANGE_FRAC * L_y if L_y > 0 else MIN_DEF_RANGE_ABS_IN
    if np.abs(u_fit[-1] - u_fit[0]) < min_def_range:
        return None
    cov = np.cov(u_fit, F_fit)[0, 1]
    var_u = np.var(u_fit)
    if var_u <= 0:
        return None
    ksh = cov / var_u
    b = ksh / kinit
    b = float(np.clip(b, 0.0, 0.2))
    # Intercept from selected b: line passes through (mean(u), mean(F)), so c = mean(F) - ksh_plot*mean(u)
    # When b=0 this gives c = mean(F) (horizontal line at mean force in segment)
    ksh_plot = b * kinit
    c = float(np.mean(F_fit) - ksh_plot * np.mean(u_fit))
    F_fitted = ksh_plot * u_fit + c
    # Tension vs compression by sign of force (or intercept): positive -> tension (b_p), negative -> compression (b_n)
    is_tension = bool(np.mean(F_fit) >= 0.0)
    amp = float(np.max(np.abs(u_fit))) if len(u_fit) else float("nan")
    return (b, u_fit, F_fitted, is_tension, amp)


def _get_b_lists(
    u: np.ndarray,
    F: np.ndarray,
    n: int,
    points: list[dict],
    E_hat: float,
    A_sc: float,
    L_T: float,
    L_y: float,
    fy: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Return (b_p_list, b_n_list, amp_p_list, amp_n_list).

    Skips segments with too small def range or peak < 75% of previous same-kind peak.
    Amplitudes are max(|u|) over the fitted hardening region of each segment.
    """
    segments = _segments_zero_to_peak(points)
    b_p_list: list[float] = []
    b_n_list: list[float] = []
    amp_p_list: list[float] = []
    amp_n_list: list[float] = []
    last_tension_peak: float | None = None
    last_compression_peak: float | None = None
    for start_idx, end_idx, _end_type in segments:
        if end_idx >= n or start_idx < 0:
            continue
        result = _segment_line_data(u, F, start_idx, end_idx, E_hat, A_sc, L_T, fy, L_y)
        if result is None:
            continue
        b, u_fit, _F_fitted, is_tension, amp = result
        peak_def = float(u_fit[-1])
        keep, last_tension_peak, last_compression_peak = _segment_peak_ok(
            peak_def, is_tension, last_tension_peak, last_compression_peak
        )
        if not keep:
            continue
        if is_tension:
            b_p_list.append(b)
            amp_p_list.append(amp)
        else:
            b_n_list.append(b)
            amp_n_list.append(amp)
    return (b_p_list, b_n_list, amp_p_list, amp_n_list)


def get_b_lists_one_specimen(specimen_id: str) -> tuple[list[float], list[float]]:
    """Load resampled F-u and cycle points; return (b_n_list, b_p_list). Empty if no resampled CSV."""
    resampled_path = resolve_resampled_force_deformation_csv(specimen_id, _PROJECT_ROOT)
    if resampled_path is None or not resampled_path.is_file():
        return ([], [])
    catalog = pd.read_csv(CATALOG_PATH)
    row = catalog[catalog["Name"].astype(str) == specimen_id]
    if row.empty:
        return ([], [])
    row = row.iloc[0]
    L_T = float(row["L_T_in"])
    L_y = float(row["L_y_in"])
    A_sc = float(row["A_c_in2"])
    A_t = float(row["A_t_in2"])
    fy = float(row["f_yc_ksi"])
    df = pd.read_csv(resampled_path)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        return ([], [])
    u = df["Deformation[in]"].values
    F = df["Force[kip]"].values
    n = len(u)
    loaded = load_cycle_points_resampled(specimen_id)
    if loaded is None:
        points, _ = find_cycle_points(df)
    else:
        points, _ = loaded
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    b_p_list, b_n_list, _amp_p, _amp_n = _get_b_lists(u, F, n, points, E_hat, A_sc, L_T, L_y, fy)
    return (b_n_list, b_p_list)


def extract_bn_bp_one_specimen(
    specimen_id: str,
    df: pd.DataFrame,
    points: list[dict],
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    fy: float,
) -> dict:
    """
    Compute Q, E_hat, and b_n / b_p (mean and median) for one specimen.
    Returns dict with catalog fields plus Q, E_hat, L_y_sqrt_A_sc, fy_over_E,
    fy_over_E_hat, and segment-wise ``b_n`` / ``b_p`` stats (mean, median, q1, q3, min, max).
    """
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    kinit = E_hat * A_sc / L_T if L_T > 0 else 0.0

    u = df["Deformation[in]"].values
    F = df["Force[kip]"].values
    n = len(u)
    if n == 0:
        return {}

    b_p_list, b_n_list, amp_p_list, amp_n_list = _get_b_lists(
        u, F, n, points, Q * E_ksi, A_sc, L_T, L_y, fy
    )

    def mean_or_nan(x: list[float]) -> float:
        """Mean of finite values or NaN."""
        return float(np.mean(x)) if x else np.nan

    def median_or_nan(x: list[float]) -> float:
        """Median of finite values or NaN."""
        return float(np.median(x)) if x else np.nan

    def q1_or_nan(x: list[float]) -> float:
        """First quartile or NaN."""
        return float(np.quantile(np.asarray(x, dtype=float), 0.25)) if x else np.nan

    def q3_or_nan(x: list[float]) -> float:
        """Third quartile or NaN."""
        return float(np.quantile(np.asarray(x, dtype=float), 0.75)) if x else np.nan

    def min_or_nan(x: list[float]) -> float:
        """Min of finite values or NaN."""
        return float(np.min(x)) if x else np.nan

    def max_or_nan(x: list[float]) -> float:
        """Max of finite values or NaN."""
        return float(np.max(x)) if x else np.nan

    def weighted_mean_or_nan(x: list[float], amps: list[float]) -> float:
        """Amplitude-weighted mean (normalized by max amp) or NaN."""
        if not x or not amps or len(x) != len(amps):
            return float("nan")
        b = np.asarray(x, dtype=float)
        a = np.asarray(amps, dtype=float)
        m = np.isfinite(b) & np.isfinite(a) & (a >= 0.0)
        if not np.any(m):
            return float("nan")
        b = b[m]
        a = a[m]
        amax = float(np.max(a)) if a.size else float("nan")
        if not np.isfinite(amax) or amax <= 0.0:
            return float(np.mean(b)) if b.size else float("nan")
        w = (a / amax) ** float(APPARENT_B_WEIGHT_POWER) + float(APPARENT_B_WEIGHT_EPS)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= 0.0:
            return float("nan")
        return float(np.sum(w * b) / sw)

    L_y_sqrt_A_sc = L_y / (A_sc ** 0.5) if A_sc > 0 else np.nan
    fy_over_E = fy / E_ksi if E_ksi != 0 else np.nan
    fy_over_E_hat = fy / E_hat if E_hat != 0 else np.nan

    return {
        "Q": Q,
        "E_hat": E_hat,
        "L_y_sqrt_A_sc": L_y_sqrt_A_sc,
        "fy_over_E": fy_over_E,
        "fy_over_E_hat": fy_over_E_hat,
        "b_n_mean": mean_or_nan(b_n_list),
        "b_n_median": median_or_nan(b_n_list),
        "b_n_q1": q1_or_nan(b_n_list),
        "b_n_q3": q3_or_nan(b_n_list),
        "b_n_min": min_or_nan(b_n_list),
        "b_n_max": max_or_nan(b_n_list),
        "b_n_weighted_mean": weighted_mean_or_nan(b_n_list, amp_n_list),
        "b_p_mean": mean_or_nan(b_p_list),
        "b_p_median": median_or_nan(b_p_list),
        "b_p_q1": q1_or_nan(b_p_list),
        "b_p_q3": q3_or_nan(b_p_list),
        "b_p_min": min_or_nan(b_p_list),
        "b_p_max": max_or_nan(b_p_list),
        "b_p_weighted_mean": weighted_mean_or_nan(b_p_list, amp_p_list),
    }


def extract_bn_bp_unordered_row(sid: str, row: pd.Series) -> dict[str, float]:
    """Envelope ``b_p`` / ``b_n`` from F-u cloud; single pair -> means only, quartiles NaN."""
    fd_path = resolve_filtered_force_deformation_csv(sid, _PROJECT_ROOT)
    if fd_path is None or not fd_path.is_file():
        fd_path = force_deformation_unordered_csv_path(sid, _PROJECT_ROOT)
    if not fd_path.is_file():
        raise FileNotFoundError(str(fd_path))
    df_fd = pd.read_csv(fd_path)
    if DEF_COL not in df_fd.columns or FORCE_COL not in df_fd.columns:
        raise ValueError(f"{sid}: {fd_path} missing {DEF_COL} / {FORCE_COL}")
    u = df_fd[DEF_COL].to_numpy(dtype=float)
    F = df_fd[FORCE_COL].to_numpy(dtype=float)
    m = np.isfinite(u) & np.isfinite(F)
    u, F = u[m], F[m]
    if len(u) == 0:
        raise ValueError(f"{sid}: no finite F-u points")
    L_T = float(row["L_T_in"])
    L_y = float(row["L_y_in"])
    A_sc = float(row["A_c_in2"])
    A_t = float(row["A_t_in2"])
    fy = float(row["f_yc_ksi"])
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    diag = compute_envelope_bn_unordered(
        u, F, L_T=L_T, L_y=L_y, A_sc=A_sc, A_t=A_t, f_yc=fy, E_ksi_val=E_ksi
    )
    L_y_sqrt_A_sc = L_y / (A_sc**0.5) if A_sc > 0 else np.nan
    nan = float("nan")
    return {
        "Q": Q,
        "E_hat": E_hat,
        "L_y_sqrt_A_sc": L_y_sqrt_A_sc,
        "fy_over_E": fy / E_ksi if E_ksi != 0 else np.nan,
        "fy_over_E_hat": fy / E_hat if E_hat != 0 else np.nan,
        "b_n_mean": float(diag.b_n) if np.isfinite(diag.b_n) else nan,
        "b_n_median": nan,
        "b_n_q1": nan,
        "b_n_q3": nan,
        "b_n_min": nan,
        "b_n_max": nan,
        "b_n_weighted_mean": float(diag.b_n) if np.isfinite(diag.b_n) else nan,
        "b_p_mean": float(diag.b_p) if np.isfinite(diag.b_p) else nan,
        "b_p_median": nan,
        "b_p_q1": nan,
        "b_p_q3": nan,
        "b_p_min": nan,
        "b_p_max": nan,
        "b_p_weighted_mean": float(diag.b_p) if np.isfinite(diag.b_p) else nan,
    }


def get_specimens_with_resampled() -> list[str]:
    """Path-ordered specimen IDs that have resampled ``force_deformation.csv`` (excludes scatter)."""
    catalog = read_catalog()
    return sorted(path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT))


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Extract b_n, b_p and geometry params per specimen; save extended catalog CSV.")
    parser.add_argument("--specimen", type=str, default=None, help="Single specimen ID; default: all with resampled data")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path; default: results/calibration/specimen_apparent_bn_bp.csv",
    )
    args = parser.parse_args()

    catalog = pd.read_csv(CATALOG_PATH)
    catalog_by_name = catalog.set_index("Name")

    with_resampled = set(get_specimens_with_resampled())
    cloud_only = set(list_names_digitized_unordered(read_catalog()))
    # Catalog order; path-ordered (resampled) and/or scatter-cloud rows
    names_all: list[str] = []
    for name in catalog["Name"].astype(str):
        if name in with_resampled or name in cloud_only:
            names_all.append(name)

    if not names_all:
        print("No specimens with resampled data and/or digitized cloud inputs.")
        return

    if args.specimen:
        if args.specimen not in names_all:
            print(f"Specimen {args.specimen} not in catalog or missing resampled/cloud inputs.")
            return
        names_all = [args.specimen]

    out_path = Path(args.output) if args.output else SPECIMEN_APPARENT_BN_BP_PATH
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in names_all:
        row = catalog_by_name.loc[sid].to_dict()
        out_row = {k: v for k, v in row.items()}
        if "Name" not in out_row:
            out_row["Name"] = sid

        if sid in with_resampled:
            rp = resolve_resampled_force_deformation_csv(sid, _PROJECT_ROOT)
            if rp is None or not rp.is_file():
                print(f"Skip {sid}: missing resampled force_deformation.csv")
                continue
            df = pd.read_csv(rp)
            if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
                print(f"Skip {sid}: missing Force or Deformation columns")
                continue
            loaded = load_cycle_points_resampled(sid)
            if loaded is None:
                points, _ = find_cycle_points(df)
            else:
                points, _ = loaded
            L_T = float(row["L_T_in"])
            L_y = float(row["L_y_in"])
            A_sc = float(row["A_c_in2"])
            A_t = float(row["A_t_in2"])
            fy = float(row["f_yc_ksi"])
            extra = extract_bn_bp_one_specimen(sid, df, points, L_T, L_y, A_sc, A_t, fy)
            out_row.update(extra)
            print(
                f"  {sid} (resampled): b_n_median={extra.get('b_n_median', np.nan):.6f}, "
                f"b_n_mean={extra.get('b_n_mean', np.nan):.6f}, b_p_median={extra.get('b_p_median', np.nan):.6f}"
            )
        else:
            try:
                extra = extract_bn_bp_unordered_row(sid, catalog_by_name.loc[sid])
            except (OSError, ValueError) as e:
                print(f"Skip {sid} (scatter): {e}")
                continue
            out_row.update(extra)
            print(
                f"  {sid} (cloud): b_n_mean={extra.get('b_n_mean', np.nan):.6f}, "
                f"b_p_mean={extra.get('b_p_mean', np.nan):.6f} (median/q1/q3 NaN)"
            )

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    # Order columns: catalog first, then added (Q, E_hat, ...)
    catalog_cols = [c for c in catalog.columns if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in catalog_cols]
    out_df = out_df[catalog_cols + rest]
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
