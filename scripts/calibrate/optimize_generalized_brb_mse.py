"""
Generalized optimization of shared steel parameters across specimens (per set_id or global).

Default optimized columns come from ``params_to_optimize.PARAMS_TO_OPTIMIZE``. Optional
``config/calibration/set_id_settings.csv`` can override the list per ``set_id``. Pooled mode
(``--no-by-set-id``) requires every training ``set_id`` to resolve to the same list when that CSV is present.

Minimizes the weighted mean of per-row J_total (same landmark + energy loss as
optimize_brb_mse). Specimen weights: ``generalized_weight`` on path-ordered rows (see
``specimen_weights.make_generalized_weight_fn``); initial averaged iterate uses ``averaged_weight``.

Initial iterate: weighted mean averaged vector from the input parameters CSV.
After optimization, evaluates specimens that have resampled and/or digitized unordered data and writes
metrics CSV under ``results/calibration/generalized_optimize/``, a per-``set_id`` eval rollup CSV under
``summary_statistics/generalized_set_id_eval_summary.csv``, plots, NPZ and CSV force histories. The **parameters** CSV lists **every** row in the
input ``--params`` file with merged generalized steel columns (same shared vector per ``set_id`` when
``--no-by-set-id`` is not used). Digitized unordered (``path_ordered=false``) specimens use the pipeline
resampled deformation drive when available; combined-directory overlays (no path metrics), same as
``eval_averaged_params``.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    energy_scale_s_e,
)
from calibrate.calibration_io import metrics_dataframe  # noqa: E402
from calibrate.calibration_loss_settings import (  # noqa: E402
    CalibrationLossSettings,
    DEFAULT_CALIBRATION_LOSS_SETTINGS,
)
from calibrate.cycle_feature_loss import (  # noqa: E402
    deformation_scale_s_d,
    load_p_y_kip_catalog,
)
from calibrate.lbfgsb_reparam import (  # noqa: E402
    prepare_lbfgsb_parameterization,
    variable_params_from_optimizer_x,
)
from calibrate.optimize_brb_mse import (  # noqa: E402
    AMPLITUDE_WEIGHTS_ARG_HELP,
    DEBUG_PARTITION,
    FAILURE_PENALTY,
    OUTPUT_CSV_SIGFIGS,
    PARAMS_TO_OPTIMIZE,
    _dataframe_for_param_csv,
    _loss_weight_snapshot,
    _metrics_dict_for_breakdown,
    _metrics_dict_nan_prefix,
    force_scale_s_f,
    save_simulated_force_history,
    save_simulated_force_history_csv,
    simulated_force_history_dir,
    simulate_and_loss_breakdown,
    _row_to_sim_params,
)
from calibrate.param_limits import bounds_dict_for  # noqa: E402
from calibrate.digitized_unordered_eval_lib import (  # noqa: E402
    compute_unordered_cloud_metrics,
    eval_row_with_envelope_bn_from_unordered,
    load_digitized_unordered_series,
)
from calibrate.plot_params_vs_filtered import (  # noqa: E402
    plot_force_def_digitized_unordered_overlays,
    plot_force_def_overlays,
    plot_unordered_binned_cloud_envelopes,
)
from calibrate.averaged_params_lib import (  # noqa: E402
    compute_weighted_averaged_param_dict,
    get_averaged_for_set_id,
    merge_averaged_into_row,
)
from calibrate.calibration_paths import (  # noqa: E402
    GENERALIZED_BRB_PARAMETERS_PATH,
    GENERALIZED_PARAMS_EVAL_METRICS_PATH,
    GENERALIZED_SET_ID_EVAL_SUMMARY_CSV,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    PARAM_LIMITS_CSV,
    PLOTS_GENERALIZED_OPTIMIZE,
    SET_ID_SETTINGS_CSV,
)
from calibrate.pipeline_log import kv, line, run_banner, saved_artifacts, section  # noqa: E402
from calibrate.report_generalized_set_id_eval_summary import (  # noqa: E402
    write_generalized_set_id_eval_summary,
    write_generalized_unordered_j_split_summaries,
)
from calibrate.set_id_optimize_params import (  # noqa: E402
    assert_global_loss_settings_consistent,
    build_param_cols_by_set_id_from_mapping,
    resolve_loss_settings_for_set_id,
    resolve_optimize_params_for_set_id,
    union_param_cols,
    unique_weighted_train_set_ids,
)
from calibrate.set_id_settings import load_set_id_optimize_and_loss  # noqa: E402
from calibrate.specimen_weights import (  # noqa: E402
    catalog_metrics_fields,
    make_generalized_weight_fn,
    make_averaged_weight_fn,
    weight_config_tag,
)
from model.corotruss import run_simulation  # noqa: E402
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    COLOR_NUMERICAL_COHORT,
    COLOR_NUMERICAL_COHORT_AUX,
)
from postprocess.plot_specimens import compute_raw_filtered_global_norm_limits  # noqa: E402
from specimen_catalog import (  # noqa: E402
    list_names_digitized_unordered,
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_resampled_force_deformation_csv,
)
DEFAULT_PARAMS_PATH = OPTIMIZED_BRB_PARAMETERS_PATH
DEFAULT_METRICS_OUT = GENERALIZED_PARAMS_EVAL_METRICS_PATH
DEFAULT_PARAMS_OUT = GENERALIZED_BRB_PARAMETERS_PATH
DEFAULT_PLOTS_DIR = PLOTS_GENERALIZED_OPTIMIZE / "overlays"
DEFAULT_CLOUD_PLOTS_DIR = DEFAULT_PLOTS_DIR


@dataclass
class GeneralizedInstance:
    name: str
    set_id: Any
    prow: pd.Series
    D_exp: np.ndarray
    F_exp: np.ndarray
    amp_meta: list[dict]
    loss: CalibrationLossSettings
    p_y_ref: float
    s_d: float
    weight: float


def _collect_instances(
    params_df: pd.DataFrame,
    available: list[str],
    weight_fn: Any,
    *,
    loss_by_set_id: dict[int, CalibrationLossSettings],
    amplitude_weights_override: bool | None,
) -> list[GeneralizedInstance]:
    """Build GeneralizedInstance list from resampled force CSVs and params."""
    out: list[GeneralizedInstance] = []
    for sid in available:
        csv_path = resolve_resampled_force_deformation_csv(sid, _PROJECT_ROOT)
        if csv_path is None or not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
            continue
        D_exp = df["Deformation[in]"].to_numpy(dtype=float)
        F_exp = df["Force[kip]"].to_numpy(dtype=float)
        loaded = load_cycle_points_resampled(sid)
        points, _ = loaded if loaded is not None else find_cycle_points(df)
        s_d_ref = deformation_scale_s_d(D_exp)
        rows_for = params_df[params_df["Name"].astype(str) == sid]
        if rows_for.empty:
            continue
        p_y_catalog = load_p_y_kip_catalog(
            _PROJECT_ROOT,
            sid,
            float(rows_for.iloc[0]["fyp"]),
            float(rows_for.iloc[0]["A_sc"]),
        )
        w = float(weight_fn(sid))
        for _, prow in rows_for.iterrows():
            loss = resolve_loss_settings_for_set_id(loss_by_set_id, prow.get("set_id"))
            use_amp_w = (
                bool(amplitude_weights_override)
                if amplitude_weights_override is not None
                else loss.use_amplitude_weights
            )
            _mse_w, amp_meta = build_amplitude_weights(
                D_exp,
                points,
                p=loss.amplitude_weight_power,
                eps=loss.amplitude_weight_eps,
                debug_partition=DEBUG_PARTITION,
                use_amplitude_weights=use_amp_w,
            )
            out.append(
                GeneralizedInstance(
                    name=sid,
                    set_id=prow.get("set_id", "?"),
                    prow=prow,
                    D_exp=D_exp,
                    F_exp=F_exp,
                    amp_meta=amp_meta,
                    loss=loss,
                    p_y_ref=p_y_catalog,
                    s_d=s_d_ref,
                    weight=w,
                )
            )
    return out


def _generalized_objective_value(
    x: np.ndarray,
    train: list[GeneralizedInstance],
    params_to_optimize: list[str],
    use_norm: list[bool],
    Ls: list[float],
    Us: list[float],
    *,
    exp_landmark_cache: dict | None = None,
) -> float:
    """Weighted mean J over training instances at optimizer x."""
    variable = variable_params_from_optimizer_x(x, params_to_optimize, use_norm, Ls, Us)
    num = 0.0
    den = 0.0
    for inst in train:
        if inst.weight <= 0.0:
            continue
        fixed = _row_to_sim_params(inst.prow)
        params = {**fixed, **variable}
        try:
            F_sim = np.asarray(run_simulation(inst.D_exp, **params), dtype=float)
        except Exception:
            return float(FAILURE_PENALTY)
        prow = inst.prow
        bd = simulate_and_loss_breakdown(
            inst.D_exp,
            inst.F_exp,
            F_sim,
            inst.amp_meta,
            s_d=inst.s_d,
            loss=inst.loss,
            fy_ksi=float(prow["fyp"]),
            a_sc=float(prow["A_sc"]),
            L_T=float(prow["L_T"]),
            L_y=float(prow["L_y"]),
            A_t=float(prow["A_t"]),
            E_ksi=float(prow["E"]),
            exp_landmark_cache=exp_landmark_cache,
            full_metrics=False,
        )
        if bd is None:
            return float(FAILURE_PENALTY)
        num += inst.weight * bd.j_total
        den += inst.weight
    if den <= 0.0:
        return float(FAILURE_PENALTY)
    return num / den


def _optimize_one_generalized_group(
    train: list[GeneralizedInstance],
    params_df: pd.DataFrame,
    init_averaged: pd.Series,
    params_to_optimize: list[str],
    bounds_dict: dict[str, tuple[float, float]],
) -> tuple[pd.Series, Any]:
    """Return optimized generalized Series and scipy OptimizeResult."""
    rep = train[0].prow.copy()
    merged = merge_averaged_into_row(rep, init_averaged, params_to_optimize)
    use_norm, Ls, Us, x0, scipy_bounds = prepare_lbfgsb_parameterization(
        params_to_optimize,
        bounds_dict,
        merged,
        specimen_hint="generalized",
    )
    exp_landmark_cache: dict = {}

    def fun(x: np.ndarray) -> float:
        """Scalar objective for L-BFGS-B."""
        return _generalized_objective_value(
            x,
            train,
            params_to_optimize,
            use_norm,
            Ls,
            Us,
            exp_landmark_cache=exp_landmark_cache,
        )

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
    out = pd.Series({k: float(physical[k]) for k in params_to_optimize})
    return out, res


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=(
            "Generalized L-BFGS-B on shared steel parameters (default PARAMS_TO_OPTIMIZE; optional "
            "per-set_id CSV): weighted mean J_total over specimens with positive weight; then full "
            "specimen-set evaluation."
        ),
    )
    p.add_argument(
        "--params",
        type=str,
        default=str(DEFAULT_PARAMS_PATH),
        help="Input parameters CSV (e.g. optimized_brb_parameters.csv).",
    )
    p.add_argument(
        "--output-params",
        type=str,
        default=str(DEFAULT_PARAMS_OUT),
        help="Output parameters CSV (merged generalized steel params per row).",
    )
    p.add_argument(
        "--no-by-set-id",
        action="store_true",
        help="One shared vector for all set_ids (pool and optimize jointly across all).",
    )
    p.add_argument(
        "--output-metrics",
        type=str,
        default=str(DEFAULT_METRICS_OUT),
        help="Output metrics CSV path.",
    )
    p.add_argument(
        "--output-plots-dir",
        type=str,
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory for resampled generalized-eval hysteresis overlays.",
    )
    p.add_argument(
        "--output-cloud-plots-dir",
        type=str,
        default=str(DEFAULT_CLOUD_PLOTS_DIR),
        help="Digitized unordered overlays (default: same as --output-plots-dir).",
    )
    p.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="If set, only this Name is optimized and evaluated.",
    )
    _pl_rel = PARAM_LIMITS_CSV
    try:
        _pl_rel = PARAM_LIMITS_CSV.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    p.add_argument(
        "--param-limits",
        type=Path,
        default=None,
        help=(
            "Parameter box limits CSV (parameter, lower, upper). "
            f"Default: {_pl_rel}."
        ),
    )
    p.add_argument(
        "--amplitude-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=AMPLITUDE_WEIGHTS_ARG_HELP,
    )
    _so_rel = SET_ID_SETTINGS_CSV
    try:
        _so_rel = SET_ID_SETTINGS_CSV.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    p.add_argument(
        "--set-id-settings",
        type=Path,
        default=None,
        help=(
            "Per-set_id settings CSV (steel seeds + optimize_params + loss weights). "
            f"Default: {_so_rel}."
        ),
    )
    args = p.parse_args()

    run_banner("optimize_generalized_brb_mse.py")

    by_set_id = not args.no_by_set_id
    params_path = Path(args.params).expanduser().resolve()
    out_metrics = Path(args.output_metrics).expanduser().resolve()
    out_params = Path(args.output_params).expanduser().resolve()
    plots_dir_ordered = Path(args.output_plots_dir).expanduser().resolve()
    plots_dir_unordered = Path(args.output_cloud_plots_dir).expanduser().resolve()
    plots_dir_ordered.mkdir(parents=True, exist_ok=True)
    plots_dir_unordered.mkdir(parents=True, exist_ok=True)
    sim_hist_dir = simulated_force_history_dir(out_metrics)

    params_df = pd.read_csv(params_path)
    if "Name" not in params_df.columns:
        raise SystemExit(f"No Name column in {params_path}")

    catalog = read_catalog()
    catalog_by_name = catalog.set_index("Name")
    norm_xy_half = compute_raw_filtered_global_norm_limits(catalog, project_root=_PROJECT_ROOT)
    pooled_w_fn = make_averaged_weight_fn(catalog)
    generalized_w_fn = make_generalized_weight_fn(catalog)
    weight_tag = weight_config_tag(catalog)

    resampled_stems = path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT)
    param_names = set(params_df["Name"].astype(str))
    available_resampled = sorted(param_names & resampled_stems)
    unordered_eligible = set(list_names_digitized_unordered(catalog))
    available_unordered = sorted((param_names & unordered_eligible) - set(available_resampled))

    if not available_resampled:
        raise SystemExit(
            "Generalized optimization needs at least one path-ordered resampled specimen "
            "(data/resampled/{Name}/force_deformation.csv)."
        )

    if args.specimen:
        if args.specimen not in available_resampled and args.specimen not in available_unordered:
            raise SystemExit(
                f"Specimen {args.specimen!r} not in resampled or digitized unordered eval set."
            )
        available_resampled = [args.specimen] if args.specimen in available_resampled else []
        available_unordered = [args.specimen] if args.specimen in available_unordered else []
        if not available_resampled:
            raise SystemExit("Generalized training requires resampled data for --specimen.")

    default_list = list(PARAMS_TO_OPTIMIZE)
    opt_csv = (
        Path(args.set_id_settings).expanduser().resolve()
        if args.set_id_settings
        else SET_ID_SETTINGS_CSV
    )
    opt_map, loss_map = load_set_id_optimize_and_loss(opt_csv)
    if opt_map or loss_map:
        line(f"set_id settings CSV: {opt_csv}")
    else:
        line(
            "set_id settings: (none) — using defaults: PARAMS_TO_OPTIMIZE and "
            "DEFAULT_CALIBRATION_LOSS_SETTINGS"
        )

    instances = _collect_instances(
        params_df,
        available_resampled,
        generalized_w_fn,
        loss_by_set_id=loss_map,
        amplitude_weights_override=args.amplitude_weights,
    )
    if not instances:
        raise SystemExit("No generalized instances collected (check resampled data and params).")

    train_all = [i for i in instances if i.weight > 0.0]
    if not train_all:
        raise SystemExit("No training instances with positive specimen weight.")

    pool_set_ids = unique_weighted_train_set_ids(params_df, pooled_w_fn)
    bounds_cache: dict[tuple[str, ...], dict[str, tuple[float, float]]] = {}

    def _bounds_for_active(active: list[str]) -> dict[str, tuple[float, float]]:
        key = tuple(active)
        if key not in bounds_cache:
            bounds_cache[key] = bounds_dict_for(list(active), limits_path=args.param_limits)
        return bounds_cache[key]

    lim_path = Path(args.param_limits).expanduser().resolve() if args.param_limits else PARAM_LIMITS_CSV
    line(f"parameter limits: {lim_path}")

    global_loss = (
        assert_global_loss_settings_consistent(
            loss_map, [inst.set_id for inst in train_all], DEFAULT_CALIBRATION_LOSS_SETTINGS
        )
        if (not by_set_id and loss_map)
        else DEFAULT_CALIBRATION_LOSS_SETTINGS
    )
    global_active = (
        assert_global_optimize_params_consistent(
            opt_map, [inst.set_id for inst in train_all], default_list
        )
        if (not by_set_id and opt_map)
        else default_list
    )

    if by_set_id:
        param_cols_by_set_id = (
            build_param_cols_by_set_id_from_mapping(opt_map, default_list, set_ids=pool_set_ids)
            if opt_map
            else None
        )
        pool_param_cols = union_param_cols(default_list, param_cols_by_set_id)
        init_pool = compute_weighted_averaged_param_dict(
            params_df,
            pool_param_cols,
            by_set_id=True,
            weight_fn=pooled_w_fn,
            param_cols_by_set_id=param_cols_by_set_id,
        )
    else:
        init_pool = compute_weighted_averaged_param_dict(
            params_df,
            global_active,
            by_set_id=False,
            weight_fn=pooled_w_fn,
        )

    generalized_pool: dict[Any, pd.Series] = {}

    section("Joint optimization (L-BFGS-B)")
    if by_set_id:
        by_sid: dict[int, list[GeneralizedInstance]] = {}
        for inst in train_all:
            sid_k = int(pd.to_numeric(inst.set_id, errors="raise"))
            by_sid.setdefault(sid_k, []).append(inst)
        for sid_key, tr in by_sid.items():
            active = resolve_optimize_params_for_set_id(opt_map, sid_key, default_list)
            init_vec = get_averaged_for_set_id(init_pool, sid_key, by_set_id=True)
            line(
                f"set_id={sid_key}  ({len(tr)} training rows, init averaged)  optimize: {active}..."
            )
            opt_vec, res = _optimize_one_generalized_group(
                tr,
                params_df,
                init_vec,
                active,
                _bounds_for_active(active),
            )
            generalized_pool[sid_key] = opt_vec
            line(
                f"L-BFGS-B  success={res.success}  fun={res.fun:.6g}  message={res.message!r}"
            )
    else:
        init_vec = get_averaged_for_set_id(init_pool, "_global", by_set_id=False)
        line(
            f"global pool  ({len(train_all)} training rows, init averaged)  "
            f"optimize: {global_active}..."
        )
        opt_vec, res = _optimize_one_generalized_group(
            train_all,
            params_df,
            init_vec,
            global_active,
            _bounds_for_active(global_active),
        )
        generalized_pool["_global"] = opt_vec
        line(
            f"L-BFGS-B  success={res.success}  fun={res.fun:.6g}  message={res.message!r}"
        )

    kv("output parameters", str(out_params))
    kv("output metrics", str(out_metrics))
    kv("plots (path-ordered)", str(plots_dir_ordered))
    if plots_dir_unordered.resolve() != plots_dir_ordered.resolve():
        kv("plots (digitized unordered)", str(plots_dir_unordered))
    kv("weights", repr(weight_tag))
    kv("loss settings", f"per-set via {SET_ID_SETTINGS_CSV.name}")
    if args.amplitude_weights is None:
        kv("J_feat cycle weights", "per-set (see CSV)")
    else:
        kv(
            "J_feat cycle weights",
            "amplitude (CLI override)" if bool(args.amplitude_weights) else "uniform (CLI override)",
        )

    rows_out: list[dict[str, Any]] = []

    section("Generalized evaluation -- path-ordered (resampled)")
    for inst in instances:
        sid = inst.name
        set_id = inst.set_id
        D_exp = inst.D_exp
        F_exp = inst.F_exp
        amp_meta = inst.amp_meta
        prow = inst.prow
        specimen_w = inst.weight
        contributes = specimen_w > 0.0
        cm = catalog_metrics_fields(sid, catalog_by_name)

        try:
            init_shared = get_averaged_for_set_id(init_pool, set_id, by_set_id=by_set_id)
        except KeyError as e:
            line(f"skip {sid} set {set_id}: missing initial seed {e}")
            continue

        try:
            shared = get_averaged_for_set_id(generalized_pool, set_id, by_set_id=by_set_id)
        except KeyError as e:
            line(f"skip {sid} set {set_id}: {e}")
            continue

        active = resolve_optimize_params_for_set_id(opt_map, set_id, default_list)
        eval_row_init = merge_averaged_into_row(prow, init_shared, active)
        eval_row = merge_averaged_into_row(prow, shared, active)
        s_f_ref = force_scale_s_f(F_exp)
        s_e_ref = energy_scale_s_e(D_exp, F_exp)
        n_cycles = len(amp_meta)

        exp_landmark_cache_eval: dict = {}
        try:
            F_sim_init = np.asarray(
                run_simulation(D_exp, **_row_to_sim_params(eval_row_init)),
                dtype=float,
            )
        except Exception as exc:
            line(f"{sid} set {set_id}: initial simulation failed: {exc}")
            continue

        if F_sim_init.shape != F_exp.shape:
            line(f"{sid} set {set_id}: length mismatch initial sim vs exp")
            continue

        loss_here = inst.loss if by_set_id else global_loss

        bd_init = simulate_and_loss_breakdown(
            D_exp,
            F_exp,
            F_sim_init,
            amp_meta,
            s_d=inst.s_d,
            loss=loss_here,
            fy_ksi=float(eval_row_init["fyp"]),
            a_sc=float(eval_row_init["A_sc"]),
            L_T=float(eval_row_init["L_T"]),
            L_y=float(eval_row_init["L_y"]),
            A_t=float(eval_row_init["A_t"]),
            E_ksi=float(eval_row_init["E"]),
            exp_landmark_cache=exp_landmark_cache_eval,
        )
        if bd_init is None:
            line(f"{sid} set {set_id}: initial loss breakdown failed")
            continue

        try:
            F_sim = np.asarray(
                run_simulation(D_exp, **_row_to_sim_params(eval_row)),
                dtype=float,
            )
        except Exception as exc:
            line(f"{sid} set {set_id}: simulation failed: {exc}")
            continue

        if F_sim.shape != F_exp.shape:
            line(f"{sid} set {set_id}: length mismatch sim vs exp")
            continue

        bd = simulate_and_loss_breakdown(
            D_exp,
            F_exp,
            F_sim,
            amp_meta,
            s_d=inst.s_d,
            loss=loss_here,
            fy_ksi=float(eval_row["fyp"]),
            a_sc=float(eval_row["A_sc"]),
            L_T=float(eval_row["L_T"]),
            L_y=float(eval_row["L_y"]),
            A_t=float(eval_row["A_t"]),
            E_ksi=float(eval_row["E"]),
            exp_landmark_cache=exp_landmark_cache_eval,
        )
        if bd is None:
            line(f"{sid} set {set_id}: loss breakdown failed")
            continue

        jtot0 = bd_init.j_total
        jtot = bd.j_total
        cloud_init = compute_unordered_cloud_metrics(D_exp, F_exp, D_exp, F_sim_init)
        cloud_final = compute_unordered_cloud_metrics(D_exp, F_exp, D_exp, F_sim)
        mi = _metrics_dict_for_breakdown(bd_init, loss_here, "initial")
        mf = _metrics_dict_for_breakdown(bd, loss_here, "final")

        try:
            plot_force_def_overlays(
                sid,
                D_exp,
                F_exp,
                F_sim,
                fy_ksi=float(eval_row["fyp"]),
                A_c_in2=float(eval_row["A_sc"]),
                L_y_in=float(eval_row["L_y"]),
                set_id=set_id,
                out_dir=plots_dir_ordered,
                norm_xy_half=norm_xy_half,
                numerical_color=(
                    COLOR_NUMERICAL_COHORT if contributes else COLOR_NUMERICAL_COHORT_AUX
                ),
                show_numerical_curve=True,
            )
        except Exception as exc:
            line(f"{sid} set {set_id}: plotting failed: {exc}")

        try:
            fyA = float(eval_row["fyp"]) * float(eval_row["A_sc"])
            ly = float(eval_row["L_y"])
            exp_plot_xy = np.column_stack(
                [cloud_final.exp_points_raw[:, 0] / ly, cloud_final.exp_points_raw[:, 1] / fyA]
            )
            num_plot_xy = np.column_stack(
                [cloud_final.num_points_raw[:, 0] / ly, cloud_final.num_points_raw[:, 1] / fyA]
            )
        except Exception as exc:
            line(f"{sid} set {set_id}: cloud plot arrays failed: {exc}")
        else:
            try:
                plot_unordered_binned_cloud_envelopes(
                    sid,
                    set_id,
                    exp_plot_xy,
                    num_plot_xy,
                    out_dir=plots_dir_ordered,
                )
            except Exception as exc:
                line(f"{sid} set {set_id}: binned cloud envelope plotting failed: {exc}")

        saved_npz = save_simulated_force_history(
            sim_hist_dir, sid, set_id, D_exp, F_exp, F_sim
        )
        saved_csv = save_simulated_force_history_csv(
            sim_hist_dir, sid, set_id, D_exp, F_exp, F_sim
        )
        saved_artifacts(
            saved_npz.name if saved_npz is not None else None,
            saved_csv.name if saved_csv is not None else None,
        )

        rows_out.append(
            {
                "Name": sid,
                "set_id": set_id,
                "specimen_weight": specimen_w,
                "contributes_to_aggregate": contributes,
                **cm,
                "weight_config": weight_tag,
                "calibration_stage": "generalized_opt",
                "aggregate_by_set_id": by_set_id,
                **mi,
                **mf,
                **_loss_weight_snapshot(loss_here),
                "S_F": s_f_ref,
                "S_D": inst.s_d,
                "S_E": s_e_ref,
                "P_y_ref": inst.p_y_ref,
                "n_cycles": n_cycles,
                "success": jtot < FAILURE_PENALTY * 0.5,
            }
        )
        line(
            f"{sid} set {set_id}: J={jtot:.6g}  J_binenv={cloud_final.J_binenv:.6g}"
        )

    section("Generalized evaluation -- digitized unordered")
    for sid in available_unordered:
        if sid not in catalog_by_name.index:
            line(f"skip {sid}: not in BRB-Specimens.csv")
            continue
        cat_row = catalog_by_name.loc[sid]
        prow_block = params_df[params_df["Name"].astype(str) == sid]
        if prow_block.empty:
            line(f"skip {sid}: no parameter rows")
            continue
        series = load_digitized_unordered_series(
            sid,
            _PROJECT_ROOT,
            steel_row=prow_block.iloc[0],
            catalog_row=cat_row,
        )
        if series is None:
            line(f"skip {sid}: digitized unordered CSVs missing or invalid")
            continue
        D_drive, u_c, F_c = series
        p_y_catalog = load_p_y_kip_catalog(
            _PROJECT_ROOT,
            sid,
            float(prow_block.iloc[0]["fyp"]),
            float(prow_block.iloc[0]["A_sc"]),
        )
        specimen_w = generalized_w_fn(sid)
        cm = catalog_metrics_fields(sid, catalog_by_name)
        s_d_ref = deformation_scale_s_d(D_drive)
        s_f_cloud = force_scale_s_f(F_c)
        s_e_cloud = energy_scale_s_e(u_c, F_c)

        for _, prow in prow_block.iterrows():
            set_id = prow.get("set_id", "?")
            contributes = specimen_w > 0.0
            active = resolve_optimize_params_for_set_id(opt_map, set_id, default_list)
            try:
                shared = get_averaged_for_set_id(generalized_pool, set_id, by_set_id=by_set_id)
            except KeyError as e:
                line(f"skip {sid} set {set_id}: {e}")
                continue

            try:
                init_shared = get_averaged_for_set_id(init_pool, set_id, by_set_id=by_set_id)
            except KeyError as e:
                line(f"skip {sid} set {set_id}: missing initial seed {e}")
                continue
            eval_row = merge_averaged_into_row(prow, shared, active)
            sim_row = eval_row_with_envelope_bn_from_unordered(eval_row, cat_row, u_c, F_c)
            eval_row_init = merge_averaged_into_row(prow, init_shared, active)
            sim_row_init = eval_row_with_envelope_bn_from_unordered(eval_row_init, cat_row, u_c, F_c)

            try:
                F_sim_init = np.asarray(
                    run_simulation(D_drive, **_row_to_sim_params(sim_row_init)),
                    dtype=float,
                )
            except Exception as exc:
                line(f"{sid} set {set_id}: initial simulation failed: {exc}")
                continue

            try:
                F_sim = np.asarray(
                    run_simulation(D_drive, **_row_to_sim_params(sim_row)),
                    dtype=float,
                )
            except Exception as exc:
                line(f"{sid} set {set_id}: simulation failed: {exc}")
                continue

            if F_sim.shape != D_drive.shape:
                line(f"{sid} set {set_id}: length mismatch sim vs deformation_history")
                continue

            if F_sim_init.shape != D_drive.shape:
                line(f"{sid} set {set_id}: length mismatch initial sim vs deformation_history")
                continue

            cloud_init = compute_unordered_cloud_metrics(u_c, F_c, D_drive, F_sim_init)
            cloud_final = compute_unordered_cloud_metrics(u_c, F_c, D_drive, F_sim)
            F_exp_na = np.full_like(D_drive, np.nan, dtype=float)

            try:
                plot_force_def_digitized_unordered_overlays(
                    sid,
                    D_drive,
                    F_sim,
                    u_c,
                    F_c,
                    fy_ksi=float(sim_row["fyp"]),
                    A_c_in2=float(sim_row["A_sc"]),
                    L_y_in=float(sim_row["L_y"]),
                    set_id=set_id,
                    out_dir=plots_dir_unordered,
                    norm_xy_half=norm_xy_half,
                    numerical_color=COLOR_NUMERICAL_COHORT_AUX,
                )
            except Exception as exc:
                line(f"{sid} set {set_id}: plotting failed: {exc}")

            try:
                fyA = float(sim_row["fyp"]) * float(sim_row["A_sc"])
                ly = float(sim_row["L_y"])
                exp_plot_xy = np.column_stack(
                    [cloud_final.exp_points_raw[:, 0] / ly, cloud_final.exp_points_raw[:, 1] / fyA]
                )
                num_plot_xy = np.column_stack(
                    [cloud_final.num_points_raw[:, 0] / ly, cloud_final.num_points_raw[:, 1] / fyA]
                )
            except Exception as exc:
                line(f"{sid} set {set_id}: cloud plot arrays failed: {exc}")
            else:
                try:
                    plot_unordered_binned_cloud_envelopes(
                        sid,
                        set_id,
                        exp_plot_xy,
                        num_plot_xy,
                        out_dir=plots_dir_unordered,
                    )
                except Exception as exc:
                    line(f"{sid} set {set_id}: binned cloud envelope plotting failed: {exc}")

            saved_npz = save_simulated_force_history(
                sim_hist_dir, sid, set_id, D_drive, F_exp_na, F_sim
            )
            saved_csv = save_simulated_force_history_csv(
                sim_hist_dir, sid, set_id, D_drive, F_exp_na, F_sim
            )
            saved_artifacts(
                saved_npz.name if saved_npz is not None else None,
                saved_csv.name if saved_csv is not None else None,
            )

            u_row: dict[str, object] = {
                "Name": sid,
                "set_id": set_id,
                "specimen_weight": specimen_w,
                "contributes_to_aggregate": contributes,
                **cm,
                "weight_config": weight_tag,
                "calibration_stage": "generalized_opt",
                "aggregate_by_set_id": by_set_id,
                **_metrics_dict_nan_prefix("initial"),
                **_metrics_dict_nan_prefix("final"),
                "initial_unordered_J_binenv": cloud_init.J_binenv,
                "final_unordered_J_binenv": cloud_final.J_binenv,
                "initial_unordered_J_binenv_l1": cloud_init.J_binenv_l1,
                "final_unordered_J_binenv_l1": cloud_final.J_binenv_l1,
                **_loss_weight_snapshot(loss_here),
                "S_F": s_f_cloud,
                "S_D": s_d_ref,
                "S_E": s_e_cloud,
                "P_y_ref": p_y_catalog,
                "n_cycles": 0,
                "success": bool(np.isfinite(cloud_final.J_binenv)),
            }
            rows_out.append(u_row)
            line(
                f"{sid} set {set_id}: digitized unordered overlay + sim + "
                f"J_binenv(init={cloud_init.J_binenv:.6g}, final={cloud_final.J_binenv:.6g})"
            )

    specimen_set_param_rows: list[dict[str, Any]] = []
    for _, prow in params_df.iterrows():
        set_id = prow.get("set_id", "?")
        shared = get_averaged_for_set_id(generalized_pool, set_id, by_set_id=by_set_id)
        active_out = resolve_optimize_params_for_set_id(opt_map, set_id, default_list)
        specimen_set_param_rows.append(
            merge_averaged_into_row(prow, shared, active_out).to_dict()
        )
    p_df = pd.DataFrame(specimen_set_param_rows)
    p_df = p_df[params_df.columns.tolist()]
    p_df = _dataframe_for_param_csv(p_df, sigfigs=OUTPUT_CSV_SIGFIGS)
    out_params.parent.mkdir(parents=True, exist_ok=True)
    p_df.to_csv(out_params, index=False)
    section("Outputs")
    kv("wrote parameters", f"{out_params}  ({len(p_df)} rows, specimen set × merged generalized steel)")

    if not rows_out:
        line("no metrics rows (no successful evaluations).")
        return

    out_df = metrics_dataframe(rows_out)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_metrics, index=False)
    kv("wrote metrics", f"{out_metrics}  ({len(out_df)} rows)")
    try:
        summary_path = write_generalized_set_id_eval_summary(
            out_df, GENERALIZED_SET_ID_EVAL_SUMMARY_CSV
        )
        kv("wrote set_id eval summary", str(summary_path))
        j_train, j_val = write_generalized_unordered_j_split_summaries(out_df)
        kv("wrote unordered J summary (train)", str(j_train))
        kv("wrote unordered J summary (validation)", str(j_val))
    except Exception as exc:
        line(f"set_id / unordered-J summaries skipped: {exc}")


if __name__ == "__main__":
    main()
