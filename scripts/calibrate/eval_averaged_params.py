"""
Weighted mean of steel parameters from a parameters CSV (default columns from
``params_to_optimize.PARAMS_TO_OPTIMIZE``), apply to every row, simulate, write parameters CSV +
metrics CSV + hysteresis plots + NPZ and CSV force histories.

Optional ``config/calibration/set_id_settings.csv`` overrides which columns are averaged and
merged per ``set_id``. Pooled mode (``--no-by-set-id``) requires identical resolved lists across
training ``set_id``s when that file is present.

Weights: ``make_averaged_weight_fn()`` from ``calibrate.specimen_weights`` -- ``averaged_weight`` on
path-ordered rows (unordered digitized rows have effective weight 0 for averaging).

Metrics match optimize_brb_mse column order via ``calibration_io.metrics_dataframe``.
Simulated histories: ``{output_metrics_stem}_simulated_force/{Name}_set{set_id}_force_history.npz``
and ``{Name}_set{set_id}_simulated.csv`` next to ``--output-metrics``.

Digitized unordered specimens (``path_ordered=false``) are evaluated with the **pipeline resampled**
``data/resampled/{Name}/deformation_history.csv`` drive when present (else raw+RDP fallback), envelope
``b_p``/``b_n`` from the unordered F–u samples, hysteresis overlays under ``--output-plots-dir`` (same
directory as resampled overlays by default), and the same
simulated CSV/NPZ (``Force[kip]`` column NaN where no path-ordered experiment exists). Metrics rows
omit path-based losses (NaN). The output parameters CSV lists **every** row in ``--params`` with merged averaged steel parameters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from calibrate.calibration_loss_settings import DEFAULT_CALIBRATION_LOSS_SETTINGS  # noqa: E402
from calibrate.cycle_feature_loss import (  # noqa: E402
    deformation_scale_s_d,
    load_p_y_kip_catalog,
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
    AVERAGED_BRB_PARAMETERS_PATH,
    AVERAGED_PARAMS_EVAL_METRICS_PATH,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    PLOTS_AVERAGED_OPTIMIZE,
    SET_ID_SETTINGS_CSV,
)
from calibrate.pipeline_log import kv, line, run_banner, saved_artifacts, section  # noqa: E402
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
DEFAULT_METRICS_OUT = AVERAGED_PARAMS_EVAL_METRICS_PATH
DEFAULT_PARAMS_OUT = AVERAGED_BRB_PARAMETERS_PATH
DEFAULT_PLOTS_DIR = PLOTS_AVERAGED_OPTIMIZE / "overlays"
# Path-ordered and digitized scatter overlays share one directory (no modality split).
DEFAULT_CLOUD_PLOTS_DIR = DEFAULT_PLOTS_DIR


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=(
            "Weighted mean of steel parameters (default PARAMS_TO_OPTIMIZE); evaluate on resampled "
            "specimens and digitized unordered specimens (deformation history + overlays); "
            "write parameters CSV, metrics CSV, plots, NPZ and CSV simulated histories."
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
        help="Output parameters CSV (merged averaged steel params per row).",
    )
    p.add_argument(
        "--no-by-set-id",
        action="store_true",
        help="Pool all input rows together (one vector for every set_id).",
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
        help="Directory for resampled (path-ordered) averaged-eval hysteresis PNGs.",
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
        help="If set, only evaluate this Name (resampled and/or digitized unordered + param rows).",
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

    run_banner("eval_averaged_params.py")

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
    weight_fn = make_averaged_weight_fn(catalog)
    weight_tag = weight_config_tag(catalog)

    default_list = list(PARAMS_TO_OPTIMIZE)
    opt_csv = (
        Path(args.set_id_settings).expanduser().resolve()
        if args.set_id_settings
        else SET_ID_SETTINGS_CSV
    )
    opt_map, loss_map = load_set_id_optimize_and_loss(opt_csv)
    pool_set_ids = unique_weighted_train_set_ids(params_df, weight_fn)
    if by_set_id:
        param_cols_by_set_id = (
            build_param_cols_by_set_id_from_mapping(opt_map, default_list, set_ids=pool_set_ids)
            if opt_map
            else None
        )
        pool_param_cols = union_param_cols(default_list, param_cols_by_set_id)
        averaged_dict = compute_weighted_averaged_param_dict(
            params_df,
            pool_param_cols,
            by_set_id=True,
            weight_fn=weight_fn,
            param_cols_by_set_id=param_cols_by_set_id,
        )
    else:
        w_series = params_df["Name"].astype(str).map(weight_fn)
        train_set_ids = params_df.loc[w_series > 0.0, "set_id"].tolist()
        global_active = (
            assert_global_optimize_params_consistent(opt_map, train_set_ids, default_list)
            if opt_map
            else default_list
        )
        global_loss = (
            assert_global_loss_settings_consistent(
                loss_map, train_set_ids, DEFAULT_CALIBRATION_LOSS_SETTINGS
            )
            if loss_map
            else DEFAULT_CALIBRATION_LOSS_SETTINGS
        )
        averaged_dict = compute_weighted_averaged_param_dict(
            params_df,
            global_active,
            by_set_id=False,
            weight_fn=weight_fn,
        )

    resampled_stems = path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT)
    param_names = set(params_df["Name"].astype(str))
    available_resampled = sorted(param_names & resampled_stems)
    unordered_eligible = set(list_names_digitized_unordered(catalog))
    available_unordered = sorted((param_names & unordered_eligible) - set(available_resampled))

    if not available_resampled and not available_unordered:
        raise SystemExit(
            "No specimens to evaluate: need resampled data and/or digitized unordered CSVs + param rows."
        )

    if args.specimen:
        if args.specimen not in available_resampled and args.specimen not in available_unordered:
            raise SystemExit(
                f"Specimen {args.specimen!r} not in resampled or digitized unordered eval set."
            )
        available_resampled = [args.specimen] if args.specimen in available_resampled else []
        available_unordered = [args.specimen] if args.specimen in available_unordered else []

    kv("PARAMS_TO_OPTIMIZE (default)", str(PARAMS_TO_OPTIMIZE))
    if opt_map:
        kv("set_id optimize params CSV", str(opt_csv))
    else:
        kv("set_id optimize params", f"(none); optional {SET_ID_SETTINGS_CSV}")
    kv("by_set_id", str(by_set_id))
    kv("weights", repr(weight_tag))
    kv("loss settings", f"per-set via {SET_ID_SETTINGS_CSV.name}")
    if args.amplitude_weights is None:
        kv("J_feat cycle weights", "per-set (see CSV)")
    else:
        kv(
            "J_feat cycle weights",
            "amplitude (CLI override)" if bool(args.amplitude_weights) else "uniform (CLI override)",
        )
    kv("output parameters", str(out_params))
    kv("output metrics", str(out_metrics))
    kv("plots (path-ordered)", str(plots_dir_ordered))
    if plots_dir_unordered.resolve() != plots_dir_ordered.resolve():
        kv("plots (digitized unordered)", str(plots_dir_unordered))

    rows_out: list[dict[str, Any]] = []

    section("Averaged evaluation -- path-ordered (resampled)")
    for sid in available_resampled:
        csv_path = resolve_resampled_force_deformation_csv(sid, _PROJECT_ROOT)
        if csv_path is None or not csv_path.is_file():
            line(f"skip {sid}: missing resampled force_deformation.csv")
            continue
        df = pd.read_csv(csv_path)
        if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
            line(f"skip {sid}: missing Force[kip] or Deformation[in]")
            continue

        D_exp = df["Deformation[in]"].to_numpy(dtype=float)
        F_exp = df["Force[kip]"].to_numpy(dtype=float)
        loaded = load_cycle_points_resampled(sid)
        points, _ = loaded if loaded is not None else find_cycle_points(df)
        s_f_ref = force_scale_s_f(F_exp)
        s_d_ref = deformation_scale_s_d(D_exp)
        s_e_ref = energy_scale_s_e(D_exp, F_exp)

        prow_block = params_df[params_df["Name"].astype(str) == sid]
        if prow_block.empty:
            line(f"skip {sid}: no parameter rows")
            continue

        p_y_catalog = load_p_y_kip_catalog(
            _PROJECT_ROOT,
            sid,
            float(prow_block.iloc[0]["fyp"]),
            float(prow_block.iloc[0]["A_sc"]),
        )

        specimen_w = weight_fn(sid)
        cm = catalog_metrics_fields(sid, catalog_by_name)
        exp_landmark_cache: dict = {}

        for _, prow in prow_block.iterrows():
            set_id = prow.get("set_id", "?")
            contributes = specimen_w > 0.0
            loss_here = (
                resolve_loss_settings_for_set_id(loss_map, set_id)
                if by_set_id
                else global_loss
            )
            use_amp_w = (
                bool(args.amplitude_weights)
                if args.amplitude_weights is not None
                else loss_here.use_amplitude_weights
            )
            _mse_weights, amp_meta = build_amplitude_weights(
                D_exp,
                points,
                p=loss_here.amplitude_weight_power,
                eps=loss_here.amplitude_weight_eps,
                debug_partition=DEBUG_PARTITION,
                use_amplitude_weights=use_amp_w,
            )
            n_cycles = len(amp_meta)

            try:
                averaged_vec = get_averaged_for_set_id(averaged_dict, set_id, by_set_id=by_set_id)
            except KeyError as e:
                line(f"skip {sid} set {set_id}: {e}")
                continue

            active = resolve_optimize_params_for_set_id(opt_map, set_id, default_list)
            eval_row = merge_averaged_into_row(prow, averaged_vec, active)

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
                s_d=s_d_ref,
                loss=loss_here,
                fy_ksi=float(eval_row["fyp"]),
                a_sc=float(eval_row["A_sc"]),
                L_T=float(eval_row["L_T"]),
                L_y=float(eval_row["L_y"]),
                A_t=float(eval_row["A_t"]),
                E_ksi=float(eval_row["E"]),
                exp_landmark_cache=exp_landmark_cache,
            )
            if bd is None:
                line(f"{sid} set {set_id}: loss breakdown failed")
                continue

            jtot = bd.j_total
            cloud = compute_unordered_cloud_metrics(D_exp, F_exp, D_exp, F_sim)
            mi = _metrics_dict_for_breakdown(bd, loss_here, "initial")
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
                    [cloud.exp_points_raw[:, 0] / ly, cloud.exp_points_raw[:, 1] / fyA]
                )
                num_plot_xy = np.column_stack(
                    [cloud.num_points_raw[:, 0] / ly, cloud.num_points_raw[:, 1] / fyA]
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
                    "calibration_stage": "averaged_eval",
                    "aggregate_by_set_id": by_set_id,
                    **mi,
                    **mf,
                    **_loss_weight_snapshot(loss_here),
                    "S_F": s_f_ref,
                    "S_D": s_d_ref,
                    "S_E": s_e_ref,
                    "P_y_ref": p_y_catalog,
                    "n_cycles": n_cycles,
                    "success": jtot < FAILURE_PENALTY * 0.5,
                }
            )
            line(
                f"{sid} set {set_id}: J={jtot:.6g}  "
                f"J_binenv={cloud.J_binenv:.6g}"
            )

    section("Averaged evaluation -- digitized unordered")
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
        specimen_w = weight_fn(sid)
        cm = catalog_metrics_fields(sid, catalog_by_name)
        s_d_ref = deformation_scale_s_d(D_drive)
        s_f_unordered = force_scale_s_f(F_c)
        s_e_unordered = energy_scale_s_e(u_c, F_c)

        for _, prow in prow_block.iterrows():
            set_id = prow.get("set_id", "?")
            contributes = specimen_w > 0.0
            try:
                averaged_vec = get_averaged_for_set_id(averaged_dict, set_id, by_set_id=by_set_id)
            except KeyError as e:
                line(f"skip {sid} set {set_id}: {e}")
                continue

            active = resolve_optimize_params_for_set_id(opt_map, set_id, default_list)
            eval_row = merge_averaged_into_row(prow, averaged_vec, active)
            sim_row = eval_row_with_envelope_bn_from_unordered(eval_row, cat_row, u_c, F_c)

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

            cloud = compute_unordered_cloud_metrics(u_c, F_c, D_drive, F_sim)
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
                    [cloud.exp_points_raw[:, 0] / ly, cloud.exp_points_raw[:, 1] / fyA]
                )
                num_plot_xy = np.column_stack(
                    [cloud.num_points_raw[:, 0] / ly, cloud.num_points_raw[:, 1] / fyA]
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

            rows_out.append(
                {
                    "Name": sid,
                    "set_id": set_id,
                    "specimen_weight": specimen_w,
                    "contributes_to_aggregate": contributes,
                    **cm,
                    "weight_config": weight_tag,
                    "calibration_stage": "averaged_eval",
                    "aggregate_by_set_id": by_set_id,
                    **_metrics_dict_nan_prefix("initial"),
                    **_metrics_dict_nan_prefix("final"),
                    "initial_unordered_J_binenv": cloud.J_binenv,
                    "final_unordered_J_binenv": cloud.J_binenv,
                    "initial_unordered_J_binenv_l1": cloud.J_binenv_l1,
                    "final_unordered_J_binenv_l1": cloud.J_binenv_l1,
                    **_loss_weight_snapshot(loss_here),
                    "S_F": s_f_unordered,
                    "S_D": s_d_ref,
                    "S_E": s_e_unordered,
                    "P_y_ref": p_y_catalog,
                    "n_cycles": 0,
                    "success": bool(
                        np.isfinite(cloud.J_binenv)
                    ),
                }
            )
            line(
                f"{sid} set {set_id}: digitized unordered overlay + sim + "
                f"J_binenv={cloud.J_binenv:.6g}"
            )

    specimen_set_param_rows: list[dict[str, Any]] = []
    for _, prow in params_df.iterrows():
        set_id = prow.get("set_id", "?")
        averaged_vec = get_averaged_for_set_id(averaged_dict, set_id, by_set_id=by_set_id)
        active_out = resolve_optimize_params_for_set_id(opt_map, set_id, default_list)
        specimen_set_param_rows.append(
            merge_averaged_into_row(prow, averaged_vec, active_out).to_dict()
        )
    p_df = pd.DataFrame(specimen_set_param_rows)
    p_df = p_df[params_df.columns.tolist()]
    p_df = _dataframe_for_param_csv(p_df, sigfigs=OUTPUT_CSV_SIGFIGS)
    out_params.parent.mkdir(parents=True, exist_ok=True)
    p_df.to_csv(out_params, index=False)
    section("Outputs")
    kv("wrote parameters", f"{out_params}  ({len(p_df)} rows, specimen set × merged averaged steel)")

    if not rows_out:
        line("no metrics rows (no successful evaluations).")
        return

    out_df = metrics_dataframe(rows_out)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_metrics, index=False)
    kv("wrote metrics", f"{out_metrics}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
