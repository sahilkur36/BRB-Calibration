"""
Compute correlations between optimal calibration parameters and geometry features.

This uses the same data basis as ``plot_individual_optimal_params_vs_geometry.py``:

- "Optimal" per specimen is the (Name,set_id) row with minimum ``final_J_feat_raw`` over
  successful metrics rows in the requested set_id range, joined to
  ``optimized_brb_parameters.csv``.
- Geometry features are the same 12 columns used in the montage plots.

Outputs:
- A tidy CSV of pairwise correlations (Pearson and Spearman) with sample counts.
- Spearman heatmap PNGs for quick scanning.

Default output locations:
- CSV (train):  summary_statistics/param_geometry_correlations_train.csv
- PNG (train):  results/plots/calibration/individual_optimize/param_geometry_correlations/spearman_heatmaps_train.png
- PNG (extended): same folder, ``spearman_heatmaps_train_extended.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    need_cols = ["Name", "set_id", "R0", "cR1", "cR2", "a1", "a3", "b_p", "b_n", "E"]
    missing = [c for c in need_cols if c not in optimized.columns]
    if missing:
        raise KeyError(f"optimized_brb_parameters missing columns: {missing}")
    opt = optimized[need_cols].copy()

    merged = m.merge(opt, on=["Name", "set_id"], how="inner")
    for c in ("R0", "cR1", "cR2", "a1", "a3", "b_p", "b_n", "E"):
        merged = merged[np.isfinite(pd.to_numeric(merged[c], errors="coerce"))]

    merged = merged.sort_values(["Name", "final_J_feat_raw"])
    best = merged.groupby("Name", as_index=False).first()
    return best


def _resolve_Q(catalog_row: pd.Series) -> float:
    # Same convention as plot_individual_optimal_params_vs_geometry.py: Q = 1 + A_t/A_sc.
    Asc = float(catalog_row["A_c_in2"])
    At = float(catalog_row["A_t_in2"])
    return 1.0 + At / Asc


def _geometry_features(catalog_row: pd.Series, E_kpsi: float, Q: float) -> dict[str, float]:
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


GEOMETRY_COLS: list[str] = [
    "L_y",
    "L_T",
    "A_sc",
    "Ly2_over_A_sc",
    "LT2_over_A_sc",
    "E_div_fy",
    "Q",
    "QE_div_fy",
    "E_Asc_over_fy_Ly2",
    "E_Asc_over_fy_LT2",
    "QE_Asc_over_fy_Ly2",
    "QE_Asc_over_fy_LT2",
]

# Parameters to correlate. Exclude R0 and E because they are kept constant in the runs.
PARAM_COLS: list[str] = ["cR1", "cR2", "a1", "a3", "b_p", "b_n"]

GEOMETRY_LATEX: dict[str, str] = {
    "L_y": r"$L_y$",
    "L_T": r"$L_T$",
    "A_sc": r"$A_{sc}$",
    "Ly2_over_A_sc": r"$L_y^2/A_{sc}$",
    "LT2_over_A_sc": r"$L_T^2/A_{sc}$",
    "E_div_fy": r"$E/f_y$",
    "Q": r"$Q$",
    "QE_div_fy": r"$QE/f_y$",
    "E_Asc_over_fy_Ly2": r"$\frac{E A_{sc}}{f_y L_y^2}$",
    "E_Asc_over_fy_LT2": r"$\frac{E A_{sc}}{f_y L_T^2}$",
    "QE_Asc_over_fy_Ly2": r"$\frac{Q E A_{sc}}{f_y L_y^2}$",
    "QE_Asc_over_fy_LT2": r"$\frac{Q E A_{sc}}{f_y L_T^2}$",
}

PARAM_LATEX: dict[str, str] = {
    "cR1": r"$c_{R1}$",
    "cR2": r"$c_{R2}$",
    "a1": r"$a_1$",
    "a3": r"$a_3$",
    "b_p": r"$b_p$",
    "b_n": r"$b_n$",
}


def _build_train_frame(catalog: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    cat = catalog.set_index("Name")
    b = best.set_index("Name")
    rows: list[dict[str, object]] = []
    for name, crow in cat.iterrows():
        if name not in b.index:
            continue
        # "Train cohort" is the same selection used in the geometry plots.
        gw = pd.to_numeric(crow.get("generalized_weight"), errors="coerce")
        if not (np.isfinite(gw) and float(gw) > 0.0):
            continue
        opt = b.loc[name]
        E = float(opt["E"])
        Q = _resolve_Q(crow)
        g = _geometry_features(crow, E, Q)
        rec: dict[str, object] = {"Name": str(name)}
        rec.update({k: float(v) for k, v in g.items()})
        for p in PARAM_COLS:
            rec[p] = float(opt[p])
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_extended_b_frame(
    catalog: pd.DataFrame, best: pd.DataFrame, apparent: pd.DataFrame
) -> pd.DataFrame:
    """
    Extended dataset "like the plots":
    - geometry is included for all specimens present in the catalog
    - cR1/cR2/a1/a3 are only populated for the train cohort (generalized_weight > 0), from optimal rows
    - b_p/b_n are populated for train from optimal rows, and for non-train from apparent means
    """
    cat = catalog.set_index("Name")
    b = best.set_index("Name")
    app = apparent.set_index("Name")
    rows: list[dict[str, object]] = []
    for name, crow in cat.iterrows():
        arow = app.loc[name] if name in app.index else None

        # E used for geometry normalization. Use optimal E when available; otherwise fall back to 29000.
        E_kpsi = 29000.0
        if name in b.index:
            try:
                E_kpsi = float(b.loc[name].get("E"))
            except Exception:
                E_kpsi = 29000.0
        Q = _resolve_Q(crow)
        g = _geometry_features(crow, float(E_kpsi), float(Q))
        rec: dict[str, object] = {"Name": str(name)}
        rec.update({k: float(v) for k, v in g.items()})

        gw = pd.to_numeric(crow.get("generalized_weight"), errors="coerce")
        is_train = bool(np.isfinite(gw) and float(gw) > 0.0)

        # Steel params: train only (from optimal rows).
        if is_train and name in b.index:
            opt = b.loc[name]
            for p in ("cR1", "cR2", "a1", "a3"):
                rec[p] = float(opt[p])
        else:
            for p in ("cR1", "cR2", "a1", "a3"):
                rec[p] = np.nan

        # b_p / b_n: train from optimal; non-train from apparent means.
        if is_train and name in b.index:
            opt = b.loc[name]
            rec["b_p"] = float(opt["b_p"])
            rec["b_n"] = float(opt["b_n"])
        else:
            bp = arow.get("b_p_mean") if arow is not None else np.nan
            bn = arow.get("b_n_mean") if arow is not None else np.nan
            rec["b_p"] = float(bp) if pd.notna(bp) else np.nan
            rec["b_n"] = float(bn) if pd.notna(bn) else np.nan

        rows.append(rec)
    return pd.DataFrame(rows)


def _pairwise_n(x: pd.Series, y: pd.Series) -> int:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    m = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
    return int(np.sum(m))


def _tidy_correlations(df: pd.DataFrame, *, geometry_cols: list[str], param_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p in param_cols:
        for g in geometry_cols:
            n = _pairwise_n(df[p], df[g])
            pear = df[[p, g]].corr(method="pearson", min_periods=2).iloc[0, 1]
            spear = df[[p, g]].corr(method="spearman", min_periods=2).iloc[0, 1]
            rows.append(
                {
                    "param": p,
                    "geometry": g,
                    "n": n,
                    "pearson_r": float(pear) if pd.notna(pear) else np.nan,
                    "spearman_r": float(spear) if pd.notna(spear) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(["param", "geometry"]).reset_index(drop=True)
    return out


def _spearman_matrix(df: pd.DataFrame, *, geometry_cols: list[str], param_cols: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(index=param_cols, columns=geometry_cols, dtype=float)
    for p in param_cols:
        for g in geometry_cols:
            v = df[[p, g]].corr(method="spearman", min_periods=2).iloc[0, 1]
            mat.loc[p, g] = float(v) if pd.notna(v) else np.nan
    return mat


def _pairwise_n_matrix(
    df: pd.DataFrame, *, row_cols: list[str], col_cols: list[str]
) -> pd.DataFrame:
    nmat = pd.DataFrame(index=row_cols, columns=col_cols, dtype=float)
    for r in row_cols:
        for c in col_cols:
            nmat.loc[r, c] = _pairwise_n(df[r], df[c])
    return nmat


def _heatmap_spearman(
    mat: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    n_mat: pd.DataFrame | None = None,
    x_label_map: dict[str, str] | None = None,
    y_label_map: dict[str, str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = mat.to_numpy(dtype=float)
    fig_w = max(10.0, 0.6 * mat.shape[1])
    fig_h = max(5.0, 0.45 * mat.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    im = ax.imshow(z, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    x_label_map = x_label_map or {}
    y_label_map = y_label_map or {}
    ax.set_yticks(
        np.arange(mat.shape[0]),
        labels=[y_label_map.get(str(x), str(x)) for x in mat.index],
    )
    ax.set_xticks(
        np.arange(mat.shape[1]),
        labels=[x_label_map.get(str(x), str(x)) for x in mat.columns],
        rotation=45,
        ha="right",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Spearman ρ")
    # Annotate with rho only, or rho + n (extended).
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = z[i, j]
            if not np.isfinite(v):
                txt = "—"
            else:
                if n_mat is not None:
                    n = n_mat.iloc[i, j]
                    nn = int(n) if pd.notna(n) else 0
                    txt = f"{v:+.2f}\n(n={nn})"
                else:
                    txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _combined_train_heatmap(
    mat_pg: pd.DataFrame,
    mat_pp: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    n_pg: pd.DataFrame | None = None,
    n_pp: pd.DataFrame | None = None,
) -> None:
    """
    One heatmap with columns [geometry..., params...] and rows [params...].
    Left block: params vs geometry. Right block: params vs params.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if list(mat_pg.index) != list(mat_pp.index):
        raise ValueError("Expected params-vs-geometry and params-vs-params to share the same row index.")

    cols = list(mat_pg.columns) + list(mat_pp.columns)
    combined = pd.concat([mat_pg, mat_pp], axis=1)
    combined = combined.loc[mat_pg.index, cols]

    combined_n: pd.DataFrame | None = None
    if n_pg is not None and n_pp is not None:
        combined_n = pd.concat([n_pg, n_pp], axis=1).loc[mat_pg.index, cols]

    z = combined.to_numpy(dtype=float)
    fig_w = max(14.0, 0.55 * combined.shape[1] + 4.0)
    fig_h = max(6.0, 0.5 * combined.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    im = ax.imshow(z, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_title(title)

    x_map: dict[str, str] = {}
    x_map.update(GEOMETRY_LATEX)
    x_map.update(PARAM_LATEX)
    y_map = PARAM_LATEX

    ax.set_yticks(
        np.arange(combined.shape[0]),
        labels=[y_map.get(str(x), str(x)) for x in combined.index],
    )
    ax.set_xticks(
        np.arange(combined.shape[1]),
        labels=[x_map.get(str(x), str(x)) for x in combined.columns],
        rotation=45,
        ha="right",
    )

    # Separator between geometry columns and parameter columns.
    sep_x = len(mat_pg.columns) - 0.5
    ax.axvline(sep_x, color="k", linewidth=1.0, alpha=0.6)

    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            v = z[i, j]
            if not np.isfinite(v):
                txt = "—"
            else:
                if combined_n is not None:
                    n = combined_n.iloc[i, j]
                    nn = int(n) if pd.notna(n) else 0
                    txt = f"{v:+.2f}\n(n={nn})"
                else:
                    txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Spearman ρ")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=root)
    p.add_argument(
        "--catalog",
        type=Path,
        default=root / "config" / "calibration" / "BRB-Specimens.csv",
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=root
        / "results"
        / "calibration"
        / "individual_optimize"
        / "optimized_brb_parameters_metrics.csv",
    )
    p.add_argument(
        "--optimized-params",
        type=Path,
        default=root
        / "results"
        / "calibration"
        / "individual_optimize"
        / "optimized_brb_parameters.csv",
    )
    p.add_argument("--sets", type=str, default="1-10")
    p.add_argument(
        "--apparent-bn-bp",
        type=Path,
        default=root / "results" / "calibration" / "specimen_apparent_bn_bp.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=root / "summary_statistics" / "param_geometry_correlations_train.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root
        / "results"
        / "plots"
        / "calibration"
        / "individual_optimize"
        / "param_geometry_correlations",
    )
    args = p.parse_args()

    set_ids = _parse_set_range(args.sets)
    catalog = _read_csv_skip_hash(Path(args.catalog))
    metrics = pd.read_csv(Path(args.metrics))
    optimized = pd.read_csv(Path(args.optimized_params))
    apparent = pd.read_csv(Path(args.apparent_bn_bp))

    # Render mathtext nicely (no external LaTeX dependency).
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    best = _pick_optimal_parameter_rows(metrics, optimized, set_ids)
    df_train = _build_train_frame(catalog, best)
    if df_train.empty:
        raise SystemExit(
            "No train-cohort rows available for correlation (check generalized_weight and inputs)."
        )
    df_ext = _build_extended_b_frame(catalog, best, apparent)

    tidy = _tidy_correlations(df_train, geometry_cols=GEOMETRY_COLS, param_cols=PARAM_COLS)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(args.out_csv, index=False)

    out_dir = Path(args.out_dir)

    # --- Params vs geometry ---
    mat = _spearman_matrix(df_train, geometry_cols=GEOMETRY_COLS, param_cols=PARAM_COLS)
    nmat = _pairwise_n_matrix(df_ext, row_cols=PARAM_COLS, col_cols=GEOMETRY_COLS)

    # Also write the wide Spearman matrix (easy to diff / inspect).
    out_dir.mkdir(parents=True, exist_ok=True)
    mat.to_csv(out_dir / "spearman_matrix_train.csv")

    mat.to_csv(out_dir / "spearman_matrix_train_extended.csv")

    # --- Params vs params (self-correlation) ---
    pmat = _spearman_matrix(df_train, geometry_cols=PARAM_COLS, param_cols=PARAM_COLS)
    pnmat = _pairwise_n_matrix(df_ext, row_cols=PARAM_COLS, col_cols=PARAM_COLS)
    pmat.to_csv(out_dir / "spearman_params_matrix_train.csv")
    pmat.to_csv(out_dir / "spearman_params_matrix_train_extended.csv")

    # --- Combined (single-matrix) figures ---
    _combined_train_heatmap(
        mat,
        pmat,
        out_path=out_dir / "spearman_heatmaps_train.png",
        title="Spearman correlation: optimal params vs geometry + params (train)",
    )

    # Extended: correlations computed on df_ext (b_p/b_n filled for non-train via apparent means),
    # and annotated with pairwise n.
    mat_ext = _spearman_matrix(df_ext, geometry_cols=GEOMETRY_COLS, param_cols=PARAM_COLS)
    pmat_ext = _spearman_matrix(df_ext, geometry_cols=PARAM_COLS, param_cols=PARAM_COLS)
    _combined_train_heatmap(
        mat_ext,
        pmat_ext,
        out_path=out_dir / "spearman_heatmaps_train_extended.png",
        title="Spearman correlation: optimal params vs geometry + params (extended b_p/b_n)",
        n_pg=nmat,
        n_pp=pnmat,
    )


if __name__ == "__main__":
    main()

