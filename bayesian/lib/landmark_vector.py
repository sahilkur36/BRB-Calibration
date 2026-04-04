"""Flat weighted landmark vectors for calibration_data.csv and model results.out."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .jfeat_landmarks import (
    LANDMARK_SLOTS_F0_EXTREMAL_D,
    N_LANDMARK_SLOTS,
    _all_f_level_crossings,
    deformation_scale_s_d,
    extract_cycle_landmarks,
    pair_sim_cycle_landmarks,
    _slot_error_combined_sq,
)


def force_scale_s_f(F_exp: np.ndarray) -> float:
    f = np.asarray(F_exp, dtype=float)
    r = float(np.nanmax(f) - np.nanmin(f))
    if not np.isfinite(r) or r <= 0.0:
        return 1.0
    return r


def load_force_deformation_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    if "Deformation[in]" not in df.columns or "Force[kip]" not in df.columns:
        raise ValueError(f"Expected Deformation[in] and Force[kip] in {path}")
    D = df["Deformation[in]"].to_numpy(dtype=float)
    F = df["Force[kip]"].to_numpy(dtype=float)
    return D, F


def load_cycle_meta_json(path: Path) -> tuple[list[dict], float | None, float | None, int | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data["meta"]
    s_f = data.get("s_f")
    s_d = data.get("s_d")
    n = data.get("n")
    return meta, s_f, s_d, int(n) if n is not None else None


def _append_weighted_cycle(
    out: list[float],
    le_m: list[tuple[float, float] | None],
    ls: list[tuple[float, float] | None],
    w_c: float,
    s_f: float,
    s_d: float,
    *,
    from_experiment: bool,
) -> None:
    n_c = sum(
        1
        for slot in range(N_LANDMARK_SLOTS)
        if _slot_error_combined_sq(le_m[slot], ls[slot], s_f, s_d) is not None
    )
    if n_c == 0:
        return

    scale = math.sqrt(w_c / n_c)
    for slot in range(N_LANDMARK_SLOTS):
        if _slot_error_combined_sq(le_m[slot], ls[slot], s_f, s_d) is None:
            continue
        pt = le_m[slot] if from_experiment else ls[slot]
        if pt is None:
            continue
        d_v, f_v = float(pt[0]), float(pt[1])
        if not (np.isfinite(d_v) and np.isfinite(f_v)):
            continue
        out.append(scale * d_v / s_d)
        out.append(scale * f_v / s_f)


def weighted_landmark_vector_experiment(
    D: np.ndarray,
    F_exp: np.ndarray,
    meta: list[dict],
    *,
    fy: float,
    a_sc: float,
    dy: float | None,
    s_d: float | None = None,
    s_f: float | None = None,
) -> list[float]:
    """
    One flat row of weighted landmark (D, F) from the **experimental** curve at each paired grid
    index (same pairing rule as the model path). For ``calibration_data.csv``.
    """
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    if s_d is None:
        s_d = deformation_scale_s_d(D)
    if s_f is None:
        s_f = force_scale_s_f(F_exp)

    out: list[float] = []
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0
        if e <= s:
            continue

        le = extract_cycle_landmarks(
            D, F_exp, s, e, fy=fy, a_sc=a_sc, dy=dy
        )
        ls, le_m, _ = pair_sim_cycle_landmarks(
            D, F_exp, F_exp, s, e, le, fy=fy, a_sc=a_sc, geometry_f=F_exp
        )
        _append_weighted_cycle(out, le_m, ls, w_c, s_f, s_d, from_experiment=True)

    return out


def build_landmark_feature_cache(
    D: np.ndarray,
    F_exp: np.ndarray,
    meta: list[dict],
    *,
    fy: float,
    a_sc: float,
    dy: float | None,
    s_d: float | None = None,
    s_f: float | None = None,
) -> dict:
    """
    Precompute per-cycle experimental ``le_metric`` and vertex indices ``j`` for the shared grid.

    Use with :func:`weighted_landmark_vector_model` so forward runs only need ``D`` and ``F_sim``.
    """
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    if s_d is None:
        s_d = deformation_scale_s_d(D)
    if s_f is None:
        s_f = force_scale_s_f(F_exp)

    cycles: list[dict] = []
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            continue
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0

        le = extract_cycle_landmarks(
            D, F_exp, s, e, fy=fy, a_sc=a_sc, dy=dy
        )
        _, le_m, j_slot = pair_sim_cycle_landmarks(
            D, F_exp, F_exp, s, e, le, fy=fy, a_sc=a_sc, geometry_f=F_exp
        )

        le_serial: list[list[float] | None] = []
        for i in range(N_LANDMARK_SLOTS):
            p = le_m[i]
            if p is None:
                le_serial.append(None)
            else:
                le_serial.append([float(p[0]), float(p[1])])

        cycles.append(
            {
                "start": s,
                "end": e,
                "w_c": w_c,
                "j": [None if j is None else int(j) for j in j_slot],
                "le_metric": le_serial,
            }
        )

    return {
        "version": 1,
        "n": int(len(D)),
        "s_d": float(s_d),
        "s_f": float(s_f),
        "cycles": cycles,
    }


def weighted_landmark_vector_model(
    D: np.ndarray,
    F_sim: np.ndarray,
    cache: dict,
) -> list[float]:
    """
    One flat row of weighted landmark (D, F) from **simulation**, using pairing frozen in ``cache``
    (from :func:`build_landmark_feature_cache`). For ``results.out``.
    """
    if int(cache.get("version", 0)) != 1:
        raise ValueError("landmark cache: expected version 1")

    D = np.asarray(D, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    n = int(cache["n"])
    if len(D) != n or len(F_sim) != n:
        raise ValueError(
            f"landmark cache expects length {n}; got D={len(D)}, F_sim={len(F_sim)}"
        )

    s_d = float(cache["s_d"])
    s_f = float(cache["s_f"])
    out: list[float] = []

    for cyc in cache["cycles"]:
        s, e = int(cyc["start"]), int(cyc["end"])
        w_c = float(cyc.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0
        if e <= s:
            continue

        j_slots: list[int | None] = cyc["j"]
        le_metric: list[tuple[float, float] | None] = []
        for p in cyc["le_metric"]:
            if p is None:
                le_metric.append(None)
            else:
                le_metric.append((float(p[0]), float(p[1])))

        ls: list[tuple[float, float] | None] = [None] * N_LANDMARK_SLOTS
        for slot in range(N_LANDMARK_SLOTS):
            if slot in LANDMARK_SLOTS_F0_EXTREMAL_D:
                continue
            j = j_slots[slot]
            if j is None:
                continue
            dj = float(D[j])
            fs = float(F_sim[j])
            if np.isfinite(dj) and np.isfinite(fs):
                ls[slot] = (dj, fs)

        if le_metric[6] is not None or le_metric[7] is not None:
            f0 = _all_f_level_crossings(D, F_sim, s, e, 0.0)
            if f0:
                if le_metric[6] is not None:
                    ls[6] = max(f0, key=lambda p: p[0])
                if le_metric[7] is not None:
                    ls[7] = min(f0, key=lambda p: p[0])

        _append_weighted_cycle(out, le_metric, ls, w_c, s_f, s_d, from_experiment=False)

    return out


def write_landmark_cache_json(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def read_landmark_cache_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv_single_row(path: Path, values: list[float]) -> None:
    path.write_text(",".join(str(v) for v in values), encoding="utf-8")
