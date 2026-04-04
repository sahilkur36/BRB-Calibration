"""
Cycle-weighted landmark loss J_feat for prescribed (D, F) hysteresis.

Twelve landmarks per weight cycle ``[s, e)`` on the experimental polyline (see
``extract_cycle_landmarks``). ``J_feat`` uses, for every slot, the **sum** of normalized
squared **force** and **displacement** error at the paired exp/sim points, so
pairing logic does not depend on per-slot metric type.

Physical quantities use one system throughout this bundle: length [in], stress [ksi],
force [kip], area [in²] (see bayesian README).
"""
from __future__ import annotations

import math

import numpy as np

from .brace_geometry import compute_Q


def deformation_scale_s_d(D: np.ndarray) -> float:
    """S_D = max(D) - min(D); fallback 1.0."""
    d = np.asarray(D, dtype=float)
    r = float(np.nanmax(d) - np.nanmin(d))
    if not np.isfinite(r) or r <= 0.0:
        return 1.0
    return r


def landmark_force_threshold(fy: float, a_sc: float) -> float:
    """Yield force threshold = f_y * A_sc (landmark gating, no 1.1 factor)."""
    return float(fy) * float(a_sc)


def yield_displacement_dy(
    *,
    fy: float,
    E: float,
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
) -> float:
    """Dy = (f_y / E_hat) * L_T with E_hat = Q * E and Q from brace geometry."""
    if not all(map(math.isfinite, (fy, E, L_T))) or E <= 0.0 or L_T <= 0.0:
        return float("nan")
    try:
        Q = compute_Q(L_T, L_y, A_sc, A_t)
    except Exception:
        return float("nan")
    E_hat = Q * E
    if not (math.isfinite(E_hat) and E_hat > 0.0):
        return float("nan")
    return (fy / E_hat) * L_T


def _interp_edge(
    da: float, db: float, fa: float, fb: float, target_f: float
) -> tuple[float, float] | None:
    """If target_f is bracketed on the edge, return (d_star, f_star) with f_star=target_f."""
    if not (math.isfinite(da) and math.isfinite(db) and math.isfinite(fa) and math.isfinite(fb)):
        return None
    lo_f, hi_f = (fa, fb) if fa <= fb else (fb, fa)
    if target_f < lo_f or target_f > hi_f:
        return None
    if fb == fa:
        return None
    frac = (target_f - fa) / (fb - fa)
    d_star = da + frac * (db - da)
    return (float(d_star), float(target_f))


def _first_f_level_crossing(
    D: np.ndarray,
    F: np.ndarray,
    e: int,
    k_min: int,
    level: float,
) -> tuple[tuple[float, float], int] | None:
    """First edge [k,k+1) with k >= k_min, k+1 < e, bracketing F=level. Returns ((d,f), k_gate)."""
    k_min = max(0, k_min)
    for k in range(k_min, e - 1):
        da, db = float(D[k]), float(D[k + 1])
        fa, fb = float(F[k]), float(F[k + 1])
        if fa == level:
            return ((da, level), k + 1)
        if fb == level:
            return ((db, level), k + 2)
        if fa * fb < 0.0 and level == 0.0:
            if fb == fa:
                continue
            frac = -fa / (fb - fa)
            d_star = da + frac * (db - da)
            return ((float(d_star), 0.0), k + 1)
        hit = _interp_edge(da, db, fa, fb, level)
        if hit is not None:
            return (hit, k + 1)
    return None


def _d_zero_cross_subpath(
    D: np.ndarray,
    F: np.ndarray,
    e: int,
    i_lo: int,
    i_hi_vert: int,
) -> tuple[float, float] | None:
    """
    First D=0 crossing on edges between vertices ``i_lo .. i_hi_vert`` inclusive.
    Returns ``(0.0, f_star)`` interpolated on the crossing edge.
    """
    if i_hi_vert <= i_lo or i_lo >= e:
        return None
    i_hi_vert = min(i_hi_vert, e - 1)
    for k in range(i_lo, i_hi_vert):
        if k + 1 >= e:
            break
        da, db = float(D[k]), float(D[k + 1])
        fa, fb = float(F[k]), float(F[k + 1])
        if not all(math.isfinite(x) for x in (da, db, fa, fb)):
            continue
        if da == 0.0:
            return (0.0, float(fa))
        if db == 0.0:
            return (0.0, float(fb))
        if da * db < 0.0:
            if db == da:
                continue
            frac = -da / (db - da)
            f_star = fa + frac * (fb - fa)
            return (0.0, float(f_star))
    return None


def _first_peak_anchor_i0_from_peaks(
    i_max: int,
    i_min: int,
    slot3_ok: bool,
    slot4_ok: bool,
) -> int | None:
    """Earliest index among valid tension (``i_max``) / compression (``i_min``) peaks, or ``None``."""
    anchors: list[int] = []
    if slot3_ok:
        anchors.append(int(i_max))
    if slot4_ok:
        anchors.append(int(i_min))
    return int(min(anchors)) if anchors else None


def _slot2_vertex_smallest_abs_d_on_first_sign_edge(
    D: np.ndarray,
    F: np.ndarray,
    e: int,
    i0: int,
) -> tuple[float, float] | None:
    """
    First edge after ``i0`` where ``D`` leaves the same side of zero as ``D[i0]``; return the
    endpoint of that edge with smaller ``|D|`` (tie: earlier index).
    If ``D[i0]==0``, fall back to first interpolated ``D=0`` crossing after ``i0``.
    """
    if i0 < 0 or i0 >= e:
        return None
    d0 = float(D[i0])
    if not np.isfinite(d0):
        return None
    if d0 == 0.0:
        return _d_zero_cross_subpath(D, F, e, i0, e - 1)

    for m in range(i0 + 1, e):
        dm = float(D[m])
        if not np.isfinite(dm):
            continue
        if d0 > 0.0:
            if dm > 0.0:
                continue
        else:
            if dm < 0.0:
                continue
        k = m - 1
        da, db = float(D[k]), float(D[k + 1])
        fa, fb = float(F[k]), float(F[k + 1])
        if not all(np.isfinite(x) for x in (da, db, fa, fb)):
            return None
        if abs(da) < abs(db):
            j = k
        elif abs(db) < abs(da):
            j = k + 1
        else:
            j = k
        return (float(D[j]), float(F[j]))
    return None


N_LANDMARK_SLOTS = 12
# Slots 7–8 (1-based): F=0 at extremal D; ``pair_sim_cycle_landmarks`` uses F_sim=0 crossings.
# Other slots pair on the shared D-grid via nearest vertex (see ``pair_sim_cycle_landmarks``).
LANDMARK_SLOTS_F0_EXTREMAL_D = frozenset({6, 7})


def _landmark_pair_path_window(
    slot: int,
    s: int,
    e: int,
    i_max: int,
    i_min: int,
    i_yt: int | None,
    i_yc: int | None,
    *,
    i0_exp: int | None = None,
) -> tuple[int, int]:
    """Vertex index window ``[s_path, e_path)`` for grid pairing (same subpaths as former interp)."""
    s_path, e_path = s, e
    if slot == 1 and i0_exp is not None:
        s_path = i0_exp
    elif slot == 8:
        e_path = i_max + 1
    elif slot == 9:
        e_path = i_min + 1
    elif slot == 10 and i_yt is not None:
        s_path = i_yt
    elif slot == 11 and i_yc is not None:
        s_path = i_yc
    if e_path <= s_path + 1:
        s_path, e_path = s, e
    return s_path, e_path


def _nearest_vertex_index(
    D: np.ndarray,
    d_star: float,
    s_path: int,
    e_path: int,
) -> int | None:
    """Index ``j ∈ [s_path, e_path)`` minimizing ``|D[j]-d_star|``; tie-break smallest ``j``."""
    if not np.isfinite(d_star) or e_path <= s_path:
        return None
    d = np.asarray(D, dtype=float)
    best_j: int | None = None
    best_abs = float("inf")
    for j in range(s_path, e_path):
        dj = float(d[j])
        if not np.isfinite(dj):
            continue
        ad = abs(dj - d_star)
        if ad < best_abs or (ad == best_abs and (best_j is None or j < best_j)):
            best_abs = ad
            best_j = j
    return best_j


def _interp_edge_d_level(
    da: float, db: float, fa: float, fb: float, target_d: float
) -> tuple[float, float] | None:
    """If target_d is bracketed on the edge, return (d_star,f_star) with d_star=target_d."""
    if not (math.isfinite(da) and math.isfinite(db) and math.isfinite(fa) and math.isfinite(fb)):
        return None
    lo_d, hi_d = (da, db) if da <= db else (db, da)
    if target_d < lo_d or target_d > hi_d:
        return None
    if db == da:
        return None
    frac = (target_d - da) / (db - da)
    f_star = fa + frac * (fb - fa)
    return (float(target_d), float(f_star))


def _all_f_level_crossings(
    D: np.ndarray,
    F: np.ndarray,
    s: int,
    e: int,
    level: float,
) -> list[tuple[float, float]]:
    """All F=level crossings on edges within [s,e). Returns list of (d_star, level)."""
    out: list[tuple[float, float]] = []
    s = max(0, int(s))
    e = int(e)
    if e <= s + 1:
        return out
    for k in range(s, e - 1):
        da, db = float(D[k]), float(D[k + 1])
        fa, fb = float(F[k]), float(F[k + 1])
        if not all(math.isfinite(x) for x in (da, db, fa, fb)):
            continue
        if fa == level:
            out.append((da, float(level)))
            continue
        if fb == level:
            out.append((db, float(level)))
            continue
        if level == 0.0 and fa * fb < 0.0:
            if fb == fa:
                continue
            frac = -fa / (fb - fa)
            d_star = da + frac * (db - da)
            out.append((float(d_star), 0.0))
            continue
        hit = _interp_edge(da, db, fa, fb, float(level))
        if hit is not None:
            out.append(hit)
    return out


def _all_d_level_crossings(
    D: np.ndarray,
    F: np.ndarray,
    s: int,
    e: int,
    target_d: float,
) -> list[tuple[float, float]]:
    """All D=target_d crossings on edges within [s,e). Returns list of (target_d, f_star)."""
    out: list[tuple[float, float]] = []
    if not math.isfinite(float(target_d)):
        return out
    s = max(0, int(s))
    e = int(e)
    if e <= s + 1:
        return out
    td = float(target_d)
    for k in range(s, e - 1):
        da, db = float(D[k]), float(D[k + 1])
        fa, fb = float(F[k]), float(F[k + 1])
        if not all(math.isfinite(x) for x in (da, db, fa, fb)):
            continue
        if da == td:
            out.append((td, float(fa)))
            continue
        if db == td:
            out.append((td, float(fb)))
            continue
        if (da - td) * (db - td) < 0.0:
            hit = _interp_edge_d_level(da, db, fa, fb, td)
            if hit is not None:
                out.append(hit)
    return out


def extract_cycle_landmarks(
    D: np.ndarray,
    F: np.ndarray,
    s: int,
    e: int,
    *,
    fy: float,
    a_sc: float,
    dy: float | None = None,
) -> list[tuple[float, float] | None]:
    """
    Twelve experimental landmarks on ``[s, e)``.

    Slot meanings (1-based labels in debug plots/CSVs):

    1. Cycle start point (D[s], F[s]).
    2. After ``i0 = min`` of valid tension / compression peak indices: first edge ``(k,k+1)`` with
       ``k >= i0`` where ``D`` crosses from the same side of zero as ``D[i0]`` to the other
       (first ``m>i0`` with ``D[m]`` non-positive if ``D[i0]>0``, else non-negative if ``D[i0]<0``).
       Landmark is the endpoint ``k`` or ``k+1`` with smaller ``|D|`` (tie: smaller index).
       If ``D[i0]==0``, use the first interpolated ``D=0`` crossing after ``i0`` (legacy).
    3. Max tension point (global max F in [s,e)).
    4. Max compression point (global min F in [s,e)).
    5. Tension yield: among points with F > +F_thr and D < 0 in [s,e), pick minimal D (tie: larger F).
    6. Compression yield: among points with F < -F_thr and D > 0 in [s,e), pick maximal D (tie: more negative F).
    7. F=0 crossing with maximal D (interpolated) within [s,e). (Displacement-scored.)
    8. F=0 crossing with minimal D (interpolated) within [s,e). (Displacement-scored.)
    9. Last crossing of D = (D_at_max_tension)/2 before the tension peak (within [s,i_max]).
    10. Last crossing of D = (D_at_max_compression)/2 before the compression peak (within [s,i_min]).
    11. First crossing of D = (D_at_tension_yield)/2 after tension yield (within [i_yield,e)).
    12. First crossing of D = (D_at_compression_yield)/2 after compression yield (within [i_yield,e)).

    ``F_thr = f_y * A_sc``.
    """
    out: list[tuple[float, float] | None] = [None] * N_LANDMARK_SLOTS
    if e <= s:
        return out

    D = np.asarray(D, dtype=float)
    F = np.asarray(F, dtype=float)
    F_thr = landmark_force_threshold(fy, a_sc)
    if not (np.isfinite(F_thr) and F_thr > 0.0):
        return out

    fseg = F[s:e]
    if len(fseg) == 0:
        return out
    i_max = s + int(np.argmax(fseg))
    i_min = s + int(np.argmin(fseg))

    # Gate: require the cycle to exceed ±2*Dy in at least one direction.
    #
    # If Dy is missing/invalid, do not compute landmarks for this cycle.
    if dy is None or not (np.isfinite(float(dy)) and float(dy) > 0.0):
        return out
    dseg = D[s:e]
    if len(dseg) == 0:
        return out
    d_max = float(np.nanmax(dseg))
    d_min = float(np.nanmin(dseg))
    thr = 2.0 * float(dy)
    if not (np.isfinite(d_max) and np.isfinite(d_min)) or not (d_max >= thr or d_min <= -thr):
        return out

    # Slot 1: cycle start
    out[0] = (float(D[s]), float(F[s])) if np.isfinite(D[s]) and np.isfinite(F[s]) else None

    # Slots 3–4: max tension / max compression points, with sign requirements on (D, F).
    # Slot 3 must be a positive-force peak occurring at positive deformation.
    # Slot 4 must be a negative-force peak occurring at negative deformation.
    slot3_ok = bool(
        np.isfinite(D[i_max])
        and np.isfinite(F[i_max])
        and float(F[i_max]) > 0.0
        and float(D[i_max]) > 0.0
    )
    slot4_ok = bool(
        np.isfinite(D[i_min])
        and np.isfinite(F[i_min])
        and float(F[i_min]) < 0.0
        and float(D[i_min]) < 0.0
    )
    out[2] = (float(D[i_max]), float(F[i_max])) if slot3_ok else None
    out[3] = (float(D[i_min]), float(F[i_min])) if slot4_ok else None

    i0 = _first_peak_anchor_i0_from_peaks(i_max, i_min, slot3_ok, slot4_ok)
    if i0 is not None:
        p2 = _slot2_vertex_smallest_abs_d_on_first_sign_edge(D, F, e, i0)
        if p2 is not None:
            out[1] = p2

    # Slots 5–6: yields by extremal D among threshold-exceeding points in [s,e)
    i_yt: int | None = None
    best_d_yt = float("inf")
    best_f_yt = -float("inf")
    for i in range(s, e):
        fi = float(F[i])
        if not np.isfinite(fi) or fi <= F_thr:
            continue
        di = float(D[i])
        if not np.isfinite(di) or di >= 0.0:
            continue
        if di < best_d_yt or (di == best_d_yt and fi > best_f_yt):
            best_d_yt = di
            best_f_yt = fi
            i_yt = i
    if i_yt is not None:
        out[4] = (float(D[i_yt]), float(F[i_yt]))

    i_yc: int | None = None
    best_d_yc = -float("inf")
    best_f_yc = float("inf")
    for i in range(s, e):
        fi = float(F[i])
        if not np.isfinite(fi) or fi >= -F_thr:
            continue
        di = float(D[i])
        if not np.isfinite(di) or di <= 0.0:
            continue
        if di > best_d_yc or (di == best_d_yc and fi < best_f_yc):
            best_d_yc = di
            best_f_yc = fi
            i_yc = i
    if i_yc is not None:
        out[5] = (float(D[i_yc]), float(F[i_yc]))

    # Slots 7–8: F=0 crossings with extremal D (interpolated) within [s,e)
    f0 = _all_f_level_crossings(D, F, s, e, 0.0)
    if f0:
        out[6] = max(f0, key=lambda p: p[0])
        out[7] = min(f0, key=lambda p: p[0])

    # Slot 9: last D = D_at_max_tension/2 crossing before tension peak (within [s,i_max])
    if slot3_ok and np.isfinite(D[i_max]) and i_max > s:
        tgt = 0.5 * float(D[i_max])
        hits = _all_d_level_crossings(D, F, s, i_max + 1, tgt)
        if hits:
            out[8] = hits[-1]

    # Slot 10: last D = D_at_max_compression/2 crossing before compression peak (within [s,i_min])
    if slot4_ok and np.isfinite(D[i_min]) and i_min > s:
        tgt = 0.5 * float(D[i_min])
        hits = _all_d_level_crossings(D, F, s, i_min + 1, tgt)
        if hits:
            out[9] = hits[-1]

    # Slot 11: first D = D_at_tension_yield/2 crossing after tension yield
    if i_yt is not None and np.isfinite(D[i_yt]) and e > i_yt + 1:
        tgt = 0.5 * float(D[i_yt])
        hits = _all_d_level_crossings(D, F, i_yt, e, tgt)
        if hits:
            out[10] = hits[0]

    # Slot 12: first D = D_at_compression_yield/2 crossing after compression yield
    if i_yc is not None and np.isfinite(D[i_yc]) and e > i_yc + 1:
        tgt = 0.5 * float(D[i_yc])
        hits = _all_d_level_crossings(D, F, i_yc, e, tgt)
        if hits:
            out[11] = hits[0]

    return out


def pair_sim_cycle_landmarks(
    D: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    s: int,
    e: int,
    le: list[tuple[float, float] | None],
    *,
    fy: float,
    a_sc: float,
    geometry_f: np.ndarray | None = None,
) -> tuple[list[tuple[float, float] | None], list[tuple[float, float] | None]]:
    """
    Pair experimental landmarks to simulated ones on the **shared displacement grid** ``D``.

    - Slots other than F=0 extremal (indices 6,7): target displacement ``d_e`` from ``le[slot]``;
      find vertex index ``j`` in the same path window as ``extract_cycle_landmarks`` / former interp
      (via ``_landmark_pair_path_window``) minimizing ``|D[j]-d_e|``; set
      ``le_metric[slot]=(D[j],F_exp[j])`` and ``ls[slot]=(D[j],F_sim[j])``. If no valid ``j``, ``ls``
      stays ``None`` and ``le_metric[slot]`` stays the original ``le[slot]``.
    - F=0 extremal-D (6,7): unchanged semantics — sim from F_sim=0 crossings when the matching exp
      slot exists; ``le_metric[6/7]`` remains ``le[6/7]``.

    If ``geometry_f`` is set (e.g. experimental force), it is used for path windows / yield picks
    (``i_max``, ``i_min``, ``i_yt``, ``i_yc``); ``F_sim`` still supplies values at vertex ``j`` and F=0
    crossings. If ``None``, behavior matches legacy use of ``F_sim`` for geometry.

    Returns ``(ls, le_metric, j_slot)`` where ``j_slot[slot]`` is the vertex index used for
    non-F0-extremal paired slots, else ``None``. Use ``le_metric`` with ``ls`` in feature loss
    so exp/sim share index ``j``.
    """
    ls: list[tuple[float, float] | None] = [None] * N_LANDMARK_SLOTS
    le_metric: list[tuple[float, float] | None] = list(le)
    j_slot: list[int | None] = [None] * N_LANDMARK_SLOTS
    if e <= s:
        return ls, le_metric, j_slot

    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if D.shape != F_exp.shape or D.shape != F_sim.shape:
        return ls, le_metric, j_slot

    F_geom = np.asarray(F_sim if geometry_f is None else geometry_f, dtype=float)
    if F_geom.shape != D.shape:
        return ls, le_metric, j_slot

    F_thr = landmark_force_threshold(fy, a_sc)

    fseg = F_geom[s:e]
    if len(fseg) == 0:
        return ls, le_metric, j_slot
    i_max = s + int(np.argmax(fseg))
    i_min = s + int(np.argmin(fseg))

    fseg_exp = F_exp[s:e]
    i_max_exp = s + int(np.argmax(fseg_exp))
    i_min_exp = s + int(np.argmin(fseg_exp))
    slot3_ok_e = bool(
        np.isfinite(D[i_max_exp])
        and np.isfinite(F_exp[i_max_exp])
        and float(F_exp[i_max_exp]) > 0.0
        and float(D[i_max_exp]) > 0.0
    )
    slot4_ok_e = bool(
        np.isfinite(D[i_min_exp])
        and np.isfinite(F_exp[i_min_exp])
        and float(F_exp[i_min_exp]) < 0.0
        and float(D[i_min_exp]) < 0.0
    )
    i0_exp = _first_peak_anchor_i0_from_peaks(
        i_max_exp, i_min_exp, slot3_ok_e, slot4_ok_e
    )

    i_yt: int | None = None
    best_d_yt = float("inf")
    best_f_yt = -float("inf")
    for i in range(s, e):
        fi = float(F_geom[i])
        if not np.isfinite(fi) or fi <= F_thr:
            continue
        di = float(D[i])
        if not np.isfinite(di):
            continue
        if di < best_d_yt or (di == best_d_yt and fi > best_f_yt):
            best_d_yt = di
            best_f_yt = fi
            i_yt = i

    i_yc: int | None = None
    best_d_yc = -float("inf")
    best_f_yc = float("inf")
    for i in range(s, e):
        fi = float(F_geom[i])
        if not np.isfinite(fi) or fi >= -F_thr:
            continue
        di = float(D[i])
        if not np.isfinite(di):
            continue
        if di > best_d_yc or (di == best_d_yc and fi < best_f_yc):
            best_d_yc = di
            best_f_yc = fi
            i_yc = i

    for slot in range(N_LANDMARK_SLOTS):
        if slot in LANDMARK_SLOTS_F0_EXTREMAL_D:
            continue
        if le[slot] is None:
            continue
        d_e, _ = le[slot]
        if not np.isfinite(d_e):
            continue
        s_path, e_path = _landmark_pair_path_window(
            slot, s, e, i_max, i_min, i_yt, i_yc, i0_exp=i0_exp
        )
        j = _nearest_vertex_index(D, float(d_e), s_path, e_path)
        if j is None:
            continue
        dj, fxe, fxs = float(D[j]), float(F_exp[j]), float(F_sim[j])
        if not (np.isfinite(dj) and np.isfinite(fxe) and np.isfinite(fxs)):
            continue
        le_metric[slot] = (dj, fxe)
        ls[slot] = (dj, fxs)
        j_slot[slot] = int(j)

    if le[6] is not None or le[7] is not None:
        f0 = _all_f_level_crossings(D, F_sim, s, e, 0.0)
        if f0:
            if le[6] is not None:
                ls[6] = max(f0, key=lambda p: p[0])
            if le[7] is not None:
                ls[7] = min(f0, key=lambda p: p[0])

    return ls, le_metric, j_slot


def _slot_error_force_sq(
    p_exp: tuple[float, float] | None,
    p_sim: tuple[float, float] | None,
    s_f: float,
) -> float | None:
    if p_exp is None or p_sim is None:
        return None
    _, fe = p_exp
    _, fs = p_sim
    inv_f = 1.0 / s_f if s_f > 0.0 and math.isfinite(s_f) else 1.0
    if not (np.isfinite(fe) and np.isfinite(fs)):
        return None
    return (fs - fe) ** 2 * inv_f**2


def _slot_error_disp_sq(
    p_exp: tuple[float, float] | None,
    p_sim: tuple[float, float] | None,
    s_d: float,
) -> float | None:
    if p_exp is None or p_sim is None:
        return None
    de, _ = p_exp
    ds, _ = p_sim
    inv_d = 1.0 / s_d if s_d > 0.0 and math.isfinite(s_d) else 1.0
    if not (np.isfinite(de) and np.isfinite(ds)):
        return None
    return (ds - de) ** 2 * inv_d**2


def _slot_error_combined_sq(
    p_exp: tuple[float, float] | None,
    p_sim: tuple[float, float] | None,
    s_f: float,
    s_d: float,
) -> float | None:
    """``((ΔF)/S_F)^2 + ((ΔD)/S_D)^2`` omitting any term whose inputs are non-finite."""
    if p_exp is None or p_sim is None:
        return None
    ef = _slot_error_force_sq(p_exp, p_sim, s_f)
    ed = _slot_error_disp_sq(p_exp, p_sim, s_d)
    if ef is None and ed is None:
        return None
    return float((ef or 0.0) + (ed or 0.0))
