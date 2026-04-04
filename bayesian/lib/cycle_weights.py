"""Cycle partitions and amplitude weights (numpy only)."""
from __future__ import annotations

import warnings

import numpy as np


def w_from_amplitude(
    A_c: float,
    A_max: float,
    *,
    p: float,
    eps: float,
) -> float:
    if not np.isfinite(A_max) or A_max <= 0.0:
        A_max = 1.0
    if not np.isfinite(A_c) or A_c < 0.0:
        A_c = 0.0
    return float((A_c / A_max) ** p + eps)


def zero_def_indices_from_points(points: list[dict], n: int) -> list[int]:
    z: list[int] = []
    for p in points:
        idx = p.get("idx")
        if idx is None:
            continue
        ptype = str(p.get("type", ""))
        if "zero_def" in ptype:
            i = int(idx)
            if 0 <= i < n:
                z.append(i)
    return sorted(set(z))


def verify_partition_coverage(ranges: list[dict], n: int) -> bool:
    if n <= 0:
        return True
    cov = np.zeros(n, dtype=np.int32)
    for r in ranges:
        s, e = int(r["start"]), int(r["end"])
        if s < 0 or e > n or s >= e:
            return False
        cov[s:e] += 1
    return bool(np.all(cov == 1))


def build_cycle_weight_ranges(
    n: int,
    points: list[dict],
    *,
    debug_partition: bool = False,
) -> list[dict]:
    if n <= 0:
        return []

    z = zero_def_indices_from_points(points, n)
    ranges: list[dict] = []

    if not z:
        ranges.append({"start": 0, "end": n, "kind": "incomplete", "incomplete": True})
    else:
        if z[0] > 0:
            ranges.append(
                {
                    "start": 0,
                    "end": z[0],
                    "kind": "incomplete_head",
                    "incomplete": True,
                }
            )
        i = 0
        while i + 2 < len(z):
            ranges.append(
                {
                    "start": z[i],
                    "end": z[i + 2],
                    "kind": "full_cycle",
                    "incomplete": False,
                }
            )
            i += 2
        if i < len(z):
            ranges.append(
                {
                    "start": z[i],
                    "end": n,
                    "kind": "incomplete_tail",
                    "incomplete": True,
                }
            )

    ranges = [r for r in ranges if r["end"] > r["start"]]

    if not ranges:
        ranges.append({"start": 0, "end": n, "kind": "incomplete", "incomplete": True})

    if debug_partition:
        assert verify_partition_coverage(ranges, n), (ranges, n)

    return ranges


def build_amplitude_weights(
    deformation: np.ndarray,
    points: list[dict],
    *,
    p: float = 2.0,
    eps: float = 0.05,
    debug_partition: bool = False,
    use_amplitude_weights: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    u = np.asarray(deformation, dtype=float)
    n = len(u)
    if n == 0:
        return np.array([], dtype=float), []

    ranges = build_cycle_weight_ranges(n, points, debug_partition=debug_partition)

    if debug_partition and not verify_partition_coverage(ranges, n):
        raise AssertionError(f"partition failed coverage check: {ranges!r} n={n}")

    amps: list[float] = []
    for r in ranges:
        s, e = int(r["start"]), int(r["end"])
        if e <= s:
            amps.append(0.0)
        else:
            amps.append(float(np.max(np.abs(u[s:e]))))

    A_max = max(amps) if amps else 0.0
    if A_max <= 0.0 or not np.isfinite(A_max):
        A_max = 1.0

    weights = np.ones(n, dtype=float)
    meta: list[dict] = []

    for r, A_c in zip(ranges, amps):
        s, e = int(r["start"]), int(r["end"])
        if e <= s:
            continue
        if use_amplitude_weights:
            w_c = w_from_amplitude(A_c, A_max, p=p, eps=eps)
        else:
            w_c = 1.0
        weights[s:e] = w_c
        rr = dict(r)
        rr["amp"] = A_c
        rr["w_c"] = w_c
        meta.append(rr)

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        warnings.warn(
            "build_amplitude_weights: non-finite or non-positive weights; using ones.",
            UserWarning,
            stacklevel=2,
        )
        weights = np.ones(n, dtype=float)

    return weights, meta
