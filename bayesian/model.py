import os
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

from params import *

from lib.brace_geometry import compute_Q
from lib.landmark_vector import read_landmark_cache_json, weighted_landmark_vector_model

# Fixed model (units: kip, inch, ksi). Hoisted so run_analysis avoids recomputing each call.
kips = 1.0
inch = 1.0
seconds = 1.0
feet = 12.0 * inch
lbs = kips / 1000.0
ksi = kips / inch**2
psi = ksi / 1000.0

A_r = 2.25 * inch**2
A_t = 5.625 * inch**2
r_T = 250 * inch
r_y = 175 * inch
KF = compute_Q(r_T, r_y, A_r, A_t)

E0 = 29000.0 * ksi
fyp = 38.6 * ksi
fyn = 38.6 * ksi
a2 = 1.0
a4 = 1.0
R0 = 20.0
dt = 1.0


def run_analysis(target_displacement=None):
    """
    Run OpenSees corotruss for the given displacement history.

    Parameters
    ----------
    target_displacement
        - ``None`` — load ``<cwd>/target_displacement.csv``.
        - ``str`` or path-like — CSV path (same layout as ``target_displacement.csv``).
        - Otherwise — treated as a 1-D array-like of control displacements (no disk I/O).
    """
    if target_displacement is None:
        tpath = os.path.join(os.getcwd(), "target_displacement.csv")
        d = np.loadtxt(tpath, delimiter=",", dtype=np.float64).ravel()
    elif isinstance(target_displacement, (str, os.PathLike)):
        d = np.loadtxt(os.fspath(target_displacement), delimiter=",", dtype=np.float64).ravel()
    else:
        d = np.asarray(target_displacement, dtype=np.float64, order="C").ravel()

    n = d.size
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Path series: uniform dt=1 => times 0..n; values length n+1 (start at 0 disp).
    path_values = np.empty(n + 1, dtype=np.float64)
    path_values[0] = 0.0
    path_values[1:] = d

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    ops.node(1, 0.0, 0.0)
    ops.fix(1, 1, 1)
    ops.node(2, r_T, 0.0)
    ops.fix(2, 0, 1)

    mat_tag = 1
    ops.uniaxialMaterial("SteelMPF", mat_tag, fyp, fyn, KF * E0, bp, bn, R0, cR1, cR2, a1, a2, a3, a4)

    ele_tag = 1
    ops.element("corotTruss", ele_tag, 1, 2, A_r, mat_tag)

    ops.timeSeries("Path", 1, "-dt", dt, "-values", *path_values, "-useLast")
    ops.pattern("Plain", 1, 1)
    ops.sp(2, 1, 1.0)

    ops.integrator("LoadControl", dt)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.analysis("Static", "-noWarnings")

    disp = np.empty(n, dtype=np.float64)
    force = np.empty(n, dtype=np.float64)
    ctrl_node = 2
    for i in range(n):
        ok = ops.analyze(1)
        if ok != 0:
            raise RuntimeError(
                f"OpenSees analyze failed at step {i + 1}/{n} "
                f"(target disp index {i}). ok={ok}"
            )
        disp[i] = ops.nodeDisp(ctrl_node, 1)
        axial = ops.eleResponse(ele_tag, "axialForce")
        force[i] = axial[0] if isinstance(axial, (list, tuple)) else float(axial)

    return disp, force


if __name__ == "__main__":
    cwd = Path(os.getcwd())
    D_sim, F_sim = run_analysis(None)

    results_path = cwd / "results.out"

    cache_path = cwd / "data" / "landmark_cache.json"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run: python scripts/precompute_landmark_cache.py "
            "(from the bayesian folder, after target_displacement/target_force are set)."
        )
    cache = read_landmark_cache_json(cache_path)
    vec = weighted_landmark_vector_model(D_sim, F_sim, cache)
    results_path.write_text(" ".join(str(v) for v in vec), encoding="utf-8")
