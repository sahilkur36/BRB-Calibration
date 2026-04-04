"""
Corotational truss (BRB) model using OpenSees with SteelMPF.

Returns force history given displacement history. Uses adjusted Young's modulus
E_hat = Q*E with Q from BRB geometry (non-yielding vs yielding length and area).
"""

from __future__ import annotations

import numpy as np

import openseespy.opensees as ops

from .brace_geometry import compute_Q


def run_simulation(
    displacement: np.ndarray,
    *,
    # Geometry (e.g. inches, in^2)
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    # SteelMPF: strengths (e.g. ksi)
    fyp: float,
    fyn: float,
    # SteelMPF: modulus and hardening (E in same units as stress, e.g. ksi)
    E: float,
    b_p: float,
    b_n: float,
    # SteelMPF: curvature (R0, cR1, cR2); optional isotropic hardening (a1-a4)
    R0: float = 20.0,
    cR1: float = 0.925,
    cR2: float = 0.15,
    a1: float = 0.0,
    a2: float = 1.0,
    a3: float = 0.0,
    a4: float = 1.0,
) -> np.ndarray:
    """
    Run the BRB corotruss simulation and return force history for the given displacement history.

    Uses a 2D corotational truss of length L_T with SteelMPF. Young's modulus is adjusted to
    represent the full brace with E_hat = Q*E, where
    Q = 1 / ( (2(L_T - L_y)/L_T)/(A_t/A_sc) + L_y/L_T ).

    Parameters
    ----------
    displacement : np.ndarray
        Displacement history (e.g. in); same units as implied by E, fyp, fyn.
    L_T : float
        Total length of the truss/brace.
    L_y : float
        Length of the yielding (core) region.
    A_sc : float
        Steel core (yielding) area.
    A_t : float
        Transition / non-yielding area (used in Q).
    fyp, fyn : float
        Yield strength in tension and compression (SteelMPF).
    E : float
        Young's modulus of the steel; E_hat = Q*E is used in the material.
    b_p, b_n : float
        Strain hardening ratio in tension and compression (SteelMPF; OpenSees bp, bn).
    R0, cR1, cR2 : float
        SteelMPF curvature parameters (defaults: 20, 0.925, 0.15).
    a1, a2, a3, a4 : float
        SteelMPF isotropic hardening (defaults: 0, 1, 0, 1 => no isotropic hardening).

    Returns
    -------
    np.ndarray
        Axial force history, same length as `displacement` (from corotTruss element).
    """
    displacement = np.asarray(displacement, dtype=float)
    n = len(displacement)

    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    ops.node(1, 0.0, 0.0)
    ops.fix(1, 1, 1)
    ops.node(2, L_T, 0.0)
    ops.fix(2, 0, 1)  # fix y only; x prescribed by pattern

    # SteelMPF: matTag, fyp, fyn, E0, bp, bn, R0, cR1, cR2, a1, a2, a3, a4
    ops.uniaxialMaterial(
        "SteelMPF", 1, fyp, fyn, E_hat, b_p, b_n,
        R0, cR1, cR2,
        a1, a2, a3, a4,
    )

    ops.element("corotTruss", 1, 1, 2, A_sc, 1)

    # Path series: uniform dt=1 => times 0..n; length n+1 with zero initial disp.
    # -useLast avoids load factor 0 when pseudo-time rounds past the final sample.
    dt = 1.0
    path_values = np.empty(n + 1, dtype=np.float64)
    path_values[0] = 0.0
    path_values[1:] = displacement
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *path_values, "-useLast")
    ops.pattern("Plain", 1, 1)
    ops.sp(2, 1, 1.0)

    ops.integrator("LoadControl", dt)
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("UmfPack")
    ops.analysis("Static", "-noWarnings")

    force = np.zeros(n)
    for i in range(n):
        ok = ops.analyze(1)
        if ok != 0:
            raise RuntimeError(
                f"OpenSees analyze failed at step {i + 1}/{n}. "
                "Try smaller displacement increments or check material/geometry."
            )
        axial_force = ops.eleResponse(1, "axialForce")
        force[i] = axial_force[0] if isinstance(axial_force, (list, tuple)) else axial_force

    return force
