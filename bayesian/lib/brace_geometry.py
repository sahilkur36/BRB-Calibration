"""Brace geometry helpers (no OpenSees dependency)."""
from __future__ import annotations


def compute_Q(L_T: float, L_y: float, A_sc: float, A_t: float) -> float:
    """
    Stiffness adjustment factor for full brace: E_hat = Q*E.

    Q = 1 / ( (2(L_T - L_y)/L_T) / (A_t/A_sc) + L_y/L_T )
    """
    if L_T <= 0 or A_sc <= 0 or A_t <= 0:
        raise ValueError("L_T, A_sc, and A_t must be positive.")
    if L_y < 0 or L_y > L_T:
        raise ValueError("L_y must be in [0, L_T].")
    term_non_yielding = (2.0 * (L_T - L_y) / L_T) / (A_t / A_sc)
    term_yielding = L_y / L_T
    return 1.0 / (term_non_yielding + term_yielding)
