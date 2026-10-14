"""B-spline basis functions (algorithms follow Piegl & Tiller, "The NURBS Book")."""

from __future__ import annotations

import numpy as np


def clamped_uniform_knots(degree: int, n_ctrl: int) -> np.ndarray:
    if n_ctrl < degree + 1:
        raise ValueError(f"need at least {degree + 1} control points for degree {degree}")
    n_inner = n_ctrl - degree - 1
    inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
    return np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])


def knots_from_spans(degree: int, spans: int) -> np.ndarray:
    return clamped_uniform_knots(degree, degree + spans)


def span_count(knots: np.ndarray, degree: int) -> int:
    return len(np.unique(knots[degree:len(knots) - degree])) - 1


def normalize_knots(knots: np.ndarray) -> np.ndarray:
    knots = np.asarray(knots, dtype=float)
    lo, hi = knots[0], knots[-1]
    if hi - lo <= 0:
        raise ValueError("degenerate knot vector")
    return (knots - lo) / (hi - lo)


def find_span(n_ctrl: int, degree: int, u: float, knots: np.ndarray) -> int:
    n = n_ctrl - 1
    if u >= knots[n + 1]:
        return n
    if u <= knots[degree]:
        return degree
    return int(np.searchsorted(knots, u, side="right") - 1)


def ders_basis_funs(span: int, u: float, degree: int, n_ders: int, knots: np.ndarray) -> np.ndarray:
    """Return array (n_ders+1, degree+1): derivatives of the non-zero basis functions
    N_{span-degree..span} at u. Rows beyond `degree` are zero."""
    p = degree
    ndu = np.zeros((p + 1, p + 1))
    ndu[0, 0] = 1.0
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    ders = np.zeros((n_ders + 1, p + 1))
    ders[0, :] = ndu[:, p]
    n_eff = min(n_ders, p)
    a = np.zeros((2, p + 1))
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, n_eff + 1):
            d = 0.0
            rk, pk = r - k, p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
    factor = p
    for k in range(1, n_eff + 1):
        ders[k, :] *= factor
        factor *= p - k
    return ders


def basis_matrix(params, degree: int, knots: np.ndarray, n_ctrl: int, n_ders: int = 0) -> np.ndarray:
    """Dense basis matrix of shape (n_ders+1, len(params), n_ctrl)."""
    params = np.atleast_1d(np.asarray(params, dtype=float))
    out = np.zeros((n_ders + 1, len(params), n_ctrl))
    for m, u in enumerate(params):
        span = find_span(n_ctrl, degree, u, knots)
        out[:, m, span - degree:span + 1] = ders_basis_funs(span, u, degree, n_ders, knots)
    return out
