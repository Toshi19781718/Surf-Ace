"""Least-squares refitting used to change degree / span count of curves and nets."""

from __future__ import annotations

import numpy as np

from .basis import basis_matrix


def lsq_fit_fixed_ends(samples: np.ndarray, params: np.ndarray, degree: int, knots: np.ndarray) -> np.ndarray:
    """Fit control points along axis 0 of `samples`. params must start at 0 and end at 1
    so the end samples are interpolated exactly."""
    samples = np.asarray(samples, dtype=float)
    m = samples.shape[0]
    n = len(knots) - degree - 1
    N = basis_matrix(params, degree, knots, n)[0]
    flat = samples.reshape(m, -1)
    ctrl = np.zeros((n, flat.shape[1]))
    ctrl[0] = flat[0]
    ctrl[-1] = flat[-1]
    if n > 2:
        rhs = flat - np.outer(N[:, 0], ctrl[0]) - np.outer(N[:, -1], ctrl[-1])
        sol, *_ = np.linalg.lstsq(N[:, 1:-1], rhs, rcond=None)
        ctrl[1:-1] = sol
    return ctrl.reshape((n,) + samples.shape[1:])


def refit_axis0(ctrl: np.ndarray, old_degree: int, old_knots: np.ndarray,
                new_degree: int, new_knots: np.ndarray) -> np.ndarray:
    """Re-approximate the spline(s) defined along axis 0 of `ctrl` with a new structure.
    Exact whenever the new space contains the old one (e.g. degree elevation)."""
    n_old = ctrl.shape[0]
    n_new = len(new_knots) - new_degree - 1
    m = max(8 * n_new, 4 * n_old, 64)
    params = np.linspace(0.0, 1.0, m)
    B = basis_matrix(params, old_degree, old_knots, n_old)[0]
    samples = np.tensordot(B, ctrl, axes=(1, 0))
    return lsq_fit_fixed_ends(samples, params, new_degree, new_knots)
