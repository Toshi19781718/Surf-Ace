"""Per-vertex surface analysis fields used by the viewer."""

from __future__ import annotations

import numpy as np

SHADED = "Shaded"
ZEBRA = "Zebra (reflection lines)"
ISOPHOTES = "Isophotes"
GAUSSIAN = "Gaussian curvature"
MEAN = "Mean curvature"
MAX_ABS = "Max |principal curvature|"

MODES = (SHADED, ZEBRA, ISOPHOTES, GAUSSIAN, MEAN, MAX_ABS)
STRIPE_MODES = (ZEBRA, ISOPHOTES)
CURVATURE_MODES = (GAUSSIAN, MEAN, MAX_ABS)


def zebra_phase(points, normals, cam_pos, cam_up, n_stripes: int) -> np.ndarray:
    """Continuous stripe phase for an environment of parallel light tubes fixed to the camera.
    Integer boundaries of the returned value are stripe edges."""
    v = points - np.asarray(cam_pos, dtype=float)
    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    vn = np.einsum("...k,...k", v, normals)[..., None]
    r = v - 2.0 * vn * normals
    up = np.asarray(cam_up, dtype=float)
    up = up / np.linalg.norm(up)
    return np.arcsin(np.clip(r @ up, -1.0, 1.0)) / (np.pi / n_stripes)


def isophote_phase(normals, light_dir, n_stripes: int) -> np.ndarray:
    d = np.asarray(light_dir, dtype=float)
    d = d / np.linalg.norm(d)
    cosang = np.abs(normals @ d)
    return np.arccos(np.clip(cosang, 0.0, 1.0)) / (0.5 * np.pi / n_stripes)


def curvature_field(diff: dict, mode: str) -> np.ndarray:
    if mode == GAUSSIAN:
        return diff["K"]
    if mode == MEAN:
        return diff["H"]
    if mode == MAX_ABS:
        return np.maximum(np.abs(diff["k1"]), np.abs(diff["k2"]))
    raise ValueError(mode)


def robust_range(values: np.ndarray, symmetric: bool, pct: float = 98.0) -> tuple[float, float]:
    vals = np.asarray(values)[np.isfinite(values)]
    if vals.size == 0:
        return -1.0, 1.0
    if symmetric:
        m = float(np.percentile(np.abs(vals), pct))
        m = m if m > 1e-12 else 1e-12
        return -m, m
    lo, hi = float(np.percentile(vals, 100 - pct)), float(np.percentile(vals, pct))
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    return lo, hi
