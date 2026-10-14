"""Edge continuity matching (G0..G3) and continuity measurement between patches.

Matching works on a canonical orientation: the reference patch A is re-oriented so the
shared edge sits at u=1, the patch to modify B so its edge sits at u=0, both running
along v in the same direction. Along-edge structures are made identical, then the first
k+1 control rows of B are solved so that its cross-boundary derivatives 0..k equal
alpha^d times A's. A single alpha (taken from B's current cross-boundary speed) keeps
B's tension while giving geometric continuity of order k.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry.basis import ders_basis_funs, find_span, knots_from_spans
from .geometry.surface import EDGES, BSplineSurface

LEVELS = {"G0": 0, "G1": 1, "G2": 2, "G3": 3}

_OPS_A = {"u1": [], "u0": ["ru"], "v1": ["t"], "v0": ["t", "ru"]}
_OPS_B = {"u0": [], "u1": ["ru"], "v0": ["t"], "v1": ["t", "ru"]}


def _apply(s: BSplineSurface, ops) -> BSplineSurface:
    for op in ops:
        s = s.transposed() if op == "t" else s.reversed_u() if op == "ru" else s.reversed_v()
    return s


def _undo(s: BSplineSurface, ops) -> BSplineSurface:
    return _apply(s, list(reversed(ops)))


def _needs_flip(a: BSplineSurface, b: BSplineSurface) -> bool:
    a0, a1 = a.ctrl[-1, 0], a.ctrl[-1, -1]
    b0, b1 = b.ctrl[0, 0], b.ctrl[0, -1]
    same = np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1)
    flip = np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)
    return flip < same


def _canonical_pair(A, edge_a, B, edge_b):
    if edge_a not in EDGES or edge_b not in EDGES:
        raise ValueError(f"edges must be one of {EDGES}")
    a = _apply(A.copy(), _OPS_A[edge_a])
    b = _apply(B.copy(), _OPS_B[edge_b])
    flip = _needs_flip(a, b)
    if flip:
        b = b.reversed_v()
    return a, b, flip


def _boundary_ders(s: BSplineSurface, k: int, at_end: bool) -> np.ndarray:
    """Cross-boundary (u) derivative control curves 0..k at u=0 or u=1: shape (k+1, nv, 3)."""
    u = 1.0 if at_end else 0.0
    span = find_span(s.nu, s.deg_u, u, s.knots_u)
    D = ders_basis_funs(span, u, s.deg_u, k, s.knots_u)
    rows = s.ctrl[span - s.deg_u:span + 1]
    return np.einsum("di,ijk->djk", D, rows)


def closest_edge_pair(A: BSplineSurface, B: BSplineSurface):
    """Pick the edge pair whose end points are closest (useful to auto-fill the UI)."""
    best = None
    for ea in EDGES:
        ca = A.edge_curve(ea).ctrl[[0, -1]]
        for eb in EDGES:
            cb = B.edge_curve(eb).ctrl[[0, -1]]
            d = min(np.linalg.norm(ca - cb, axis=1).sum(), np.linalg.norm(ca - cb[::-1], axis=1).sum())
            if best is None or d < best[0]:
                best = (d, ea, eb)
    return best[1], best[2]


def match_continuity(A: BSplineSurface, edge_a: str, B: BSplineSurface, edge_b: str, level: str):
    """Return (modified copy of B, list of messages). A is never modified."""
    k = LEVELS[level]
    msgs: list[str] = []
    a, b, flip = _canonical_pair(A, edge_a, B, edge_b)

    if b.deg_v != a.deg_v or not np.allclose(b.knots_v, a.knots_v):
        b = b.with_v_structure(a.deg_v, a.knots_v)
        msgs.append(f"Rebuilt '{B.name}' along the edge to degree {a.deg_v}, "
                    f"{a.spans_v} span(s) to match '{A.name}'.")
    if b.deg_u < k:
        b = b.with_u_structure(k, knots_from_spans(k, b.spans_u))
        msgs.append(f"Raised '{B.name}' cross-edge degree to {k} (needed for {level}).")
    if a.deg_u < k:
        msgs.append(f"'{A.name}' has cross-edge degree {a.deg_u} < {k}; its higher derivatives are zero.")
    if b.nu < 2 * (k + 1):
        msgs.append(f"'{B.name}' has only {b.nu} rows across the edge; {level} matching also moves "
                    "rows that influence the opposite edge. Add degree or spans for independence.")

    target = _boundary_ders(a, k, at_end=True)
    if k >= 1:
        current = _boundary_ders(b, 1, at_end=False)[1]
        la = np.linalg.norm(target[1], axis=-1).mean()
        lb = np.linalg.norm(current, axis=-1).mean()
        alpha = lb / la if la > 1e-12 and lb > 1e-12 else 1.0
        target = target * (alpha ** np.arange(k + 1))[:, None, None]

    span = find_span(b.nu, b.deg_u, 0.0, b.knots_u)
    D = ders_basis_funs(span, 0.0, b.deg_u, k, b.knots_u)[:, :k + 1]
    rows = np.linalg.solve(D, target.reshape(k + 1, -1)).reshape(k + 1, b.nv, 3)
    b.ctrl[:k + 1] = rows

    if flip:
        b = b.reversed_v()
    b = _undo(b, _OPS_B[edge_b])
    b.name = B.name
    return b, msgs


@dataclass
class ContinuityReport:
    max_gap: float
    max_angle_deg: float
    max_curv_diff: float
    max_curv_rel: float
    achieved: str
    params: np.ndarray = field(repr=False)
    gap: np.ndarray = field(repr=False)
    angle_deg: np.ndarray = field(repr=False)
    curv_a: np.ndarray = field(repr=False)
    curv_b: np.ndarray = field(repr=False)

    def summary(self) -> str:
        return (f"Achieved: {self.achieved}\n"
                f"  Max positional gap:      {self.max_gap:.6f}\n"
                f"  Max normal deviation:    {self.max_angle_deg:.5f} deg\n"
                f"  Max cross curvature diff: {self.max_curv_diff:.3e} "
                f"({100 * self.max_curv_rel:.3f} %)")


def _cross_normal_curvature(d: dict, idx) -> tuple[float, np.ndarray]:
    Su, Sv, n = d["Su"][idx], d["Sv"][idx], d["N"][idx]
    e = Sv / max(np.linalg.norm(Sv), 1e-14)
    c = np.cross(n, e)
    E, F, G = d["E"][idx], d["F"][idx], d["G"][idx]
    a, b = np.linalg.solve(np.array([[E, F], [F, G]]), np.array([c @ Su, c @ Sv]))
    num = d["L"][idx] * a * a + 2 * d["M"][idx] * a * b + d["Nn"][idx] * b * b
    den = E * a * a + 2 * F * a * b + G * b * b
    return num / max(den, 1e-20), n


def continuity_report(A: BSplineSurface, edge_a: str, B: BSplineSurface, edge_b: str,
                      samples: int = 41, tol_pos: float = 0.005, tol_angle: float = 0.05,
                      tol_curv: float = 0.02) -> ContinuityReport:
    a, b, _ = _canonical_pair(A, edge_a, B, edge_b)
    t = np.linspace(0.0, 1.0, samples)
    da = a.differential(np.array([1.0]), t)
    edge_b_curve = b.iso_u(0.0)
    gap, ang, ka, kb = (np.zeros(samples) for _ in range(4))
    s_prev = None
    for i, ti in enumerate(t):
        pa = da["S"][0, i]
        s = edge_b_curve.closest_param(pa, s_prev)
        s_prev = s
        db = b.differential(np.array([0.0]), np.array([s]))
        gap[i] = np.linalg.norm(db["S"][0, 0] - pa)
        kA, nA = _cross_normal_curvature(da, (0, i))
        kB, nB = _cross_normal_curvature(db, (0, 0))
        dot = nA @ nB
        ang[i] = np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0)))
        ka[i] = kA
        kb[i] = kB if dot >= 0 else -kB
    diff = np.abs(ka - kb)
    rel = diff / np.maximum(np.maximum(np.abs(ka), np.abs(kb)), 1e-9)
    rel = np.where(diff < 1e-7, 0.0, rel)
    achieved = "none"
    if gap.max() <= tol_pos:
        achieved = "G0"
        if ang.max() <= tol_angle:
            achieved = "G1"
            if rel.max() <= tol_curv:
                achieved = "G2"
    return ContinuityReport(float(gap.max()), float(ang.max()), float(diff.max()), float(rel.max()),
                            achieved, t, gap, ang, ka, kb)
