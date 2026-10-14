from __future__ import annotations

import numpy as np

from .basis import basis_matrix, clamped_uniform_knots, knots_from_spans, normalize_knots, span_count
from .fitting import refit_axis0


class BSplineCurve:
    """Non-rational clamped B-spline curve on [0, 1]. One span == Bezier curve."""

    kind = "curve"

    def __init__(self, ctrl, degree: int, knots=None, name: str = "Curve"):
        self.ctrl = np.array(ctrl, dtype=float).reshape(-1, 3)
        self.degree = int(degree)
        if knots is None:
            knots = clamped_uniform_knots(self.degree, len(self.ctrl))
        self.knots = normalize_knots(knots)
        self.name = name
        if len(self.knots) != len(self.ctrl) + self.degree + 1:
            raise ValueError("knot vector length does not match control points and degree")

    @classmethod
    def bezier(cls, ctrl, name: str = "Curve") -> "BSplineCurve":
        ctrl = np.asarray(ctrl, dtype=float)
        return cls(ctrl, len(ctrl) - 1, name=name)

    @property
    def n_ctrl(self) -> int:
        return len(self.ctrl)

    @property
    def spans(self) -> int:
        return span_count(self.knots, self.degree)

    def copy(self) -> "BSplineCurve":
        return BSplineCurve(self.ctrl.copy(), self.degree, self.knots.copy(), self.name)

    def evaluate(self, t, n_ders: int = 0) -> np.ndarray:
        """Return (n_ders+1, len(t), 3)."""
        B = basis_matrix(t, self.degree, self.knots, self.n_ctrl, n_ders)
        return np.einsum("dmi,ik->dmk", B, self.ctrl)

    def points(self, n: int = 200) -> np.ndarray:
        return self.evaluate(np.linspace(0.0, 1.0, n))[0]

    def curvature(self, t):
        """Return (curvature, unit principal normal pointing to the centre of curvature)."""
        d = self.evaluate(t, 2)
        c1, c2 = d[1], d[2]
        cross = np.cross(c1, c2)
        speed = np.linalg.norm(c1, axis=1)
        kappa = np.linalg.norm(cross, axis=1) / np.maximum(speed, 1e-12) ** 3
        nvec = np.cross(cross, c1)
        nlen = np.linalg.norm(nvec, axis=1, keepdims=True)
        normal = np.where(nlen > 1e-12, nvec / np.maximum(nlen, 1e-12), 0.0)
        return kappa, normal

    def comb(self, n: int, scale: float):
        """Curvature comb: base points and tooth tips (teeth point away from the centre)."""
        t = np.linspace(0.0, 1.0, n)
        base = self.evaluate(t)[0]
        kappa, normal = self.curvature(t)
        tips = base - normal * (kappa * scale)[:, None]
        return base, tips

    def closest_param(self, p, t0: float | None = None, iters: int = 20) -> float:
        p = np.asarray(p, dtype=float)
        if t0 is None:
            ts = np.linspace(0.0, 1.0, 200)
            t0 = ts[np.argmin(np.linalg.norm(self.evaluate(ts)[0] - p, axis=1))]
        t = float(t0)
        for _ in range(iters):
            d = self.evaluate([t], 2)[:, 0]
            diff = d[0] - p
            denom = d[1] @ d[1] + diff @ d[2]
            if abs(denom) < 1e-14:
                break
            step = (diff @ d[1]) / denom
            t = float(np.clip(t - step, 0.0, 1.0))
            if abs(step) < 1e-13:
                break
        return t

    def rebuilt(self, degree: int, spans: int) -> "BSplineCurve":
        knots = knots_from_spans(degree, spans)
        ctrl = refit_axis0(self.ctrl, self.degree, self.knots, degree, knots)
        return BSplineCurve(ctrl, degree, knots, self.name)

    def to_dict(self) -> dict:
        return {"type": "curve", "name": self.name, "degree": self.degree,
                "knots": self.knots.tolist(), "ctrl": self.ctrl.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "BSplineCurve":
        return cls(d["ctrl"], d["degree"], d["knots"], d.get("name", "Curve"))
