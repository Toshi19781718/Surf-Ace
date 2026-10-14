from __future__ import annotations

import numpy as np

from .basis import basis_matrix, clamped_uniform_knots, knots_from_spans, normalize_knots, span_count
from .curve import BSplineCurve
from .fitting import refit_axis0

EDGES = ("u0", "u1", "v0", "v1")


class BSplineSurface:
    """Non-rational clamped tensor-product B-spline surface on [0,1]^2.

    ctrl has shape (nu, nv, 3); index i runs along u, j along v.
    A single span in both directions is a Bezier patch.
    """

    kind = "surface"

    def __init__(self, ctrl, deg_u: int, deg_v: int, knots_u=None, knots_v=None, name: str = "Patch"):
        self.ctrl = np.array(ctrl, dtype=float)
        if self.ctrl.ndim != 3 or self.ctrl.shape[2] != 3:
            raise ValueError("ctrl must have shape (nu, nv, 3)")
        self.deg_u = int(deg_u)
        self.deg_v = int(deg_v)
        nu, nv = self.ctrl.shape[:2]
        self.knots_u = normalize_knots(clamped_uniform_knots(self.deg_u, nu) if knots_u is None else knots_u)
        self.knots_v = normalize_knots(clamped_uniform_knots(self.deg_v, nv) if knots_v is None else knots_v)
        if len(self.knots_u) != nu + self.deg_u + 1 or len(self.knots_v) != nv + self.deg_v + 1:
            raise ValueError("knot vectors do not match control net and degrees")
        self.name = name

    @classmethod
    def bezier_patch(cls, origin=(0, 0, 0), size=(100.0, 100.0), degree=(5, 5),
                     crown=0.0, name: str = "Patch") -> "BSplineSurface":
        du, dv = degree
        x = np.linspace(0.0, size[0], du + 1)
        y = np.linspace(0.0, size[1], dv + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")
        sx = 2 * X / size[0] - 1
        sy = 2 * Y / size[1] - 1
        Z = crown * (1 - sx ** 2) * (1 - 0.5 * sy ** 2)
        ctrl = np.stack([X, Y, Z], axis=-1) + np.asarray(origin, dtype=float)
        return cls(ctrl, du, dv, name=name)

    @property
    def nu(self) -> int:
        return self.ctrl.shape[0]

    @property
    def nv(self) -> int:
        return self.ctrl.shape[1]

    @property
    def spans_u(self) -> int:
        return span_count(self.knots_u, self.deg_u)

    @property
    def spans_v(self) -> int:
        return span_count(self.knots_v, self.deg_v)

    def copy(self) -> "BSplineSurface":
        return BSplineSurface(self.ctrl.copy(), self.deg_u, self.deg_v,
                              self.knots_u.copy(), self.knots_v.copy(), self.name)

    # ------------------------------------------------------------------ evaluation
    def evaluate_grid(self, us, vs, n_ders: int = 0) -> dict:
        """Partial derivatives on a grid: result[(a, b)] = d^(a+b) S / du^a dv^b, shape (mu, mv, 3)."""
        Bu = basis_matrix(us, self.deg_u, self.knots_u, self.nu, n_ders)
        Bv = basis_matrix(vs, self.deg_v, self.knots_v, self.nv, n_ders)
        out = {}
        for a in range(n_ders + 1):
            tmp = np.einsum("mi,ijk->mjk", Bu[a], self.ctrl)
            for b in range(n_ders + 1 - a):
                out[(a, b)] = np.einsum("mjk,nj->mnk", tmp, Bv[b])
        return out

    def differential(self, us, vs) -> dict:
        """Positions, unit normals, fundamental forms and curvatures on a grid."""
        d = self.evaluate_grid(us, vs, 2)
        Su, Sv = d[(1, 0)], d[(0, 1)]
        n = np.cross(Su, Sv)
        nlen = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.maximum(nlen, 1e-14)
        E = np.einsum("...k,...k", Su, Su)
        F = np.einsum("...k,...k", Su, Sv)
        G = np.einsum("...k,...k", Sv, Sv)
        L = np.einsum("...k,...k", d[(2, 0)], n)
        M = np.einsum("...k,...k", d[(1, 1)], n)
        N = np.einsum("...k,...k", d[(0, 2)], n)
        det = np.maximum(E * G - F * F, 1e-20)
        K = (L * N - M * M) / det
        H = (E * N - 2 * F * M + G * L) / (2 * det)
        disc = np.sqrt(np.maximum(H * H - K, 0.0))
        return {"S": d[(0, 0)], "Su": Su, "Sv": Sv, "N": n,
                "E": E, "F": F, "G": G, "L": L, "M": M, "Nn": N,
                "K": K, "H": H, "k1": H + disc, "k2": H - disc}

    # ------------------------------------------------------------------ curves
    def iso_u(self, u: float) -> BSplineCurve:
        """Iso-parametric curve at fixed u (runs along v)."""
        Bu = basis_matrix([u], self.deg_u, self.knots_u, self.nu)[0, 0]
        return BSplineCurve(np.einsum("i,ijk->jk", Bu, self.ctrl), self.deg_v, self.knots_v.copy())

    def iso_v(self, v: float) -> BSplineCurve:
        """Iso-parametric curve at fixed v (runs along u)."""
        Bv = basis_matrix([v], self.deg_v, self.knots_v, self.nv)[0, 0]
        return BSplineCurve(np.einsum("j,ijk->ik", Bv, self.ctrl), self.deg_u, self.knots_u.copy())

    def edge_curve(self, edge: str) -> BSplineCurve:
        return {"u0": lambda: self.iso_u(0.0), "u1": lambda: self.iso_u(1.0),
                "v0": lambda: self.iso_v(0.0), "v1": lambda: self.iso_v(1.0)}[edge]()

    # ------------------------------------------------------------------ re-orientation
    def transposed(self) -> "BSplineSurface":
        return BSplineSurface(self.ctrl.transpose(1, 0, 2).copy(), self.deg_v, self.deg_u,
                              self.knots_v.copy(), self.knots_u.copy(), self.name)

    def reversed_u(self) -> "BSplineSurface":
        return BSplineSurface(self.ctrl[::-1].copy(), self.deg_u, self.deg_v,
                              1.0 - self.knots_u[::-1], self.knots_v.copy(), self.name)

    def reversed_v(self) -> "BSplineSurface":
        return BSplineSurface(self.ctrl[:, ::-1].copy(), self.deg_u, self.deg_v,
                              self.knots_u.copy(), 1.0 - self.knots_v[::-1], self.name)

    # ------------------------------------------------------------------ restructuring
    def with_u_structure(self, degree: int, knots) -> "BSplineSurface":
        knots = np.asarray(knots, dtype=float)
        ctrl = refit_axis0(self.ctrl, self.deg_u, self.knots_u, degree, knots)
        return BSplineSurface(ctrl, degree, self.deg_v, knots, self.knots_v.copy(), self.name)

    def with_v_structure(self, degree: int, knots) -> "BSplineSurface":
        return self.transposed().with_u_structure(degree, knots).transposed()

    def rebuilt(self, deg_u: int, spans_u: int, deg_v: int, spans_v: int) -> "BSplineSurface":
        s = self
        if deg_u != s.deg_u or spans_u != s.spans_u:
            s = s.with_u_structure(deg_u, knots_from_spans(deg_u, spans_u))
        if deg_v != s.deg_v or spans_v != s.spans_v:
            s = s.with_v_structure(deg_v, knots_from_spans(deg_v, spans_v))
        return s.copy() if s is self else s

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        return {"type": "surface", "name": self.name, "deg_u": self.deg_u, "deg_v": self.deg_v,
                "knots_u": self.knots_u.tolist(), "knots_v": self.knots_v.tolist(),
                "ctrl": self.ctrl.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "BSplineSurface":
        return cls(d["ctrl"], d["deg_u"], d["deg_v"], d["knots_u"], d["knots_v"], d.get("name", "Patch"))
