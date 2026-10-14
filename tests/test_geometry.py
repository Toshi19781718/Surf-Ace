import numpy as np
from math import comb

from surf_ace.geometry import BSplineCurve, BSplineSurface
from surf_ace.geometry.basis import basis_matrix, clamped_uniform_knots


def bernstein(n, i, t):
    return comb(n, i) * t ** i * (1 - t) ** (n - i)


def test_single_span_is_bezier():
    t = np.linspace(0, 1, 11)
    B = basis_matrix(t, 5, clamped_uniform_knots(5, 6), 6)[0]
    ref = np.array([[bernstein(5, i, x) for i in range(6)] for x in t])
    assert np.allclose(B, ref)


def test_partition_of_unity_and_derivatives():
    knots = clamped_uniform_knots(3, 8)
    t = np.linspace(0, 1, 50)
    B = basis_matrix(t, 3, knots, 8, 2)
    assert np.allclose(B[0].sum(axis=1), 1.0)
    assert np.allclose(B[1].sum(axis=1), 0.0, atol=1e-9)
    assert np.allclose(B[2].sum(axis=1), 0.0, atol=1e-8)


def test_curve_derivative_matches_finite_difference():
    c = BSplineCurve(np.random.default_rng(0).normal(size=(7, 3)), 4)
    t, h = 0.37, 1e-6
    d = c.evaluate([t], 1)[1, 0]
    fd = (c.evaluate([t + h])[0, 0] - c.evaluate([t - h])[0, 0]) / (2 * h)
    assert np.allclose(d, fd, atol=1e-5)


def test_circle_like_curvature():
    # quarter arc approximation: curvature should be close to 1/R
    R = 50.0
    ang = np.linspace(0, np.pi / 2, 60)
    pts = np.stack([R * np.cos(ang), R * np.sin(ang), 0 * ang], axis=1)
    from surf_ace.geometry.fitting import lsq_fit_fixed_ends
    knots = clamped_uniform_knots(5, 10)
    c = BSplineCurve(lsq_fit_fixed_ends(pts, np.linspace(0, 1, 60), 5, knots), 5, knots)
    k, _ = c.curvature(np.linspace(0.1, 0.9, 9))
    assert np.allclose(k, 1 / R, rtol=1e-3)


def test_degree_elevation_is_exact():
    s = BSplineSurface.bezier_patch(degree=(3, 4), crown=12.0)
    s.ctrl[1, 2, 2] += 7.0
    e = s.rebuilt(6, 1, 7, 2)
    t = np.linspace(0, 1, 13)
    assert np.allclose(s.evaluate_grid(t, t)[(0, 0)], e.evaluate_grid(t, t)[(0, 0)], atol=1e-8)


def test_reorientation_preserves_surface():
    s = BSplineSurface.bezier_patch(degree=(3, 5), crown=8.0).rebuilt(3, 2, 5, 1)
    t = np.linspace(0, 1, 7)
    P = s.evaluate_grid(t, t)[(0, 0)]
    assert np.allclose(s.transposed().evaluate_grid(t, t)[(0, 0)], P.transpose(1, 0, 2))
    assert np.allclose(s.reversed_u().evaluate_grid(t, t)[(0, 0)], P[::-1])
    assert np.allclose(s.reversed_v().evaluate_grid(t, t)[(0, 0)], P[:, ::-1])


def test_sphere_like_curvatures():
    # paraboloid z = (x^2+y^2)/(2R): at apex K = 1/R^2, H = 1/R
    R = 100.0
    x = np.linspace(-10, 10, 3)
    X, Y = np.meshgrid(x, x, indexing="ij")
    # degree-2 Bezier control net that reproduces the paraboloid exactly
    Z = np.zeros_like(X)
    Z[[0, 2], :] += 100 / (2 * R)
    Z[:, [0, 2]] += 100 / (2 * R)
    Z[1, :] -= 100 / (2 * R)
    Z[:, 1] -= 100 / (2 * R)
    s = BSplineSurface(np.stack([X, Y, Z], -1), 2, 2)
    d = s.differential(np.array([0.5]), np.array([0.5]))
    assert np.isclose(abs(d["K"][0, 0]), 1 / R ** 2, rtol=1e-6)
    assert np.isclose(abs(d["H"][0, 0]), 1 / R, rtol=1e-6)
