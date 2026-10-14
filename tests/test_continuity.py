import numpy as np
import pytest

from surf_ace.continuity import closest_edge_pair, continuity_report, match_continuity
from surf_ace.geometry import BSplineSurface


def make_pair(rng=0):
    r = np.random.default_rng(rng)
    a = BSplineSurface.bezier_patch(degree=(5, 5), crown=15.0, name="A")
    a.ctrl[..., 2] += r.normal(scale=2.0, size=a.ctrl.shape[:2])
    b = BSplineSurface.bezier_patch(origin=(101.0, 0, 3.0), degree=(6, 5), crown=-5.0, name="B")
    b.ctrl[..., 2] += r.normal(scale=2.0, size=b.ctrl.shape[:2])
    return a, b


@pytest.mark.parametrize("level,expected", [("G0", "G0"), ("G1", "G1"), ("G2", "G2")])
def test_match_reaches_level(level, expected):
    a, b = make_pair()
    before = continuity_report(a, "u1", b, "u0")
    assert before.achieved == "none"
    b2, _ = match_continuity(a, "u1", b, "u0", level)
    rep = continuity_report(a, "u1", b2, "u0")
    order = ["none", "G0", "G1", "G2"]
    assert order.index(rep.achieved) >= order.index(expected), rep.summary()


def test_g3_matches_third_derivative():
    a, b = make_pair(1)
    b2, _ = match_continuity(a, "u1", b, "u0", "G3")
    rep = continuity_report(a, "u1", b2, "u0")
    assert rep.achieved == "G2"
    da = a.evaluate_grid([1.0], [0.3], 3)
    db = b2.evaluate_grid([0.0], [0.3], 3)
    alpha = np.linalg.norm(db[(1, 0)]) / np.linalg.norm(da[(1, 0)])
    assert np.allclose(db[(3, 0)], alpha ** 3 * da[(3, 0)], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("ea,eb", [("u1", "v1"), ("v0", "u0"), ("v1", "v0"), ("u0", "u1")])
def test_any_edge_orientation(ea, eb):
    # Build B adjacent to A's edge `ea` with its own edge `eb` and reversed direction.
    a = BSplineSurface.bezier_patch(degree=(5, 5), crown=10.0, name="A")
    edge = a.edge_curve(ea)
    p0, p1 = edge.ctrl[0], edge.ctrl[-1]
    centre = a.ctrl.reshape(-1, 3).mean(0)
    mid = 0.5 * (p0 + p1)
    out = mid - centre
    out[2] = 0
    out /= np.linalg.norm(out)
    rows = [np.linspace(p1, p0, 6) + out * (k * 20.0) + np.array([0, 0, 2.0 + k]) for k in range(6)]
    b = BSplineSurface(np.array(rows), 5, 5, name="B")
    b = {"u0": b, "u1": b.reversed_u(), "v0": b.transposed(), "v1": b.transposed().reversed_v()}[eb]
    b2, _ = match_continuity(a, ea, b, eb, "G2")
    assert continuity_report(a, ea, b2, eb).achieved == "G2"


def test_closest_edge_pair():
    a, b = make_pair()
    assert closest_edge_pair(a, b) == ("u1", "u0")
