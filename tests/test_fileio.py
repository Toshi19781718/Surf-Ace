import numpy as np
import pytest

from surf_ace import fileio
from surf_ace.geometry import BSplineCurve, BSplineSurface


def sample_objects():
    s = BSplineSurface.bezier_patch(degree=(5, 4), crown=9.0, name="P").rebuilt(5, 2, 4, 1)
    s.ctrl[2, 2, 2] += 3.0
    c = BSplineCurve.bezier([[0, 0, 0], [10, 5, 0], [20, -5, 3], [30, 0, 0]], name="C")
    return [s, c]


def test_session_roundtrip(tmp_path):
    objs = sample_objects()
    fileio.save_session(tmp_path / "s.json", objs)
    back = fileio.load_session(tmp_path / "s.json")
    assert np.allclose(back[0].ctrl, objs[0].ctrl) and back[0].spans_u == 2
    assert np.allclose(back[1].ctrl, objs[1].ctrl)


def test_obj_export(tmp_path):
    fileio.export_obj(tmp_path / "m.obj", [sample_objects()[0]], resolution=10)
    text = (tmp_path / "m.obj").read_text()
    assert text.count("\nv ") == 100 and text.count("\nf ") == 81


@pytest.mark.skipif(not fileio.occ_available(), reason="pythonOCC not installed")
@pytest.mark.parametrize("ext", ["step", "igs"])
def test_cad_roundtrip(tmp_path, ext):
    s = sample_objects()[0]
    path = tmp_path / f"m.{ext}"
    (fileio.export_step if ext == "step" else fileio.export_iges)(path, sample_objects())
    surfaces, notes = fileio.import_cad(path)
    assert len(surfaces) == 1, notes
    t = np.linspace(0, 1, 9)
    a = s.evaluate_grid(t, t)[(0, 0)]
    b = surfaces[0].evaluate_grid(t, t)[(0, 0)]
    assert np.allclose(a, b, atol=1e-6)
