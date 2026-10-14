"""Session (JSON), OBJ export and STEP/IGES exchange via pythonOCC (OpenCASCADE)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry.basis import knots_from_spans
from .geometry.curve import BSplineCurve
from .geometry.surface import BSplineSurface

SESSION_VERSION = 1


# ---------------------------------------------------------------------- session
def save_session(path, objects) -> None:
    data = {"version": SESSION_VERSION, "objects": [o.to_dict() for o in objects]}
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_session(path) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for d in data.get("objects", []):
        out.append(BSplineSurface.from_dict(d) if d["type"] == "surface" else BSplineCurve.from_dict(d))
    return out


# ---------------------------------------------------------------------- OBJ
def export_obj(path, surfaces, resolution: int = 64) -> None:
    lines = ["# Surf-Ace OBJ export"]
    offset = 1
    t = np.linspace(0.0, 1.0, resolution)
    for s in surfaces:
        d = s.differential(t, t)
        P = d["S"].reshape(-1, 3)
        N = d["N"].reshape(-1, 3)
        lines.append(f"o {s.name}")
        lines += [f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" for p in P]
        lines += [f"vn {n[0]:.6g} {n[1]:.6g} {n[2]:.6g}" for n in N]
        r = resolution
        for i in range(r - 1):
            for j in range(r - 1):
                a = offset + i * r + j
                b, c, e = a + r, a + r + 1, a + 1
                lines.append(f"f {a}//{a} {b}//{b} {c}//{c} {e}//{e}")
        offset += r * r
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- OpenCASCADE
def occ_available() -> bool:
    try:
        import OCC.Core.Geom  # noqa: F401
        return True
    except ImportError:
        return False


def _unique_knots(knots):
    vals, mults = [], []
    for k in knots:
        if vals and abs(k - vals[-1]) < 1e-12:
            mults[-1] += 1
        else:
            vals.append(float(k))
            mults.append(1)
    return vals, mults


def _real_array(values):
    from OCC.Core.TColStd import TColStd_Array1OfReal
    arr = TColStd_Array1OfReal(1, len(values))
    for i, v in enumerate(values, 1):
        arr.SetValue(i, float(v))
    return arr


def _int_array(values):
    from OCC.Core.TColStd import TColStd_Array1OfInteger
    arr = TColStd_Array1OfInteger(1, len(values))
    for i, v in enumerate(values, 1):
        arr.SetValue(i, int(v))
    return arr


def surface_to_occ(s: BSplineSurface):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.Geom import Geom_BSplineSurface
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array2OfPnt

    poles = TColgp_Array2OfPnt(1, s.nu, 1, s.nv)
    for i in range(s.nu):
        for j in range(s.nv):
            poles.SetValue(i + 1, j + 1, gp_Pnt(*map(float, s.ctrl[i, j])))
    ku, mu = _unique_knots(s.knots_u)
    kv, mv = _unique_knots(s.knots_v)
    geom = Geom_BSplineSurface(poles, _real_array(ku), _real_array(kv),
                               _int_array(mu), _int_array(mv), s.deg_u, s.deg_v)
    return BRepBuilderAPI_MakeFace(geom, 1e-6).Face()


def curve_to_occ(c: BSplineCurve):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt

    poles = TColgp_Array1OfPnt(1, c.n_ctrl)
    for i, p in enumerate(c.ctrl, 1):
        poles.SetValue(i, gp_Pnt(*map(float, p)))
    k, m = _unique_knots(c.knots)
    geom = Geom_BSplineCurve(poles, _real_array(k), _int_array(m), c.degree)
    return BRepBuilderAPI_MakeEdge(geom).Edge()


def _compound(objects):
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound

    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for o in objects:
        builder.Add(comp, surface_to_occ(o) if o.kind == "surface" else curve_to_occ(o))
    return comp


def export_step(path, objects) -> None:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    writer.Transfer(_compound(objects), STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise IOError(f"STEP export failed: {path}")


def export_iges(path, objects) -> None:
    from OCC.Core.IGESControl import IGESControl_Writer

    writer = IGESControl_Writer()
    writer.AddShape(_compound(objects))
    writer.ComputeModel()
    if not writer.Write(str(path)):
        raise IOError(f"IGES export failed: {path}")


def _as_face(shape):
    from OCC.Core import TopoDS
    if hasattr(TopoDS, "topods") and hasattr(TopoDS.topods, "Face"):
        return TopoDS.topods.Face(shape)
    return TopoDS.topods_Face(shape)


def _bspline_from_face(face):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
    from OCC.Core.Geom import Geom_BSplineSurface

    converted = _as_face(BRepBuilderAPI_NurbsConvert(face, True).Shape())
    handle = BRep_Tool.Surface(converted)
    return Geom_BSplineSurface.DownCast(handle)


def _occ_surface_to_ours(bs, name: str) -> tuple[BSplineSurface, bool]:
    nu, nv = bs.NbUPoles(), bs.NbVPoles()
    ctrl = np.array([[[bs.Pole(i, j).X(), bs.Pole(i, j).Y(), bs.Pole(i, j).Z()]
                      for j in range(1, nv + 1)] for i in range(1, nu + 1)])

    def flat_knots(n, get_k, get_m):
        out = []
        for i in range(1, n + 1):
            out += [get_k(i)] * get_m(i)
        return np.array(out)

    ku = flat_knots(bs.NbUKnots(), bs.UKnot, bs.UMultiplicity)
    kv = flat_knots(bs.NbVKnots(), bs.VKnot, bs.VMultiplicity)
    rational = bs.IsURational() or bs.IsVRational()
    if not rational and not bs.IsUPeriodic() and not bs.IsVPeriodic():
        return BSplineSurface(ctrl, bs.UDegree(), bs.VDegree(), ku, kv, name), False

    # Rational/periodic: approximate with a non-rational clamped surface by sampling OCC.
    u0, u1, v0, v1 = bs.Bounds()
    du, dv = bs.UDegree(), bs.VDegree()
    su, sv = max(1, bs.NbUKnots() - 1), max(1, bs.NbVKnots() - 1)
    m = 60
    us, vs = np.linspace(u0, u1, m), np.linspace(v0, v1, m)
    pts = np.array([[[bs.Value(u, v).X(), bs.Value(u, v).Y(), bs.Value(u, v).Z()] for v in vs] for u in us])
    from .geometry.fitting import lsq_fit_fixed_ends
    params = np.linspace(0.0, 1.0, m)
    kun, kvn = knots_from_spans(du, su * 2), knots_from_spans(dv, sv * 2)
    c = lsq_fit_fixed_ends(pts, params, du, kun)
    c = lsq_fit_fixed_ends(c.transpose(1, 0, 2), params, dv, kvn).transpose(1, 0, 2)
    return BSplineSurface(c, du, dv, kun, kvn, name), True


def import_cad(path) -> tuple[list[BSplineSurface], list[str]]:
    """Import faces of a STEP/IGES file as untrimmed B-spline surfaces."""
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer

    path = Path(path)
    if path.suffix.lower() in (".stp", ".step"):
        from OCC.Core.STEPControl import STEPControl_Reader
        reader = STEPControl_Reader()
    else:
        from OCC.Core.IGESControl import IGESControl_Reader
        reader = IGESControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"could not read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    surfaces, notes = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 0
    while exp.More():
        idx += 1
        name = f"{path.stem}_{idx}"
        try:
            bs = _bspline_from_face(_as_face(exp.Current()))
            if bs is None:
                notes.append(f"Face {idx}: not convertible to B-spline, skipped.")
            else:
                surf, approx = _occ_surface_to_ours(bs, name)
                surfaces.append(surf)
                if approx:
                    notes.append(f"Face {idx}: rational/periodic surface approximated.")
        except Exception as exc:  # keep importing the remaining faces
            notes.append(f"Face {idx}: {exc}")
        exp.Next()
    if surfaces:
        notes.append("Trim boundaries are not imported; faces come in as their full underlying surfaces.")
    return surfaces, notes
