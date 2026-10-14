"""VTK viewport: tessellation, analysis colouring, control nets, combs and point dragging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vtkmodules.qt

vtkmodules.qt.PyQtImpl = "PySide6"

import vtkmodules.vtkInteractionStyle  # noqa: E402,F401
import vtkmodules.vtkRenderingFreeType  # noqa: E402,F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: E402,F401
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray  # noqa: E402
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData  # noqa: E402
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera  # noqa: E402
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget  # noqa: E402
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkScalarBarActor  # noqa: E402
from vtkmodules.vtkRenderingCore import (  # noqa: E402
    vtkActor,
    vtkPolyDataMapper,
    vtkPropPicker,
    vtkRenderer,
)

from .. import analysis  # noqa: E402
from ..scene import GEOMETRY, OBJECTS, SELECTION, Scene  # noqa: E402

PICK_RADIUS_PX = 12
DRAG_CONSTRAINTS = ("View plane", "X", "Y", "Z")
NET_MODES = ("Selected", "All", "None")


@dataclass
class DisplaySettings:
    mode: str = analysis.SHADED
    stripes: int = 30
    resolution: int = 90
    nets: str = "Selected"
    curve_combs: bool = True
    iso_combs: bool = False
    iso_count: int = 5
    comb_scale: float = 0.05
    boundaries: bool = True
    drag_constraint: str = "View plane"


# ---------------------------------------------------------------------- VTK helpers
def _vtk_points(arr) -> vtkPoints:
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.float64).reshape(-1, 3), deep=True))
    return pts


def _cell_array(cells) -> vtkCellArray:
    """cells: 2D int array (n, k) of equal-size cells, or a list of index arrays."""
    if isinstance(cells, np.ndarray) and cells.ndim == 2:
        n = len(cells)
        legacy = np.hstack([np.full((n, 1), cells.shape[1]), cells]).ravel()
    else:
        n = len(cells)
        legacy = np.concatenate([np.concatenate([[len(c)], c]) for c in cells]) if n else np.zeros(0)
    ca = vtkCellArray()
    ca.SetCells(n, numpy_to_vtkIdTypeArray(np.ascontiguousarray(legacy, dtype=np.int64), deep=True))
    return ca


def _grid_polydata(pts, normals, mu, mv) -> vtkPolyData:
    pd = vtkPolyData()
    pd.SetPoints(_vtk_points(pts))
    i, j = np.meshgrid(np.arange(mu - 1), np.arange(mv - 1), indexing="ij")
    a = (i * mv + j).ravel()
    pd.SetPolys(_cell_array(np.stack([a, a + mv, a + mv + 1, a + 1], axis=1)))
    nrm = numpy_to_vtk(np.ascontiguousarray(normals, dtype=np.float64).reshape(-1, 3), deep=True)
    nrm.SetName("Normals")
    pd.GetPointData().SetNormals(nrm)
    return pd


def _lines_polydata(points, lines, verts=False) -> vtkPolyData:
    pd = vtkPolyData()
    pd.SetPoints(_vtk_points(points))
    if lines:
        pd.SetLines(_cell_array(lines))
    if verts:
        pd.SetVerts(_cell_array(np.arange(len(points)).reshape(-1, 1)))
    return pd


def _actor(pd, color, width=1.0, point_size=0.0, pickable=False) -> vtkActor:
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.ScalarVisibilityOff()
    actor = vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetLineWidth(width)
    prop.LightingOff()
    if point_size:
        prop.SetPointSize(point_size)
        prop.SetRenderPointsAsSpheres(True)
    actor.SetPickable(pickable)
    return actor


def _stripe_lut(n_entries: int) -> vtkLookupTable:
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(n_entries)
    for k in range(n_entries):
        c = 0.97 if k % 2 == 0 else 0.05
        lut.SetTableValue(k, c, c, c, 1.0)
    lut.Build()
    return lut


def _diverging_lut() -> vtkLookupTable:
    lut = vtkLookupTable()
    n = 256
    lut.SetNumberOfTableValues(n)
    blue, white, red = np.array([0.15, 0.3, 0.85]), np.array([0.95, 0.95, 0.95]), np.array([0.85, 0.15, 0.1])
    for k in range(n):
        t = k / (n - 1)
        c = blue + (white - blue) * (t / 0.5) if t < 0.5 else white + (red - white) * ((t - 0.5) / 0.5)
        lut.SetTableValue(k, *c, 1.0)
    lut.Build()
    return lut


def _rainbow_lut() -> vtkLookupTable:
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetHueRange(0.667, 0.0)
    lut.Build()
    return lut


# ---------------------------------------------------------------------- viewer
class Viewer(QWidget):
    def __init__(self, scene: Scene, settings: DisplaySettings, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk)

        self.ren = vtkRenderer()
        self.ren.GradientBackgroundOn()
        self.ren.SetBackground(0.16, 0.18, 0.22)
        self.ren.SetBackground2(0.46, 0.51, 0.58)
        self.ren.SetTwoSidedLighting(True)
        self.rw = self.vtk.GetRenderWindow()
        self.rw.AddRenderer(self.ren)
        self.rw.SetMultiSamples(8)
        self.iren = self.rw.GetInteractor()
        self.style = vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)
        self.style.AddObserver("LeftButtonPressEvent", self._on_press)
        self.style.AddObserver("MouseMoveEvent", self._on_move)
        self.style.AddObserver("LeftButtonReleaseEvent", self._on_release)
        self.ren.GetActiveCamera().AddObserver("ModifiedEvent", self._on_camera)

        self._diverging = _diverging_lut()
        self._rainbow = _rainbow_lut()
        self.scalar_bar = vtkScalarBarActor()
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.SetWidth(0.08)
        self.scalar_bar.SetHeight(0.5)
        self.scalar_bar.SetPosition(0.9, 0.25)
        self.scalar_bar.VisibilityOff()
        self.ren.AddViewProp(self.scalar_bar)

        self._axes_widget = vtkOrientationMarkerWidget()
        self._axes_widget.SetOrientationMarker(vtkAxesActor())
        self._axes_widget.SetInteractor(self.iren)
        self._axes_widget.SetViewport(0.0, 0.0, 0.16, 0.16)

        self._items: dict[int, dict] = {}
        self._owner: dict = {}
        self._drag = None
        self._press_pos = None
        self._initialized = False
        self._first_fit = True
        self.scene.subscribe(self._on_scene)

    def initialize(self) -> None:
        if not self._initialized:
            self.vtk.Initialize()
            self._axes_widget.EnabledOn()
            self._axes_widget.InteractiveOff()
            self._initialized = True
        self.rebuild_all()

    def render(self) -> None:
        if self._initialized:
            self.rw.Render()

    # ------------------------------------------------------------------ scene events
    def _on_scene(self, kind: str, oid) -> None:
        if kind == OBJECTS:
            self.rebuild_all()
        elif kind == GEOMETRY and oid in self.scene.objects:
            self._rebuild_geometry(oid)
            self._rebuild_overlay(oid)
            self._update_curvature_range()
            self.render()
        elif kind == SELECTION:
            for k in list(self._items):
                self._rebuild_overlay(k)
            self.render()

    def rebuild_all(self) -> None:
        for oid in list(self._items):
            self._remove_item(oid)
        for oid in self.scene.objects:
            self._rebuild_geometry(oid)
            self._rebuild_overlay(oid)
        self._update_curvature_range()
        if self._first_fit and self.scene.objects:
            self.fit_view()
            self._first_fit = False
        self.render()

    def fit_view(self) -> None:
        self.ren.ResetCamera()
        self.render()

    def set_parallel(self, on: bool) -> None:
        self.ren.GetActiveCamera().SetParallelProjection(on)
        self.render()

    def _remove_item(self, oid) -> None:
        item = self._items.pop(oid, None)
        if not item:
            return
        for a in item.get("geom_actors", []) + item.get("overlay_actors", []):
            self.ren.RemoveActor(a)
            self._owner.pop(a, None)

    def _scene_diag(self) -> float:
        pts = [o.ctrl.reshape(-1, 3) for o in self.scene.objects.values()]
        if not pts:
            return 100.0
        allp = np.vstack(pts)
        return float(np.linalg.norm(allp.max(0) - allp.min(0))) or 100.0

    def _add(self, oid, actor, group: str) -> None:
        self.ren.AddActor(actor)
        self._items[oid].setdefault(group, []).append(actor)
        self._owner[actor] = oid

    def _clear_group(self, oid, group: str) -> None:
        item = self._items.setdefault(oid, {})
        for a in item.get(group, []):
            self.ren.RemoveActor(a)
            self._owner.pop(a, None)
        item[group] = []

    # ------------------------------------------------------------------ geometry actors
    def _comb_actor(self, curve, scale, color, n=150):
        base, tips = curve.comb(n, scale)
        pts = np.vstack([base, tips])
        lines = [np.array([i, n + i]) for i in range(n)] + [np.arange(n, 2 * n)]
        return _actor(_lines_polydata(pts, lines), color, width=1.0)

    def _rebuild_geometry(self, oid) -> None:
        self._clear_group(oid, "geom_actors")
        obj = self.scene.get(oid)
        item = self._items[oid]
        s = self.settings
        comb_scale = s.comb_scale * self._scene_diag() ** 2
        if obj.kind == "surface":
            r = s.resolution
            t = np.linspace(0.0, 1.0, r)
            d = obj.differential(t, t)
            pd = _grid_polydata(d["S"], d["N"], r, r)
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(pd)
            actor = vtkActor()
            actor.SetMapper(mapper)
            item.update(diff=d, pd=pd, mapper=mapper, surf_actor=actor)
            self._add(oid, actor, "geom_actors")
            self._apply_analysis(oid)
            if s.boundaries:
                S = d["S"]
                edges = [S[0, :], S[-1, :], S[:, 0], S[:, -1]]
                pts = np.vstack(edges)
                lines = [np.arange(k * r, (k + 1) * r) for k in range(4)]
                self._add(oid, _actor(_lines_polydata(pts, lines), (0.08, 0.08, 0.1), width=2.0), "geom_actors")
            if s.iso_combs:
                for p in np.linspace(0.0, 1.0, s.iso_count):
                    for c in (obj.iso_u(p), obj.iso_v(p)):
                        self._add(oid, self._comb_actor(c, comb_scale, (0.3, 0.75, 1.0), 80), "geom_actors")
        else:
            pts = obj.points(400)
            line = _actor(_lines_polydata(pts, [np.arange(len(pts))]), (0.95, 0.85, 0.2), width=3.0, pickable=True)
            self._add(oid, line, "geom_actors")
            if s.curve_combs:
                self._add(oid, self._comb_actor(obj, comb_scale, (0.9, 0.35, 0.8)), "geom_actors")

    def _apply_analysis(self, oid) -> None:
        item = self._items[oid]
        mapper, actor, pd, d = item["mapper"], item["surf_actor"], item["pd"], item["diff"]
        prop = actor.GetProperty()
        mode = self.settings.mode
        selected = oid == self.scene.selected
        if mode == analysis.SHADED:
            mapper.ScalarVisibilityOff()
            prop.SetColor(*((0.78, 0.82, 0.9) if selected else (0.72, 0.74, 0.78)))
            prop.SetAmbient(0.1)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.5)
            prop.SetSpecularPower(60)
            return
        if mode in analysis.STRIPE_MODES:
            phase, lo, hi = self._stripe_phase(d)
            arr = numpy_to_vtk(np.ascontiguousarray(phase.ravel(), dtype=np.float64), deep=True)
            arr.SetName("phase")
            pd.GetPointData().SetScalars(arr)
            mapper.SetLookupTable(_stripe_lut(hi - lo))
            mapper.SetScalarRange(lo, hi)
            mapper.SetInterpolateScalarsBeforeMapping(True)
            mapper.ScalarVisibilityOn()
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
            prop.SetSpecular(0.0)
            return
        values = analysis.curvature_field(d, mode)
        arr = numpy_to_vtk(np.ascontiguousarray(values.ravel(), dtype=np.float64), deep=True)
        arr.SetName("curvature")
        pd.GetPointData().SetScalars(arr)
        mapper.SetLookupTable(self._rainbow if mode == analysis.MAX_ABS else self._diverging)
        mapper.SetInterpolateScalarsBeforeMapping(True)
        mapper.ScalarVisibilityOn()
        prop.SetAmbient(0.35)
        prop.SetDiffuse(0.65)
        prop.SetSpecular(0.15)

    def _stripe_phase(self, d):
        n = self.settings.stripes
        if self.settings.mode == analysis.ZEBRA:
            cam = self.ren.GetActiveCamera()
            pos = np.array(cam.GetPosition())
            if cam.GetParallelProjection():
                pos = np.array(cam.GetFocalPoint()) - np.array(cam.GetDirectionOfProjection()) * 1e7
            phase = analysis.zebra_phase(d["S"], d["N"], pos, cam.GetViewUp(), n)
            m = int(np.ceil(n / 2))
            return phase, -m, m
        cam = self.ren.GetActiveCamera()
        phase = analysis.isophote_phase(d["N"], cam.GetDirectionOfProjection(), n)
        return phase, 0, n

    def _update_curvature_range(self) -> None:
        mode = self.settings.mode
        if mode not in analysis.CURVATURE_MODES:
            self.scalar_bar.VisibilityOff()
            return
        fields = [analysis.curvature_field(it["diff"], mode).ravel()
                  for it in self._items.values() if "diff" in it and it.get("surf_actor") is not None]
        if not fields:
            self.scalar_bar.VisibilityOff()
            return
        lo, hi = analysis.robust_range(np.concatenate(fields), symmetric=mode != analysis.MAX_ABS)
        for oid, it in self._items.items():
            if "mapper" in it and self.scene.get(oid) is not None and self.scene.get(oid).kind == "surface":
                it["mapper"].SetScalarRange(lo, hi)
        self.scalar_bar.SetLookupTable(self._rainbow if mode == analysis.MAX_ABS else self._diverging)
        self._rainbow.SetTableRange(lo, hi)
        self._diverging.SetTableRange(lo, hi)
        self.scalar_bar.SetTitle(mode.replace(" curvature", "\ncurvature"))
        self.scalar_bar.VisibilityOn()

    def _on_camera(self, *_):
        if self.settings.mode not in analysis.STRIPE_MODES:
            return
        for oid, it in self._items.items():
            if "diff" in it and self.scene.get(oid) is not None and self.scene.get(oid).kind == "surface":
                phase, _, _ = self._stripe_phase(it["diff"])
                arr = numpy_to_vtk(np.ascontiguousarray(phase.ravel(), dtype=np.float64), deep=True)
                arr.SetName("phase")
                it["pd"].GetPointData().SetScalars(arr)
                it["pd"].Modified()

    def settings_changed(self, geometry: bool = True) -> None:
        if geometry:
            self.rebuild_all()
            return
        for oid in self._items:
            if self.scene.get(oid) is not None and self.scene.get(oid).kind == "surface":
                self._apply_analysis(oid)
        self._update_curvature_range()
        self.render()

    # ------------------------------------------------------------------ overlays
    def _net_visible(self, oid) -> bool:
        mode = self.settings.nets
        return mode == "All" or (mode == "Selected" and oid == self.scene.selected)

    def _rebuild_overlay(self, oid) -> None:
        self._clear_group(oid, "overlay_actors")
        obj = self.scene.get(oid)
        if obj is None:
            return
        if obj.kind == "surface" and "surf_actor" in self._items[oid] and self.settings.mode == analysis.SHADED:
            self._apply_analysis(oid)
        if not self._net_visible(oid):
            return
        selected = oid == self.scene.selected
        color = (1.0, 0.55, 0.1) if selected else (0.6, 0.6, 0.65)
        if obj.kind == "surface":
            nu, nv = obj.nu, obj.nv
            idx = np.arange(nu * nv).reshape(nu, nv)
            lines = [idx[i, :] for i in range(nu)] + [idx[:, j] for j in range(nv)]
            pts = obj.ctrl.reshape(-1, 3)
        else:
            pts = obj.ctrl
            lines = [np.arange(len(pts))]
        self._add(oid, _actor(_lines_polydata(pts, lines, verts=True), color, width=1.2, point_size=9), "overlay_actors")
        cp = self.scene.selected_cp if selected else None
        if cp is not None:
            p = obj.ctrl[cp].reshape(1, 3)
            self._add(oid, _actor(_lines_polydata(p, [], verts=True), (1.0, 0.1, 0.1), point_size=15), "overlay_actors")

    # ------------------------------------------------------------------ picking & dragging
    def _world_to_display(self, p):
        self.ren.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
        self.ren.WorldToDisplay()
        return self.ren.GetDisplayPoint()

    def _display_to_world(self, x, y, z):
        self.ren.SetDisplayPoint(float(x), float(y), float(z))
        self.ren.DisplayToWorld()
        w = self.ren.GetWorldPoint()
        return np.array(w[:3]) / (w[3] if w[3] else 1.0)

    def _pick_cp(self, x, y):
        best = None
        order = sorted(self.scene.objects, key=lambda k: k != self.scene.selected)
        for oid in order:
            if not self._net_visible(oid):
                continue
            obj = self.scene.get(oid)
            shape = obj.ctrl.shape[:-1]
            for flat, p in enumerate(obj.ctrl.reshape(-1, 3)):
                dx, dy, dz = self._world_to_display(p)
                dist = np.hypot(dx - x, dy - y)
                if dist <= PICK_RADIUS_PX and (best is None or dist < best[0] - 1e-9):
                    best = (dist, oid, np.unravel_index(flat, shape), dz)
        return best

    def _on_press(self, *_):
        x, y = self.iren.GetEventPosition()
        hit = self._pick_cp(x, y)
        if hit is None:
            self._press_pos = (x, y)
            self.style.OnLeftButtonDown()
            return
        _, oid, idx, z = hit
        idx = tuple(int(i) for i in idx)
        obj = self.scene.get(oid)
        self.scene.checkpoint()
        self._drag = {"oid": oid, "idx": idx, "z": z, "start": self._display_to_world(x, y, z),
                      "orig": obj.ctrl.copy(), "row": bool(self.iren.GetControlKey()),
                      "col": bool(self.iren.GetShiftKey())}
        self.scene.select(oid, idx)

    def _on_move(self, *_):
        if self._drag is None:
            self.style.OnMouseMove()
            return
        x, y = self.iren.GetEventPosition()
        dr = self._drag
        delta = self._display_to_world(x, y, dr["z"]) - dr["start"]
        axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}.get(self.settings.drag_constraint)
        if axis is not None:
            a = np.array(axis, dtype=float)
            delta = a * (delta @ a)
        obj = self.scene.get(dr["oid"])
        ctrl = dr["orig"].copy()
        idx = dr["idx"]
        if obj.kind == "surface" and dr["row"]:
            ctrl[idx[0], :] += delta
        elif obj.kind == "surface" and dr["col"]:
            ctrl[:, idx[1]] += delta
        else:
            ctrl[idx] += delta
        obj.ctrl = ctrl
        self.scene.notify(GEOMETRY, dr["oid"])

    def _on_release(self, *_):
        if self._drag is not None:
            self._drag = None
            return
        self.style.OnLeftButtonUp()
        if self._press_pos is None:
            return
        x, y = self.iren.GetEventPosition()
        px, py = self._press_pos
        self._press_pos = None
        if abs(x - px) + abs(y - py) > 3:
            return
        picker = vtkPropPicker()
        picker.Pick(x, y, 0, self.ren)
        actor = picker.GetActor()
        self.scene.select(self._owner.get(actor) if actor is not None else None)
