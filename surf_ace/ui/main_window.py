from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import analysis, fileio
from ..continuity import LEVELS, closest_edge_pair, continuity_report, match_continuity
from ..geometry import EDGES, BSplineCurve, BSplineSurface
from ..scene import GEOMETRY, OBJECTS, SELECTION, Scene
from .viewer import DRAG_CONSTRAINTS, NET_MODES, DisplaySettings, Viewer

EDGE_LABELS = {"u0": "u = 0", "u1": "u = 1", "v0": "v = 0", "v1": "v = 1"}


def _spin(lo, hi, val, step=1):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    return s


def _dspin(lo=-1e6, hi=1e6, val=0.0, dec=4):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    s.setKeyboardTracking(False)
    return s


class NewPatchDialog(QDialog):
    def __init__(self, parent=None, curve=False):
        super().__init__(parent)
        self.setWindowTitle("New B-spline curve" if curve else "New Bezier / B-spline patch")
        form = QFormLayout(self)
        self.deg_u = _spin(1, 9, 5)
        self.spans_u = _spin(1, 20, 1)
        self.size_u = _dspin(0.001, 1e5, 100.0, 2)
        form.addRow("Degree" if curve else "Degree u", self.deg_u)
        form.addRow("Spans" if curve else "Spans u", self.spans_u)
        form.addRow("Length" if curve else "Size u", self.size_u)
        self.curve = curve
        if not curve:
            self.deg_v = _spin(1, 9, 5)
            self.spans_v = _spin(1, 20, 1)
            self.size_v = _dspin(0.001, 1e5, 100.0, 2)
            self.crown = _dspin(-1e4, 1e4, 10.0, 2)
            form.addRow("Degree v", self.deg_v)
            form.addRow("Spans v", self.spans_v)
            form.addRow("Size v", self.size_v)
            form.addRow("Crown height", self.crown)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surf-Ace - Class A Surfacing")
        self.resize(1500, 920)
        self.scene = Scene()
        self.settings = DisplaySettings()
        self.viewer = Viewer(self.scene, self.settings, self)
        self.setCentralWidget(self.viewer)
        self.session_path: Path | None = None
        self._updating = False

        self._build_actions()
        self._build_object_dock()
        self._build_side_dock()
        self.scene.subscribe(self._on_scene)
        self.statusBar().showMessage(
            "Left-drag: rotate / move control point | Ctrl+drag: move row | Shift+drag: move column | "
            "Middle: pan | Wheel: zoom")

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._after_show)

    def _after_show(self):
        if not self.viewer._initialized:
            self.viewer.initialize()
            if not self.scene.objects:
                self.load_demo()

    # ------------------------------------------------------------------ actions/menus
    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _build_actions(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._act(f, "New", self.new_session, QKeySequence.New)
        self._act(f, "Open session...", self.open_session, QKeySequence.Open)
        self._act(f, "Save session", self.save_session, QKeySequence.Save)
        self._act(f, "Save session as...", lambda: self.save_session(ask=True), QKeySequence.SaveAs)
        f.addSeparator()
        occ = fileio.occ_available()
        self._act(f, "Import STEP / IGES...", self.import_cad).setEnabled(occ)
        self._act(f, "Export STEP...", lambda: self.export_cad("step")).setEnabled(occ)
        self._act(f, "Export IGES...", lambda: self.export_cad("iges")).setEnabled(occ)
        self._act(f, "Export OBJ (tessellated)...", self.export_obj)
        f.addSeparator()
        self._act(f, "Quit", self.close, QKeySequence.Quit)

        e = mb.addMenu("&Edit")
        self._act(e, "Undo", self.undo, QKeySequence.Undo)
        self._act(e, "Redo", self.redo, QKeySequence.Redo)
        e.addSeparator()
        self._act(e, "Duplicate", self.duplicate, "Ctrl+D")
        self._act(e, "Delete", self.delete_selected, QKeySequence.Delete)

        c = mb.addMenu("&Create")
        self._act(c, "Patch...", self.create_patch, "Ctrl+Shift+P")
        self._act(c, "Curve...", self.create_curve, "Ctrl+Shift+C")
        c.addSeparator()
        self._act(c, "Demo scene", self.load_demo)

        v = mb.addMenu("&View")
        self._act(v, "Fit all", self.viewer.fit_view, "F")
        par = self._act(v, "Parallel projection", lambda on: self.viewer.set_parallel(on), "P")
        par.setCheckable(True)
        for name, (pos, up) in {"Top (+Z)": ((0, 0, 1), (0, 1, 0)), "Front (-Y)": ((0, -1, 0), (0, 0, 1)),
                                "Side (+X)": ((1, 0, 0), (0, 0, 1)), "Iso": ((1, -1, 0.8), (0, 0, 1))}.items():
            self._act(v, name, lambda _=False, p=pos, u=up: self.set_view(p, u))

    def set_view(self, direction, up):
        cam = self.viewer.ren.GetActiveCamera()
        fp = np.array(cam.GetFocalPoint())
        dist = cam.GetDistance()
        d = np.array(direction, dtype=float)
        cam.SetPosition(*(fp + d / np.linalg.norm(d) * dist))
        cam.SetViewUp(*up)
        self.viewer.fit_view()

    # ------------------------------------------------------------------ object list dock
    def _build_object_dock(self):
        dock = QDockWidget("Objects", self)
        dock.setObjectName("objects")
        w = QWidget()
        lay = QVBoxLayout(w)
        self.obj_list = QListWidget()
        self.obj_list.currentItemChanged.connect(self._list_selection)
        self.obj_list.itemChanged.connect(self._list_renamed)
        lay.addWidget(self.obj_list)
        row = QHBoxLayout()
        for text, slot in (("+ Patch", self.create_patch), ("+ Curve", self.create_curve),
                           ("Delete", self.delete_selected)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        dock.setWidget(w)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _refresh_list(self):
        self._updating = True
        self.obj_list.clear()
        for oid, obj in self.scene.objects.items():
            tag = "Patch" if obj.kind == "surface" else "Curve"
            it = QListWidgetItem(obj.name)
            it.setToolTip(tag)
            it.setData(Qt.UserRole, oid)
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.obj_list.addItem(it)
            if oid == self.scene.selected:
                self.obj_list.setCurrentItem(it)
        self._updating = False

    def _list_selection(self, cur, _prev):
        if self._updating:
            return
        self.scene.select(cur.data(Qt.UserRole) if cur else None)

    def _list_renamed(self, item):
        if self._updating:
            return
        obj = self.scene.get(item.data(Qt.UserRole))
        if obj is not None and item.text() and item.text() != obj.name:
            obj.name = item.text()
            self._refresh_continuity_combos()

    # ------------------------------------------------------------------ side dock
    def _build_side_dock(self):
        dock = QDockWidget("Tools", self)
        dock.setObjectName("tools")
        tabs = QTabWidget()
        tabs.addTab(self._build_props_tab(), "Properties")
        tabs.addTab(self._build_display_tab(), "Display / Analysis")
        tabs.addTab(self._build_continuity_tab(), "Continuity")
        dock.setWidget(tabs)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_props_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.prop_title = QLabel("Nothing selected")
        self.prop_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        lay.addWidget(self.prop_title)

        box = QGroupBox("Structure (rebuild = least-squares refit, exact for degree raise)")
        form = QFormLayout(box)
        self.p_deg_u, self.p_spans_u = _spin(1, 9, 5), _spin(1, 30, 1)
        self.p_deg_v, self.p_spans_v = _spin(1, 9, 5), _spin(1, 30, 1)
        form.addRow("Degree u", self.p_deg_u)
        form.addRow("Spans u", self.p_spans_u)
        self.p_v_label1, self.p_v_label2 = QLabel("Degree v"), QLabel("Spans v")
        form.addRow(self.p_v_label1, self.p_deg_v)
        form.addRow(self.p_v_label2, self.p_spans_v)
        self.p_info = QLabel("")
        form.addRow(self.p_info)
        rebuild = QPushButton("Rebuild")
        rebuild.clicked.connect(self.rebuild_selected)
        form.addRow(rebuild)
        lay.addWidget(box)
        self.struct_box = box

        cbox = QGroupBox("Selected control point")
        cform = QFormLayout(cbox)
        self.cp_label = QLabel("-")
        cform.addRow("Index", self.cp_label)
        self.cp_xyz = [_dspin() for _ in range(3)]
        for name, sb in zip("XYZ", self.cp_xyz):
            sb.valueChanged.connect(self._cp_edited)
            cform.addRow(name, sb)
        lay.addWidget(cbox)
        self.cp_box = cbox

        tbox = QGroupBox("Transform selected object")
        tl = QHBoxLayout(tbox)
        self.t_xyz = [_dspin(val=0.0, dec=3) for _ in range(3)]
        for name, sb in zip("XYZ", self.t_xyz):
            tl.addWidget(QLabel(name))
            tl.addWidget(sb)
        mv = QPushButton("Translate")
        mv.clicked.connect(self.translate_selected)
        tl.addWidget(mv)
        lay.addWidget(tbox)
        lay.addStretch(1)
        return w

    def _build_display_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        s = self.settings
        self.d_mode = QComboBox()
        self.d_mode.addItems(analysis.MODES)
        self.d_mode.currentTextChanged.connect(self._display_changed_light)
        form.addRow("Analysis", self.d_mode)
        self.d_stripes = _spin(2, 80, s.stripes, 2)
        self.d_stripes.valueChanged.connect(self._display_changed_light)
        form.addRow("Stripes", self.d_stripes)
        self.d_res = _spin(10, 300, s.resolution, 10)
        self.d_res.valueChanged.connect(self._display_changed)
        form.addRow("Tessellation", self.d_res)
        self.d_nets = QComboBox()
        self.d_nets.addItems(NET_MODES)
        self.d_nets.currentTextChanged.connect(self._display_changed)
        form.addRow("Control nets", self.d_nets)
        self.d_bound = QCheckBox("Show patch boundaries")
        self.d_bound.setChecked(s.boundaries)
        self.d_bound.toggled.connect(self._display_changed)
        form.addRow(self.d_bound)
        self.d_ccombs = QCheckBox("Curvature combs on curves")
        self.d_ccombs.setChecked(s.curve_combs)
        self.d_ccombs.toggled.connect(self._display_changed)
        form.addRow(self.d_ccombs)
        self.d_icombs = QCheckBox("Curvature combs on surface iso-curves")
        self.d_icombs.setChecked(s.iso_combs)
        self.d_icombs.toggled.connect(self._display_changed)
        form.addRow(self.d_icombs)
        self.d_icount = _spin(2, 30, s.iso_count)
        self.d_icount.valueChanged.connect(self._display_changed)
        form.addRow("Iso-curves per direction", self.d_icount)
        self.d_scale = QSlider(Qt.Horizontal)
        self.d_scale.setRange(0, 100)
        self.d_scale.setValue(self._scale_to_slider(s.comb_scale))
        self.d_scale.valueChanged.connect(self._display_changed)
        form.addRow("Comb scale", self.d_scale)
        self.d_drag = QComboBox()
        self.d_drag.addItems(DRAG_CONSTRAINTS)
        self.d_drag.currentTextChanged.connect(lambda t: setattr(self.settings, "drag_constraint", t))
        form.addRow("Drag constraint", self.d_drag)
        hint = QLabel("Zebra stripes follow the camera: rotate the view to sweep reflections across "
                      "patch boundaries. Kinks in stripes = G1 breaks, sharp bends = G2 breaks.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        form.addRow(hint)
        return w

    @staticmethod
    def _scale_to_slider(v):
        return int(round((math.log10(v) + 3) / 3 * 100))

    @staticmethod
    def _slider_to_scale(p):
        return 10 ** (p / 100 * 3 - 3)

    def _read_display(self):
        s = self.settings
        s.mode = self.d_mode.currentText()
        s.stripes = self.d_stripes.value()
        s.resolution = self.d_res.value()
        s.nets = self.d_nets.currentText()
        s.boundaries = self.d_bound.isChecked()
        s.curve_combs = self.d_ccombs.isChecked()
        s.iso_combs = self.d_icombs.isChecked()
        s.iso_count = self.d_icount.value()
        s.comb_scale = self._slider_to_scale(self.d_scale.value())

    def _display_changed(self, *_):
        self._read_display()
        self.viewer.settings_changed(geometry=True)

    def _display_changed_light(self, *_):
        self._read_display()
        self.viewer.settings_changed(geometry=False)

    def _build_continuity_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.c_a, self.c_b = QComboBox(), QComboBox()
        self.c_ea, self.c_eb = QComboBox(), QComboBox()
        for cb in (self.c_ea, self.c_eb):
            for e in EDGES:
                cb.addItem(EDGE_LABELS[e], e)
        form.addRow("Reference patch (A)", self.c_a)
        form.addRow("Edge of A", self.c_ea)
        form.addRow("Patch to modify (B)", self.c_b)
        form.addRow("Edge of B", self.c_eb)
        self.c_level = QComboBox()
        self.c_level.addItems(list(LEVELS))
        self.c_level.setCurrentText("G2")
        form.addRow("Continuity", self.c_level)
        lay.addLayout(form)
        row = QHBoxLayout()
        for text, slot in (("Auto-detect edges", self.auto_edges), ("Match B to A", self.match_edges),
                           ("Check", self.check_edges)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        tol = QGroupBox("Check tolerances")
        tf = QFormLayout(tol)
        self.c_tpos = _dspin(0, 10, 0.005, 4)
        self.c_tang = _dspin(0, 10, 0.05, 3)
        self.c_tcurv = _dspin(0, 100, 2.0, 2)
        tf.addRow("Position", self.c_tpos)
        tf.addRow("Tangent (deg)", self.c_tang)
        tf.addRow("Curvature (%)", self.c_tcurv)
        lay.addWidget(tol)
        self.c_report = QPlainTextEdit()
        self.c_report.setReadOnly(True)
        self.c_report.setStyleSheet("font-family: Consolas, monospace;")
        lay.addWidget(self.c_report, 1)
        return w

    def _refresh_continuity_combos(self):
        for cb in (self.c_a, self.c_b):
            cur = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for oid, obj in self.scene.surfaces().items():
                cb.addItem(obj.name, oid)
            i = cb.findData(cur)
            if i >= 0:
                cb.setCurrentIndex(i)
            cb.blockSignals(False)
        if self.c_a.currentData() == self.c_b.currentData() and self.c_b.count() > 1:
            self.c_b.setCurrentIndex(1 if self.c_a.currentIndex() == 0 else 0)

    def _continuity_inputs(self):
        ia, ib = self.c_a.currentData(), self.c_b.currentData()
        if ia is None or ib is None or ia == ib:
            QMessageBox.information(self, "Continuity", "Choose two different patches.")
            return None
        return ia, self.c_ea.currentData(), ib, self.c_eb.currentData()

    def auto_edges(self):
        inp = self._continuity_inputs()
        if not inp:
            return
        ea, eb = closest_edge_pair(self.scene.get(inp[0]), self.scene.get(inp[2]))
        self.c_ea.setCurrentIndex(self.c_ea.findData(ea))
        self.c_eb.setCurrentIndex(self.c_eb.findData(eb))

    def match_edges(self):
        inp = self._continuity_inputs()
        if not inp:
            return
        ia, ea, ib, eb = inp
        level = self.c_level.currentText()
        try:
            newb, msgs = match_continuity(self.scene.get(ia), ea, self.scene.get(ib), eb, level)
        except Exception as exc:
            QMessageBox.warning(self, "Match failed", str(exc))
            return
        self.scene.replace(ib, newb)
        rep = self._report(ia, ea, ib, eb)
        self.c_report.setPlainText("\n".join([f"Matched {level}."] + msgs + ["", rep.summary()]))

    def _report(self, ia, ea, ib, eb):
        return continuity_report(self.scene.get(ia), ea, self.scene.get(ib), eb,
                                 tol_pos=self.c_tpos.value(), tol_angle=self.c_tang.value(),
                                 tol_curv=self.c_tcurv.value() / 100.0)

    def check_edges(self):
        inp = self._continuity_inputs()
        if inp:
            self.c_report.setPlainText(self._report(*inp).summary())

    # ------------------------------------------------------------------ scene sync
    def _on_scene(self, kind, oid):
        if kind == OBJECTS:
            self._refresh_list()
            self._refresh_continuity_combos()
        elif kind == SELECTION:
            self._updating = True
            for i in range(self.obj_list.count()):
                if self.obj_list.item(i).data(Qt.UserRole) == self.scene.selected:
                    self.obj_list.setCurrentRow(i)
                    break
            else:
                self.obj_list.clearSelection()
            self._updating = False
        self._refresh_props()
        self._update_title()

    def _update_title(self):
        name = self.session_path.name if self.session_path else "untitled"
        self.setWindowTitle(f"Surf-Ace - {name}{' *' if self.scene.modified else ''}")

    def _refresh_props(self):
        obj = self.scene.get(self.scene.selected) if self.scene.selected else None
        self._updating = True
        self.struct_box.setEnabled(obj is not None)
        if obj is None:
            self.prop_title.setText("Nothing selected")
        elif obj.kind == "surface":
            self.prop_title.setText(f"Patch: {obj.name}")
            self.p_deg_u.setValue(obj.deg_u)
            self.p_spans_u.setValue(obj.spans_u)
            self.p_deg_v.setValue(obj.deg_v)
            self.p_spans_v.setValue(obj.spans_v)
            kind = "Bezier patch" if obj.spans_u == 1 and obj.spans_v == 1 else "B-spline patch"
            self.p_info.setText(f"{kind}, {obj.nu} x {obj.nv} control points")
        else:
            self.prop_title.setText(f"Curve: {obj.name}")
            self.p_deg_u.setValue(obj.degree)
            self.p_spans_u.setValue(obj.spans)
            kind = "Bezier curve" if obj.spans == 1 else "B-spline curve"
            self.p_info.setText(f"{kind}, {obj.n_ctrl} control points")
        is_surf = obj is not None and obj.kind == "surface"
        for wdg in (self.p_deg_v, self.p_spans_v, self.p_v_label1, self.p_v_label2):
            wdg.setVisible(is_surf or obj is None)
        cp = self.scene.selected_cp
        self.cp_box.setEnabled(obj is not None and cp is not None)
        if obj is not None and cp is not None:
            self.cp_label.setText(str(tuple(int(i) for i in cp)))
            for sb, val in zip(self.cp_xyz, obj.ctrl[cp]):
                sb.blockSignals(True)
                sb.setValue(float(val))
                sb.blockSignals(False)
        else:
            self.cp_label.setText("-")
        self._updating = False

    # ------------------------------------------------------------------ editing commands
    def _selected(self):
        return self.scene.selected, self.scene.get(self.scene.selected) if self.scene.selected else None

    def _cp_edited(self, *_):
        oid, obj = self._selected()
        cp = self.scene.selected_cp
        if self._updating or obj is None or cp is None:
            return
        self.scene.checkpoint()
        obj.ctrl[cp] = [sb.value() for sb in self.cp_xyz]
        self.scene.notify(GEOMETRY, oid)

    def rebuild_selected(self):
        oid, obj = self._selected()
        if obj is None:
            return
        if obj.kind == "surface":
            new = obj.rebuilt(self.p_deg_u.value(), self.p_spans_u.value(),
                              self.p_deg_v.value(), self.p_spans_v.value())
        else:
            new = obj.rebuilt(self.p_deg_u.value(), self.p_spans_u.value())
        self.scene.replace(oid, new)
        self.statusBar().showMessage(f"Rebuilt '{obj.name}'.", 4000)

    def translate_selected(self):
        oid, obj = self._selected()
        if obj is None:
            return
        self.scene.checkpoint()
        obj.ctrl = obj.ctrl + np.array([sb.value() for sb in self.t_xyz])
        self.scene.notify(GEOMETRY, oid)

    def create_patch(self):
        dlg = NewPatchDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        s = BSplineSurface.bezier_patch(origin=self._free_origin(), size=(dlg.size_u.value(), dlg.size_v.value()),
                                        degree=(dlg.deg_u.value(), dlg.deg_v.value()), crown=dlg.crown.value(),
                                        name=self.scene.unique_name("Patch"))
        s = s.rebuilt(s.deg_u, dlg.spans_u.value(), s.deg_v, dlg.spans_v.value())
        self.scene.add(s)

    def create_curve(self):
        dlg = NewPatchDialog(self, curve=True)
        if dlg.exec() != QDialog.Accepted:
            return
        deg, spans, length = dlg.deg_u.value(), dlg.spans_u.value(), dlg.size_u.value()
        n = deg + spans
        x = np.linspace(0.0, length, n)
        y = 0.15 * length * np.sin(np.linspace(0, np.pi, n))
        ctrl = np.stack([x, y, np.zeros(n)], 1) + self._free_origin()
        self.scene.add(BSplineCurve(ctrl, deg, name=self.scene.unique_name("Curve")))

    def _free_origin(self):
        if not self.scene.objects:
            return np.zeros(3)
        allp = np.vstack([o.ctrl.reshape(-1, 3) for o in self.scene.objects.values()])
        return np.array([allp[:, 0].min(), allp[:, 1].max() + 20.0, 0.0])

    def duplicate(self):
        oid, obj = self._selected()
        if obj is None:
            return
        new = obj.copy()
        new.name = self.scene.unique_name(obj.name.rsplit(" ", 1)[0])
        size = np.ptp(obj.ctrl.reshape(-1, 3), axis=0)
        new.ctrl = new.ctrl + np.array([0.0, size[1] + 10.0, 0.0])
        self.scene.add(new)

    def delete_selected(self):
        if self.scene.selected is not None:
            self.scene.remove(self.scene.selected)

    def undo(self):
        if not self.scene.undo():
            self.statusBar().showMessage("Nothing to undo.", 2000)

    def redo(self):
        if not self.scene.redo():
            self.statusBar().showMessage("Nothing to redo.", 2000)

    def load_demo(self):
        """Two Bezier patches sharing an edge with a deliberate curvature break, plus a curve."""
        rng = np.random.default_rng(7)
        a = BSplineSurface.bezier_patch(size=(120, 100), degree=(5, 5), crown=14.0, name="Hood A")
        b = BSplineSurface.bezier_patch(origin=(120, 0, 0), size=(90, 100), degree=(6, 5), crown=-6.0, name="Hood B")
        b.ctrl[0] = a.ctrl[-1] + np.array([0.0, 0.0, 0.0])
        b.ctrl[1:, :, 2] += 4.0 + rng.normal(scale=0.8, size=b.ctrl[1:, :, 2].shape)
        t = np.linspace(0, 1, 6)
        curve = BSplineCurve(np.stack([t * 210, np.full(6, -25.0), 8 * np.sin(np.pi * t) + 2 * t], 1), 5,
                             name="Feature line")
        for obj in (a, b, curve):
            self.scene.add(obj, checkpoint=False)
        self.scene.select(None)
        self._refresh_continuity_combos()
        self.c_a.setCurrentIndex(self.c_a.findText("Hood A"))
        self.c_b.setCurrentIndex(self.c_b.findText("Hood B"))
        self.c_ea.setCurrentIndex(self.c_ea.findData("u1"))
        self.c_eb.setCurrentIndex(self.c_eb.findData("u0"))
        self.scene.modified = False
        self._update_title()
        self.viewer.fit_view()

    # ------------------------------------------------------------------ files
    def _confirm_discard(self) -> bool:
        if not self.scene.modified or not self.scene.objects:
            return True
        r = QMessageBox.question(self, "Discard changes?", "The current session has unsaved changes. Discard them?")
        return r == QMessageBox.Yes

    def new_session(self):
        if self._confirm_discard():
            self.scene.clear()
            self.session_path = None
            self._update_title()

    def open_session(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open session", "", "Surf-Ace session (*.json)")
        if not path:
            return
        try:
            objs = fileio.load_session(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.scene.clear()
        for o in objs:
            self.scene.add(o, select=False, checkpoint=False)
        self.scene.modified = False
        self.session_path = Path(path)
        self.viewer.fit_view()
        self._update_title()

    def save_session(self, ask=False):
        if ask or self.session_path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save session", "session.json", "Surf-Ace session (*.json)")
            if not path:
                return
            self.session_path = Path(path)
        fileio.save_session(self.session_path, list(self.scene.objects.values()))
        self.scene.modified = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.session_path}", 4000)

    def import_cad(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import", "", "CAD (*.step *.stp *.iges *.igs)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            surfaces, notes = fileio.import_cad(path)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.scene.checkpoint()
        for s in surfaces:
            self.scene.add(s, select=False, checkpoint=False)
        self.viewer.fit_view()
        QMessageBox.information(self, "Import", f"Imported {len(surfaces)} surface(s).\n\n" + "\n".join(notes[:30]))

    def _export_targets(self):
        objs = list(self.scene.objects.values())
        if not objs:
            QMessageBox.information(self, "Export", "Nothing to export.")
        return objs

    def export_cad(self, fmt):
        objs = self._export_targets()
        if not objs:
            return
        filt = "STEP (*.step *.stp)" if fmt == "step" else "IGES (*.iges *.igs)"
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", f"surf-ace.{fmt}", filt)
        if not path:
            return
        try:
            (fileio.export_step if fmt == "step" else fileio.export_iges)(path, objs)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {len(objs)} object(s) to {path}", 5000)

    def export_obj(self):
        surfs = [o for o in self._export_targets() if o.kind == "surface"]
        if not surfs:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "surf-ace.obj", "Wavefront OBJ (*.obj)")
        if path:
            fileio.export_obj(path, surfs, resolution=self.settings.resolution)
            self.statusBar().showMessage(f"Exported {len(surfs)} surface(s) to {path}", 5000)

    def closeEvent(self, event):
        if self._confirm_discard():
            self.viewer.vtk.Finalize()
            event.accept()
        else:
            event.ignore()


def run(argv=None):
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Surf-Ace")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()
