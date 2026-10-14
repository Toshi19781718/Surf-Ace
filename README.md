# Surf-Ace

A Class A surfacing modeler written in Python (PySide6 + VTK + OpenCASCADE).

## Features

- **Bezier / B-spline patches and curves** with draggable control points.
  - Left-drag a control point: move it in the view plane (or along X/Y/Z, see *Drag constraint*).
  - **Ctrl**+drag moves the whole u-row, **Shift**+drag the whole v-column.
  - Exact X/Y/Z entry for the selected point, object translation.
- **Degree & span control**: rebuild any curve/patch to a new degree and span count
  (least-squares refit with fixed end points; exact when raising degree).
- **Continuity matching G0 / G1 / G2 / G3** between any edge of two patches, in any
  orientation. Patch B is modified, patch A is the reference. Along-edge structures are
  made compatible automatically.
- **Continuity check**: max positional gap, max normal deviation and cross-boundary
  normal-curvature difference along the shared edge, with configurable tolerances.
- **Surface analysis**: zebra stripes (camera-relative reflection lines), isophotes,
  Gaussian, mean and max-principal curvature maps with a scalar bar.
- **Curvature combs** on curves and on surface iso-curves.
- **Files**: JSON sessions; STEP and IGES import/export (OpenCASCADE); OBJ export.
- Undo/redo (Ctrl+Z / Ctrl+Y).

## Setup

Requires [Miniforge](https://github.com/conda-forge/miniforge) (conda), because pythonOCC is only
distributed through conda-forge.

```powershell
conda env create -f environment.yml
```

> On Windows, if environment creation fails with a "package cache" error, the package cache path
> is too long. Use a short cache folder: `$env:CONDA_PKGS_DIRS = "C:\Users\<you>\.cpkgs"` and retry.

## Run

```powershell
.\run.bat
```

or, from an activated environment: `python -m surf_ace`.

The app opens with a demo: two hood patches that share an edge with a 21 deg tangent break, and a
feature curve. Switch *Display / Analysis* to **Zebra** to see the break, then on the
**Continuity** tab click **Match B to A** (G2) and watch the stripes flow across the seam.

## Tests

```powershell
conda activate surf-ace
python -m pytest
```

## Layout

```
surf_ace/
  geometry/     basis functions, BSplineCurve, BSplineSurface, refitting
  continuity.py G0-G3 matching and continuity report
  analysis.py   zebra / isophote / curvature fields
  fileio.py     JSON session, OBJ, STEP/IGES via pythonOCC
  scene.py      document model, selection, undo/redo
  ui/           PySide6 main window and VTK viewport
tests/          pytest suite (geometry, continuity, file round-trips)
```

## Current limitations

- Surfaces are non-rational and untrimmed; imported faces lose their trim curves, and rational
  surfaces are approximated.
- The continuity report measures up to G2; G3 matching is exact by construction (verified in tests)
  but not yet measured in the report.
