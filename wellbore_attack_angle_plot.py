"""
wellbore_attack_angle.py
========================

Calculates the angle at which wellbores cut through surfaces (horizons, faults
represented as surfaces, etc.).

Expected folder layout
----------------------
    data/
      wells/        one ASCII deviation survey file per well (*.txt, *.csv, *.dev ...)
      surfaces/     one ASCII surface file per surface (*.txt, *.xyz, *.irap, *.asc ...)

Supported well files
--------------------
1. Petrel "Well path/deviation (ASCII)" export. Header lines start with '#', and a
   column line such as:  MD X Y Z TVD DX DY AZIM INCL DLS
   -> the X, Y, Z trajectory is used directly.
2. Any text/CSV file with a header containing MD, INCL (or INC) and AZIM (or AZI).
   The trajectory is calculated with the minimum curvature method. The wellhead
   X, Y and KB are taken from Petrel-style header lines
   ('# WELL HEAD X-COORDINATE: ...', '# WELL HEAD Y-COORDINATE: ...', '# WELL KB: ...')
   or from --wellhead-file (CSV with columns: well,x,y,kb).
3. Header-less files with exactly 3 columns are read as MD, INCL, AZIM
   (override with --well-columns).

Multiple wells / multiple surfaces
----------------------------------
Every well in the wells folder is tested against every surface in the surfaces
folder. Each surface is loaded once, tested against all wells and released, so
many large horizons can be processed. Use --only-wells / --only-surfaces to
restrict the run to named files. The Excel output contains the crossings, a
well x surface matrix and a coverage sheet listing the pairs without a crossing.

Supported surface files
-----------------------
1. XYZ points: the first three numeric columns are X, Y, Z. Regular grids are
   detected automatically (bilinear interpolation); scattered points are
   triangulated (Delaunay, linear interpolation).
2. Irap Classic ASCII grid (as exported by Petrel/RMS, file starts with -996).
   Grid rotation is not supported (a warning is printed).

Figures
-------
One section per well along its path (x = horizontal distance from the wellhead,
y = Z), the horizons sampled along the same path, the crossings as dots and the
TRUE 3D attack angle annotated at every crossing.

Definitions of the results
--------------------------
attack_angle_deg   Angle between the wellbore tangent and the surface PLANE.
                   0 deg  = wellbore parallel to the surface
                   90 deg = wellbore perpendicular to the surface
angle_to_normal    Angle between wellbore tangent and surface normal (= 90 - attack angle).
direction          'down-section' if the well passes from above to below the surface
                   going down-hole (increasing MD), otherwise 'up-section'.

Z convention
------------
Internally everything is handled as ELEVATION (Z positive up, so depth is
negative). Well Z is normalised automatically. For surfaces use --surface-z:
  auto  (default) flips the sign of the surface if its median Z has the opposite
                  sign of the well's median Z
  up    surface Z is elevation (negative at depth)
  down  surface Z is depth (positive downwards)

Requirements: numpy, pandas, scipy, openpyxl, matplotlib
              (pip install numpy pandas scipy openpyxl matplotlib)

Use in Jupyter
--------------
Paste the whole script into one notebook cell, or split it at the '# %%' markers
(each marker = one cell). Edit the CONFIG block in the LAST cell (DATA_DIR etc.)
and run. Run the cells above it once first. Results:
  - one figure per well, shown in the notebook and saved as PNG in FIG_DIR
  - an Excel workbook (crossings, well x horizon matrix, coverage, notes)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import (LinearNDInterpolator, PchipInterpolator,
                               RegularGridInterpolator)
from scipy.optimize import brentq
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# %% ------------------------------------------------------------------------ #
# Helpers
# --------------------------------------------------------------------------- #

NULL_EXACT = (-999.0, -999.25, -9999.0)
NULL_ABS_LIMIT = 1.0e5          # |z| >= 1e5 is treated as undefined (e.g. 9999900, 1e30)

COLUMN_ALIASES = {
    "md":    {"md", "mdkb", "mdrkb", "mdbrt", "measureddepth"},
    "incl":  {"incl", "inclination", "inc", "incldeg", "dev", "deviation"},
    "azim":  {"azim", "azimuth", "az", "azi", "azimdeg"},
    "x":     {"x", "xcoord", "easting", "xutm"},
    "y":     {"y", "ycoord", "northing", "yutm"},
    "z":     {"z", "zsubsea", "zss"},
    "tvd":   {"tvd", "tvdkb", "tvdrkb"},
    "tvdss": {"tvdss"},
}
ALIAS_LOOKUP = {a: k for k, v in COLUMN_ALIASES.items() for a in v}


def _norm(token):
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def _split(line):
    return [t for t in re.split(r"[,;\s]+", line.strip()) if t != ""]


# %% ------------------------------------------------------------------------ #
# Wells
# --------------------------------------------------------------------------- #

class Well:
    def __init__(self, name, md, x, y, z):
        self.name = name
        self.md = np.asarray(md, float)
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.z = np.asarray(z, float)          # elevation, positive up


def read_wellhead_file(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    heads = {}
    for _, r in df.iterrows():
        heads[str(r["well"]).strip().lower()] = (
            float(r["x"]), float(r["y"]), float(r.get("kb", 0.0) or 0.0))
    return heads


def read_well_file(path, default_columns=None):
    """Parse a deviation file -> (name, dict of column arrays, meta dict)."""
    meta = {}
    header = None
    rows = []
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            is_comment = s.startswith("#") or s.startswith("//")
            body = s.lstrip("#/").strip()

            if is_comment:
                m = re.match(r"WELL\s*NAME\s*:\s*(.+)", body, re.I)
                if m:
                    meta["name"] = m.group(1).strip()
                m = re.match(r"WELL\s*HEAD\s*X\S*\s*:\s*([-+\d.eE]+)", body, re.I)
                if m:
                    meta["x"] = float(m.group(1))
                m = re.match(r"WELL\s*HEAD\s*Y\S*\s*:\s*([-+\d.eE]+)", body, re.I)
                if m:
                    meta["y"] = float(m.group(1))
                m = re.match(r"WELL\s*KB\s*:\s*([-+\d.eE]+)", body, re.I)
                if m:
                    meta["kb"] = float(m.group(1))

            toks = _split(body)
            if not toks:
                continue

            if _is_number(toks[0]):
                if not is_comment:
                    try:
                        rows.append([float(t) for t in toks])
                    except ValueError:
                        pass
            elif header is None:
                keys = [ALIAS_LOOKUP.get(_norm(t)) for t in toks]
                if sum(k is not None for k in keys) >= 2:
                    header = keys

    if not rows:
        raise ValueError("no numeric data rows found")

    ncol = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    rows = np.array([r for r in rows if len(r) == ncol])

    if header is None:
        if default_columns:
            header = [ALIAS_LOOKUP.get(_norm(c)) for c in default_columns]
        elif ncol == 3:
            header = ["md", "incl", "azim"]
        else:
            raise ValueError("no recognisable header; use --well-columns")
    header = header[:ncol]

    cols = {}
    for i, k in enumerate(header):
        if k is not None and k not in cols:
            cols[k] = rows[:, i]

    name = meta.get("name") or Path(path).stem
    return name, cols, meta


def min_curvature(md, incl_deg, azim_deg, x0, y0, z0):
    """Minimum curvature trajectory. Returns x, y, z (z = elevation, positive up)."""
    md = np.asarray(md, float)
    if md[0] > 0:                                # assume vertical from reference to 1st station
        md = np.insert(md, 0, 0.0)
        incl_deg = np.insert(incl_deg, 0, 0.0)
        azim_deg = np.insert(azim_deg, 0, 0.0)
    inc = np.radians(incl_deg)
    az = np.radians(azim_deg)
    dmd = np.diff(md)
    i0, i1, a0, a1 = inc[:-1], inc[1:], az[:-1], az[1:]
    cos_dl = np.cos(i1 - i0) - np.sin(i0) * np.sin(i1) * (1 - np.cos(a1 - a0))
    dl = np.arccos(np.clip(cos_dl, -1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        rf = np.where(dl < 1e-9, 1.0, 2.0 / dl * np.tan(dl / 2.0))
    dn = dmd / 2 * (np.sin(i0) * np.cos(a0) + np.sin(i1) * np.cos(a1)) * rf
    de = dmd / 2 * (np.sin(i0) * np.sin(a0) + np.sin(i1) * np.sin(a1)) * rf
    dv = dmd / 2 * (np.cos(i0) + np.cos(i1)) * rf
    x = np.concatenate([[x0], x0 + np.cumsum(de)])
    y = np.concatenate([[y0], y0 + np.cumsum(dn)])
    z = np.concatenate([[z0], z0 - np.cumsum(dv)])
    return md, x, y, z


def build_well(path, wellheads, default_columns=None):
    name, cols, meta = read_well_file(path, default_columns)
    key = name.strip().lower()
    stem = Path(path).stem.lower()
    head = wellheads.get(key) or wellheads.get(stem)
    kb = meta.get("kb", head[2] if head else 0.0)

    has_xy = "x" in cols and "y" in cols
    has_vert = any(k in cols for k in ("z", "tvd", "tvdss"))

    if has_xy and has_vert:
        x, y = cols["x"], cols["y"]
        if "z" in cols:
            z = cols["z"].copy()
            if "tvd" in cols and len(z) > 2 and np.corrcoef(z, cols["tvd"])[0, 1] > 0:
                z = -z                           # Z was given as depth -> elevation
        elif "tvdss" in cols:
            z = -cols["tvdss"]
        else:
            z = kb - cols["tvd"]
        if "md" in cols:
            md = cols["md"]
        else:
            md = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))])
    elif all(k in cols for k in ("md", "incl", "azim")):
        if "x" in meta and "y" in meta:
            x0, y0 = meta["x"], meta["y"]
        elif head:
            x0, y0 = head[0], head[1]
        else:
            raise ValueError("no wellhead coordinates found (use header lines or --wellhead-file)")
        md, x, y, z = min_curvature(cols["md"], cols["incl"], cols["azim"], x0, y0, kb)
    else:
        raise ValueError("need X,Y,Z/TVD columns or MD,INCL,AZIM columns")

    order = np.argsort(md)
    md, x, y, z = md[order], x[order], y[order], z[order]
    keep = np.concatenate([[True], np.diff(md) > 1e-9])
    return Well(name, md[keep], x[keep], y[keep], z[keep])


# %% ------------------------------------------------------------------------ #
# Surfaces
# --------------------------------------------------------------------------- #

class Surface:
    def __init__(self, name, interp, h):
        self.name = name
        self.interp = interp
        self.h = h              # step used for finite-difference gradient
        self.sign = 1.0

    def z(self, x, y):
        pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        return self.sign * np.asarray(self.interp(pts), float)

    def median_z(self):
        return float(np.nanmedian(self.interp.values))

    def gradient(self, x, y):
        h = self.h
        sz = lambda px, py: float(self.z(px, py)[0])
        z0 = sz(x, y)

        def deriv(zp, zm):
            if np.isfinite(zp) and np.isfinite(zm):
                return (zp - zm) / (2 * h)
            if np.isfinite(zp) and np.isfinite(z0):
                return (zp - z0) / h
            if np.isfinite(zm) and np.isfinite(z0):
                return (z0 - zm) / h
            return np.nan

        return (deriv(sz(x + h, y), sz(x - h, y)),
                deriv(sz(x, y + h), sz(x, y - h)))


def _clean_z(z):
    z = np.asarray(z, float).copy()
    bad = (np.abs(z) >= NULL_ABS_LIMIT) | np.isin(z, NULL_EXACT)
    z[bad] = np.nan
    return z


def load_irap(path, name):
    with open(path, "r", errors="ignore") as f:
        tok = f.read().split()
    ny = int(tok[1]); xinc = float(tok[2]); yinc = float(tok[3])
    xmin = float(tok[4]); ymin = float(tok[6])
    nx = int(tok[8]); rot = float(tok[9])
    if abs(rot) > 1e-6:
        print(f"  WARNING: {name}: grid rotation {rot} deg is not supported; "
              f"results may be wrong", file=sys.stderr)
    vals = _clean_z(np.array(tok[19:19 + nx * ny], float))
    Z = vals.reshape(ny, nx).T                      # Z[i (x), j (y)]
    xs = xmin + xinc * np.arange(nx)
    ys = ymin + yinc * np.arange(ny)
    interp = RegularGridInterpolator((xs, ys), Z, bounds_error=False, fill_value=np.nan)
    return Surface(name, interp, h=min(abs(xinc), abs(yinc)))


def _read_xyz(path):
    try:
        arr = np.loadtxt(path, comments="#", usecols=(0, 1, 2))
        return np.atleast_2d(arr)
    except Exception:
        pass
    out = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            t = _split(line)
            if len(t) >= 3 and _is_number(t[0]) and _is_number(t[1]) and _is_number(t[2]):
                out.append((float(t[0]), float(t[1]), float(t[2])))
    return np.array(out)


def load_xyz(path, name):
    arr = _read_xyz(path)
    if arr.size == 0:
        raise ValueError("no XYZ data found")
    x, y, z = arr[:, 0], arr[:, 1], _clean_z(arr[:, 2])
    xr, yr = np.round(x, 3), np.round(y, 3)
    xs, ys = np.unique(xr), np.unique(yr)

    if len(xs) > 1 and len(ys) > 1 and len(xs) * len(ys) <= 1.2 * len(x):
        Z = np.full((len(xs), len(ys)), np.nan)
        Z[np.searchsorted(xs, xr), np.searchsorted(ys, yr)] = z
        interp = RegularGridInterpolator((xs, ys), Z, bounds_error=False, fill_value=np.nan)
        h = min(np.median(np.diff(xs)), np.median(np.diff(ys)))
        return Surface(name, interp, h=float(h))

    good = np.isfinite(z)
    pts = np.column_stack([x[good], y[good]])
    interp = LinearNDInterpolator(pts, z[good])
    sample = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 50000), replace=False)]
    nn = cKDTree(pts).query(sample, k=2)[0][:, 1]
    return Surface(name, interp, h=float(np.median(nn)))


def load_surface(path):
    name = Path(path).stem
    with open(path, "r", errors="ignore") as f:
        first = f.read(300).split()
    if first and first[0] == "-996":
        return load_irap(path, name)
    return load_xyz(path, name)


# %% ------------------------------------------------------------------------ #
# Intersection and angle calculation
# --------------------------------------------------------------------------- #

def densify(well, step, method="pchip"):
    md = np.append(np.arange(well.md[0], well.md[-1], step), well.md[-1])
    if method == "pchip" and len(well.md) >= 3:
        fx, fy, fz = (PchipInterpolator(well.md, a) for a in (well.x, well.y, well.z))
        return md, fx(md), fy(md), fz(md)
    return (md, np.interp(md, well.md, well.x),
            np.interp(md, well.md, well.y), np.interp(md, well.md, well.z))


def crossings(well, dense, surf, horizontal_incl=80.0):
    """All crossings of one well with one surface. Returns (rows, d) where d is
    the signed vertical distance well - surface along the sampled path."""
    md, x, y, z = dense
    d = z - surf.z(x, y)
    ok = np.isfinite(d[:-1]) & np.isfinite(d[1:])
    cross = ok & ((d[:-1] * d[1:] < 0) | ((d[1:] == 0) & (d[:-1] != 0)))
    results = []

    for i in np.where(cross)[0]:
        def f(t, i=i):
            xi = x[i] + t * (x[i + 1] - x[i])
            yi = y[i] + t * (y[i + 1] - y[i])
            zi = z[i] + t * (z[i + 1] - z[i])
            return zi - float(surf.z(xi, yi)[0])

        f0, f1 = d[i], d[i + 1]
        if f0 == 0:
            t = 0.0
        elif f1 == 0:
            t = 1.0
        else:
            try:
                t = brentq(f, 0.0, 1.0, xtol=1e-9)
            except ValueError:
                t = f0 / (f0 - f1)

        xc = x[i] + t * (x[i + 1] - x[i])
        yc = y[i] + t * (y[i + 1] - y[i])
        zc = z[i] + t * (z[i + 1] - z[i])
        mdc = md[i] + t * (md[i + 1] - md[i])

        tvec = np.array([x[i + 1] - x[i], y[i + 1] - y[i], z[i + 1] - z[i]])
        tvec /= np.linalg.norm(tvec)             # unit tangent, direction of increasing MD

        dzdx, dzdy = surf.gradient(xc, yc)
        if not (np.isfinite(dzdx) and np.isfinite(dzdy)):
            continue
        normal = np.array([-dzdx, -dzdy, 1.0])
        normal /= np.linalg.norm(normal)

        sin_attack = np.clip(abs(np.dot(tvec, normal)), 0.0, 1.0)
        attack = np.degrees(np.arcsin(sin_attack))

        incl = np.degrees(np.arccos(np.clip(-tvec[2], -1, 1)))
        if incl >= horizontal_incl:
            well_type = "horizontal"
        elif incl >= 10.0:
            well_type = "deviated"
        else:
            well_type = "vertical"

        results.append({
            "well": well.name,
            "surface": surf.name,
            "md": mdc,
            "x": xc,
            "y": yc,
            "z_elevation": zc,
            "well_incl_deg": np.degrees(np.arccos(np.clip(-tvec[2], -1, 1))),
            "well_azim_deg": np.degrees(np.arctan2(tvec[0], tvec[1])) % 360,
            "surface_dip_deg": np.degrees(np.arctan(np.hypot(dzdx, dzdy))),
            "surface_dip_azim_deg": np.degrees(np.arctan2(-dzdx, -dzdy)) % 360,
            "attack_angle_deg": attack,
            "angle_to_normal_deg": 90.0 - attack,
            "crossing": "from above to below" if d[i] > 0 else "from below to above",
            "well_type": well_type,
        })
    return results, d


RESULT_COLUMNS = ["well", "surface", "md", "x", "y", "z_elevation", "well_incl_deg",
                  "well_azim_deg", "surface_dip_deg", "surface_dip_azim_deg",
                  "attack_angle_deg", "angle_to_normal_deg", "crossing", "well_type"]


def pair_summary(well, surf, rows, d):
    """One line describing what happened for a well/surface combination."""
    inside = np.isfinite(d)
    if not inside.any():
        status = "no overlap (well outside surface extent or surface undefined)"
    elif rows:
        status = "crossed"
    elif np.nanmedian(d[inside]) > 0:
        status = "well stays above surface"
    else:
        status = "well stays below surface"
    angles = [r["attack_angle_deg"] for r in rows]
    return {
        "well": well.name,
        "surface": surf.name,
        "status": status,
        "n_crossings": len(rows),
        "well_path_inside_surface_pct": 100.0 * float(inside.mean()),
        "crossing_sequence": "; ".join(
            "above>below" if r["crossing"] == "from above to below" else "below>above"
            for r in rows),
        "first_md": rows[0]["md"] if rows else np.nan,
        "attack_angle_first_deg": angles[0] if rows else np.nan,
        "attack_angle_min_deg": min(angles) if rows else np.nan,
        "attack_angle_max_deg": max(angles) if rows else np.nan,
    }


# %% ------------------------------------------------------------------------ #
# Output
# --------------------------------------------------------------------------- #

def write_excel(results, coverage, matrix, path):
    """Workbook with: all crossings, a well x surface matrix, coverage per pair, notes."""
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    notes = pd.DataFrame([
        ("Sheet: Attack angles", "One row per crossing between a wellbore and a surface (a well can cross a surface several times)."),
        ("Sheet: Matrix", "Attack angle of the FIRST crossing (lowest MD) for every well (rows) and surface (columns), "
                          "surfaces ordered shallow to deep. Blank = no crossing. Colour scale: red = near parallel, green = near perpendicular."),
        ("Sheet: Coverage", "Status for every well/surface combination, including those without a crossing, "
                            "with number of crossings and min/max attack angle."),
        ("attack_angle_deg", "Angle between wellbore tangent and surface plane (0 = parallel, 90 = perpendicular)"),
        ("angle_to_normal_deg", "Angle between wellbore tangent and surface normal (= 90 - attack angle)"),
        ("crossing", "Side the well approaches the surface from, going down-hole (increasing MD), judged vertically: "
                     "'from above to below' = well is above the surface before the crossing and below it after; "
                     "'from below to above' = the opposite. Typical for horizontal wells that pass through a dipping horizon."),
        ("well_type", "Wellbore inclination at the crossing: horizontal (>= --horizontal-incl, default 80 deg), "
                      "deviated (10 deg up to the horizontal limit) or vertical (< 10 deg)"),
        ("crossing_sequence", "Coverage sheet: order of crossings along the well, e.g. 'above>below; below>above'"),
        ("md", "Measured depth of the intersection"),
        ("x, y, z_elevation", "Intersection coordinates; Z is elevation (positive up, depth is negative)"),
        ("well_incl_deg / well_azim_deg", "Wellbore inclination (from vertical) and azimuth at the intersection"),
        ("surface_dip_deg / surface_dip_azim_deg", "Surface dip and dip azimuth at the intersection"),
        ("well_path_inside_surface_pct", "Share of the sampled well path that lies within the extent of the surface"),
    ], columns=["Item", "Description"])

    arial = Font(name="Arial", size=10)
    head_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="305496")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Attack angles", index=False)
        matrix.to_excel(writer, sheet_name="Matrix", index_label="well")
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

        for ws in writer.book.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    c.font = arial
                    if isinstance(c.value, float):
                        c.number_format = "0.00"
            for c in ws[1]:
                c.font = head_font
                c.fill = head_fill
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for i, col in enumerate(ws.columns, start=1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(i)].width = min(max(width + 3, 12), 90)
            ws.freeze_panes = "B2" if ws.title == "Matrix" else "A2"

        for name in ("Attack angles", "Coverage"):
            ws = writer.book[name]
            ws.auto_filter.ref = ws.dimensions
        wm = writer.book["Matrix"]
        for c in wm["A"][1:]:
            c.font = Font(name="Arial", size=10, bold=True)
        if matrix.shape[0] and matrix.shape[1]:
            rng = f"B2:{get_column_letter(matrix.shape[1] + 1)}{matrix.shape[0] + 1}"
            wm.conditional_formatting.add(rng, ColorScaleRule(
                start_type="num", start_value=0, start_color="F8696B",
                mid_type="num", mid_value=45, mid_color="FFEB84",
                end_type="num", end_value=90, end_color="63BE7B"))


# %% ------------------------------------------------------------------------ #
# Main
# --------------------------------------------------------------------------- #

def find_subdir(root, keywords):
    for p in sorted(Path(root).iterdir()):
        if p.is_dir() and any(k in p.name.lower() for k in keywords):
            return p
    return None


def list_files(folder, only=None):
    files = sorted(p for p in Path(folder).iterdir()
                   if p.is_file() and not p.name.startswith(".")
                   and p.suffix.lower() not in {".py", ".xlsx", ".docx", ".pdf"})
    if only:
        names = only.split(",") if isinstance(only, str) else only
        wanted = {str(n).strip().lower() for n in names if str(n).strip()}
        files = [p for p in files if p.stem.lower() in wanted or p.name.lower() in wanted]
    return files


def unique_name(name, used):
    if name not in used:
        used.add(name)
        return name
    k = 2
    while f"{name}_{k}" in used:
        k += 1
    used.add(f"{name}_{k}")
    return f"{name}_{k}"


# %% ------------------------------------------------------------------------ #
# Figures: one section per well
# --------------------------------------------------------------------------- #

def _text_box(ann, renderer):
    """Pixel box of an annotation's label (text + box padding), without the leader line."""
    from matplotlib.text import Text
    ann.update_positions(renderer)            # resolve the offset position before measuring
    return Text.get_window_extent(ann, renderer).expanded(1.12, 1.30)


def plot_well(well, dense, curves, rows, colors, *, figsize=(13, 6.5),
              z_pad_frac=0.15, show_labels=True):
    """Section along the well path.

    x-axis : horizontal distance along the well path from the wellhead
             (the path is 'unrolled', so azimuth changes do not distort it)
    y-axis : Z (elevation, depth is negative)
    lines  : well path (black) and each horizon sampled along the same path
    dots   : crossings, labelled with the TRUE 3D attack angle (wellbore vs surface plane)
    """
    md, x, y, z = dense
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    z_lo, z_hi = float(np.min(z)), float(np.max(z))
    z_rng = max(z_hi - z_lo, 1.0)
    pad = max(z_pad_frac * z_rng, 25.0)
    ax.set_ylim(z_lo - pad, z_hi + pad)

    # near-vertical well: no horizontal extent -> give the section some width
    vertical = s[-1] < 0.02 * z_rng
    if vertical:
        half = 0.35 * z_rng
        ax.set_xlim(-half, half)
        x_left, x_right = -half, half
    else:
        xpad = 0.03 * s[-1]
        ax.set_xlim(-xpad, s[-1] + 0.10 * s[-1])      # room on the right for labels
        x_left, x_right = 0.0, s[-1]

    # horizons ---------------------------------------------------------------
    for name, zs in curves.items():
        col = colors[name]
        ok = np.isfinite(zs)
        if not ok.any():
            continue
        if vertical:
            zv = float(np.nanmedian(zs))
            ax.hlines(zv, x_left, x_right, colors=col, lw=1.8)
            lx, lz, ha = x_right, zv, "right"
        else:
            ax.plot(np.where(ok, s, np.nan), np.where(ok, zs, np.nan), color=col, lw=1.8)
            last = np.where(ok)[0][-1]
            lx, lz, ha = s[last], zs[last], "left"
        if show_labels:
            ax.annotate(name, (lx, lz), xytext=(4 if ha == "left" else -4, 3),
                        textcoords="offset points", ha=ha, va="bottom",
                        fontsize=9, color=col, fontweight="bold",
                        clip_on=True)

    # well path --------------------------------------------------------------
    if vertical:
        ax.plot(np.zeros_like(z), z, color="black", lw=2.6, solid_capstyle="round", zorder=5)
        s_plot = np.zeros_like(s)
    else:
        ax.plot(s, z, color="black", lw=2.6, solid_capstyle="round", zorder=5)
        s_plot = s
    ax.plot(s_plot[0], z[0], marker="^", ms=11, color="black", zorder=6)
    ax.annotate("wellhead", (s_plot[0], z[0]), xytext=(8, 6), textcoords="offset points", fontsize=9)
    ax.plot(s_plot[-1], z[-1], marker="s", ms=7, color="black", zorder=6)
    ax.annotate(f"TD  MD {md[-1]:.0f}", (s_plot[-1], z[-1]), xytext=(-10, 10),
                textcoords="offset points", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # crossings: dots first, then labels placed so that they do not overlap -----
    pts = []
    for r in rows:
        sx = 0.0 if vertical else float(np.interp(r["md"], md, s))
        sz = r["z_elevation"]
        col = colors[r["surface"]]
        ax.plot(sx, sz, "o", ms=9, mfc=col, mec="black", mew=1.4, zorder=8)
        pts.append((sx, sz, col, r))

    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        renderer = fig._get_renderer()
    axbox = ax.get_window_extent(renderer).expanded(0.985, 0.985)
    from matplotlib.transforms import Bbox
    placed = []                                   # boxes already used (pixels): dots + labels
    for sx, sz, _, _ in pts:
        px, py = ax.transData.transform((sx, sz))
        placed.append(Bbox([[px - 9, py - 9], [px + 9, py + 9]]))
    candidates = [(32, 42), (-32, 42), (32, -42), (-32, -42), (70, 80), (-70, 80),
                  (70, -80), (-70, -80), (0, 100), (0, -100), (110, 30), (-110, 30),
                  (110, -30), (-110, -30), (60, 130), (-60, -130)]
    for sx, sz, col, r in pts:
        side = "above\u2192below" if r["crossing"] == "from above to below" else "below\u2192above"
        txt = f"{r['surface']}\n{r['attack_angle_deg']:.1f}\u00b0  ({side})\nMD {r['md']:.0f}"
        best = None
        for dx, dy in candidates:
            ann = ax.annotate(
                txt, (sx, sz), xytext=(dx, dy), textcoords="offset points",
                ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center",
                fontsize=8.5, zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.4, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=col, lw=1.2))
            bb = _text_box(ann, renderer)
            inside = (bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and
                      bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1)
            free = not any(bb.overlaps(o) for o in placed)
            if inside and free:
                best = (ann, bb)
                break
            ann.remove()
        if best is None:                            # nothing fits: take first candidate
            dx, dy = candidates[0]
            ann = ax.annotate(
                txt, (sx, sz), xytext=(dx, dy), textcoords="offset points", ha="left", va="center",
                fontsize=8.5, zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.4, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=col, lw=1.2))
            best = (ann, _text_box(ann, renderer))
        placed.append(best[1])

    if vertical:
        ax.set_xticks([])
        ax.set_xlabel("Vertical well (horizontal axis not to scale)")
    else:
        ax.set_xlabel("Horizontal distance along well path from wellhead (m)")
    ax.set_ylabel("Z, elevation (m)  \u2013  depth is negative")
    ax.grid(True, alpha=0.25)

    fig.canvas.draw()
    bb = ax.get_window_extent()
    xs, zs_ = ax.get_xlim(), ax.get_ylim()
    ve = ((bb.height / (zs_[1] - zs_[0])) / (bb.width / (xs[1] - xs[0]))) if not vertical else np.nan
    sub = f"{len(rows)} crossing(s)"
    if np.isfinite(ve):
        sub += f"   |   vertical exaggeration \u2248 {ve:.1f}\u00d7"
    ax.set_title(f"Well {well.name}  \u2013  attack angle at horizon crossings\n{sub}", fontsize=12)
    ax.text(0.995, 0.99,
            "Labels = true 3D angle between wellbore and surface plane "
            "(0\u00b0 parallel, 90\u00b0 perpendicular).\nSection follows the well path.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="0.35")
    return fig


# %% ------------------------------------------------------------------------ #
# Driver
# --------------------------------------------------------------------------- #

def run(data_dir, *, wells_dir=None, surfaces_dir=None, only_wells=None, only_surfaces=None,
        wellhead_file=None, well_columns=None, surface_z="auto", step=1.0, interp="pchip",
        horizontal_incl=80.0, out_xlsx="attack_angles.xlsx", fig_dir="attack_angle_figures",
        show_figs=True, fig_dpi=150):
    """Run all wells against all surfaces. Returns (crossings, coverage, matrix, figures)."""
    wd = Path(wells_dir) if wells_dir else find_subdir(data_dir, ("well", "dev", "survey"))
    sd = Path(surfaces_dir) if surfaces_dir else find_subdir(data_dir, ("surf", "horizon"))
    if not wd or not wd.is_dir():
        raise FileNotFoundError("Could not find the wells folder; set WELLS_DIR")
    if not sd or not sd.is_dir():
        raise FileNotFoundError("Could not find the surfaces folder; set SURFACES_DIR")
    print(f"Wells folder:    {wd}\nSurfaces folder: {sd}")

    wellheads = read_wellhead_file(wellhead_file) if wellhead_file else {}
    default_cols = list(well_columns) if well_columns else None

    # ---- wells ----------------------------------------------------------------
    wells, used = [], set()
    for p in list_files(wd, only_wells):
        try:
            w = build_well(p, wellheads, default_cols)
            w.name = unique_name(w.name, used)
            wells.append(w)
            print(f"  well:    {w.name}  ({len(w.md)} stations, MD {w.md[0]:.0f}-{w.md[-1]:.0f})")
        except Exception as e:
            print(f"  SKIPPED well file {p.name}: {e}", file=sys.stderr)
    surf_files = list_files(sd, only_surfaces)
    if not wells or not surf_files:
        raise RuntimeError("No usable wells or surface files were found.")
    print(f"\n{len(wells)} well(s) x {len(surf_files)} surface file(s)")

    dense = {w.name: densify(w, step, interp) for w in wells}
    well_z_med = float(np.median(np.concatenate([w.z for w in wells])))

    # ---- surfaces: one at a time, against every well ----------------------------
    results, cover, surf_depth, used = [], [], {}, set()
    curves = {w.name: {} for w in wells}            # horizon Z sampled along each well path
    for p in surf_files:
        try:
            s = load_surface(p)
        except Exception as e:
            print(f"  SKIPPED surface file {p.name}: {e}", file=sys.stderr)
            continue
        s.name = unique_name(s.name, used)

        if surface_z == "up":
            s.sign = 1.0
        elif surface_z == "down":
            s.sign = -1.0
        else:
            sm = s.median_z()
            s.sign = -1.0 if (np.isfinite(sm) and sm * well_z_med < 0) else 1.0
            if s.sign < 0:
                print(f"  note: Z of surface '{s.name}' treated as depth (sign flipped)")
        surf_depth[s.name] = s.sign * s.median_z()

        n_cross = 0
        for w in wells:
            rows, d = crossings(w, dense[w.name], s, horizontal_incl)
            results.extend(rows)
            cover.append(pair_summary(w, s, rows, d))
            curves[w.name][s.name] = dense[w.name][3] - d      # = surface Z along the path
            n_cross += len(rows)
        print(f"  surface: {s.name:<30s} {n_cross} crossing(s)")
        del s

    if not cover:
        raise RuntimeError("No surfaces could be read.")

    # ---- tables ------------------------------------------------------------------
    order = sorted(surf_depth, key=lambda n: -surf_depth[n] if np.isfinite(surf_depth[n]) else np.inf)
    rank = {n: i for i, n in enumerate(order)}
    well_order = [w.name for w in wells]

    coverage = pd.DataFrame(cover)
    coverage["_s"] = coverage["surface"].map(rank)
    coverage = coverage.sort_values(["well", "_s"]).drop(columns="_s").reset_index(drop=True)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if len(df):
        df = df.sort_values(["well", "md"]).reset_index(drop=True)

    matrix = (coverage.pivot(index="well", columns="surface", values="attack_angle_first_deg")
              .reindex(index=well_order, columns=order))

    if out_xlsx:
        out = Path(out_xlsx)
        if out.suffix.lower() != ".xlsx":
            out = out.with_suffix(".xlsx")
        write_excel(df, coverage, matrix, out)
        print(f"\n{len(df)} crossing(s) written to {out}")

    # ---- figures: one per well -------------------------------------------------------
    cmap = plt.get_cmap("tab10" if len(order) <= 10 else "tab20")
    colors = {n: cmap(i % cmap.N) for i, n in enumerate(order)}
    fdir = Path(fig_dir) if fig_dir else None
    if fdir:
        fdir.mkdir(parents=True, exist_ok=True)

    figs = {}
    for w in wells:
        rows = [r for r in results if r["well"] == w.name]
        rows.sort(key=lambda r: r["md"])
        cv = {n: curves[w.name][n] for n in order if n in curves[w.name]}
        fig = plot_well(w, dense[w.name], cv, rows, colors)
        if fdir:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", w.name)
            fig.savefig(fdir / f"{safe}_attack_angles.png", dpi=fig_dpi)
        figs[w.name] = fig
        if show_figs:
            plt.show()
        plt.close(fig)
    if fdir:
        print(f"{len(figs)} figure(s) saved in {fdir.resolve()}")

    return df, coverage, matrix, figs


# %% ------------------------------------------------------------------------ #
# CONFIG - edit this cell and run it
# --------------------------------------------------------------------------- #

DATA_DIR = r"C:\path\to\data"          # folder with the wells and surfaces subfolders

WELLS_DIR = None                       # None = auto-detect subfolder (*well*, *dev*, *survey*)
SURFACES_DIR = None                    # None = auto-detect subfolder (*surf*, *horizon*)
ONLY_WELLS = None                      # e.g. ["V1", "D1"]  (file names without extension); None = all
ONLY_SURFACES = None                   # e.g. ["top_a", "top_b"]; None = all
WELLHEAD_FILE = None                   # CSV with columns well,x,y,kb (if not in the survey files)
WELL_COLUMNS = None                    # e.g. ["MD", "INCL", "AZIM"] for header-less survey files
SURFACE_Z = "auto"                     # "auto", "up" (elevation) or "down" (depth positive)
STEP = 1.0                             # MD sampling step in metres
INTERP = "pchip"                       # "pchip" or "linear"
HORIZONTAL_INCL = 80.0                 # inclination (deg) from which a crossing is called horizontal

OUT_XLSX = "attack_angles.xlsx"        # None = do not write Excel
FIG_DIR = "attack_angle_figures"       # None = do not save PNG files
SHOW_FIGS = True                       # show the figures in the notebook

crossings_df, coverage_df, matrix_df, figures = run(
    DATA_DIR, wells_dir=WELLS_DIR, surfaces_dir=SURFACES_DIR,
    only_wells=ONLY_WELLS, only_surfaces=ONLY_SURFACES,
    wellhead_file=WELLHEAD_FILE, well_columns=WELL_COLUMNS,
    surface_z=SURFACE_Z, step=STEP, interp=INTERP, horizontal_incl=HORIZONTAL_INCL,
    out_xlsx=OUT_XLSX, fig_dir=FIG_DIR, show_figs=SHOW_FIGS)

matrix_df.round(1)        # attack angle (deg) of the first crossing: wells x horizons
