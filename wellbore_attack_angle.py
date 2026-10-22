#!/usr/bin/env python3
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

Requirements: numpy, pandas, scipy, openpyxl      (pip install numpy pandas scipy openpyxl)

Example
-------
    python wellbore_attack_angle.py data --out attack_angles.xlsx
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import (LinearNDInterpolator, PchipInterpolator,
                               RegularGridInterpolator)
from scipy.optimize import brentq
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
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
        wanted = {n.strip().lower() for n in only.split(",") if n.strip()}
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


def main():
    ap = argparse.ArgumentParser(
        description="Attack angle between wellbores and surfaces - all wells against all surfaces")
    ap.add_argument("data_dir", help="folder containing the wells and surfaces subfolders")
    ap.add_argument("--wells-dir", help="override wells folder (default: subfolder named *well*/*dev*/*survey*)")
    ap.add_argument("--surfaces-dir", help="override surfaces folder (default: subfolder named *surf*/*horizon*)")
    ap.add_argument("--only-wells", help="comma separated well file names (without extension) to include; default all")
    ap.add_argument("--only-surfaces", help="comma separated surface file names (without extension) to include; default all")
    ap.add_argument("--wellhead-file", help="CSV with columns well,x,y,kb (if not in the survey files)")
    ap.add_argument("--well-columns", help="comma separated column names for header-less survey files")
    ap.add_argument("--surface-z", choices=["auto", "up", "down"], default="auto",
                    help="surface Z convention: up=elevation, down=depth positive (default auto)")
    ap.add_argument("--step", type=float, default=1.0, help="MD sampling step in metres (default 1.0)")
    ap.add_argument("--interp", choices=["pchip", "linear"], default="pchip",
                    help="interpolation of the well path between stations (default pchip)")
    ap.add_argument("--horizontal-incl", type=float, default=80.0,
                    help="inclination (deg) at or above which a crossing is labelled horizontal (default 80)")
    ap.add_argument("--out", default="attack_angles.xlsx", help="output Excel file (default attack_angles.xlsx)")
    args = ap.parse_args()

    wells_dir = Path(args.wells_dir) if args.wells_dir else find_subdir(args.data_dir, ("well", "dev", "survey"))
    surf_dir = Path(args.surfaces_dir) if args.surfaces_dir else find_subdir(args.data_dir, ("surf", "horizon"))
    if not wells_dir or not wells_dir.is_dir():
        sys.exit("Could not find the wells folder; use --wells-dir")
    if not surf_dir or not surf_dir.is_dir():
        sys.exit("Could not find the surfaces folder; use --surfaces-dir")
    print(f"Wells folder:    {wells_dir}\nSurfaces folder: {surf_dir}")

    wellheads = read_wellhead_file(args.wellhead_file) if args.wellhead_file else {}
    default_cols = args.well_columns.split(",") if args.well_columns else None

    # ---- wells: read all, sample each path once ------------------------------
    wells, used = [], set()
    for p in list_files(wells_dir, args.only_wells):
        try:
            w = build_well(p, wellheads, default_cols)
            w.name = unique_name(w.name, used)
            wells.append(w)
            print(f"  well:    {w.name}  ({len(w.md)} stations, MD {w.md[0]:.0f}-{w.md[-1]:.0f})")
        except Exception as e:
            print(f"  SKIPPED well file {p.name}: {e}", file=sys.stderr)
    surf_files = list_files(surf_dir, args.only_surfaces)
    if not wells or not surf_files:
        sys.exit("No usable wells or surface files were found.")
    print(f"\n{len(wells)} well(s) x {len(surf_files)} surface file(s)")

    dense = {w.name: densify(w, args.step, args.interp) for w in wells}
    well_z_med = float(np.median(np.concatenate([w.z for w in wells])))

    # ---- surfaces: one at a time (keeps memory low), against every well -----
    results, cover, surf_depth, used = [], [], {}, set()
    for p in surf_files:
        try:
            s = load_surface(p)
        except Exception as e:
            print(f"  SKIPPED surface file {p.name}: {e}", file=sys.stderr)
            continue
        s.name = unique_name(s.name, used)

        if args.surface_z == "up":
            s.sign = 1.0
        elif args.surface_z == "down":
            s.sign = -1.0
        else:
            sm = s.median_z()
            s.sign = -1.0 if (np.isfinite(sm) and sm * well_z_med < 0) else 1.0
            if s.sign < 0:
                print(f"  note: Z of surface '{s.name}' treated as depth (sign flipped)")
        surf_depth[s.name] = s.sign * s.median_z()

        n_cross = 0
        for w in wells:
            rows, d = crossings(w, dense[w.name], s, args.horizontal_incl)
            results.extend(rows)
            cover.append(pair_summary(w, s, rows, d))
            n_cross += len(rows)
        print(f"  surface: {s.name:<30s} {n_cross} crossing(s)")
        del s

    if not cover:
        sys.exit("No surfaces could be read.")

    # ---- tables ---------------------------------------------------------------
    order = sorted(surf_depth, key=lambda n: -surf_depth[n] if np.isfinite(surf_depth[n]) else np.inf)
    rank = {n: i for i, n in enumerate(order)}
    well_order = [w.name for w in wells]

    coverage = pd.DataFrame(cover)
    coverage["_s"] = coverage["surface"].map(rank)
    coverage = (coverage.sort_values(["well", "_s"]).drop(columns="_s").reset_index(drop=True))

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if len(df):
        df["_s"] = df["surface"].map(rank)
        df = df.sort_values(["well", "md"]).drop(columns="_s").reset_index(drop=True)

    matrix = (coverage.pivot(index="well", columns="surface", values="attack_angle_first_deg")
              .reindex(index=well_order, columns=order))

    out = Path(args.out)
    if out.suffix.lower() != ".xlsx":
        out = out.with_suffix(".xlsx")
    write_excel(df, coverage, matrix, out)

    # ---- console summary -----------------------------------------------------
    print("\nAttack angle (deg) of first crossing, wells x surfaces (shallow -> deep):")
    print(matrix.round(1).fillna("-").to_string(max_cols=12))
    print("\nStatus counts:")
    print(coverage["status"].value_counts().to_string())
    if not len(df):
        print("\nNo intersections found. Check coordinates/Z conventions (--surface-z).")
    print(f"\n{len(df)} crossing(s) written to {out}")


if __name__ == "__main__":
    main()
