# Mostly created with AI, still need to review what it created
# and make edits

# load_data.py
# Minimal helpers to read mixed-format files, normalize units, and
# resample to a common 1-nm grid so curves can be multiplied/integrated.

# data_files/load_data.py
from __future__ import annotations
import re
import numpy as np
import os

# Common 1-nm grid
COMMON_GRID_NM = np.arange(300.0, 1201.0, 1.0)

_NUM_RE = re.compile(r"^[\s]*[+-]?\d")

def percent_to_fraction(y):
    return np.asarray(y, float) / 100.0

def _to_nm(x, unit="nm"):
    unit = (unit or "nm").lower()
    x = np.asarray(x, float)
    if unit in ("nm",): return x
    if unit in ("um","µm","micron","microns"): return x*1000.0
    if unit in ("a","å","angstrom","angstroms","ang"): return x*0.1
    raise ValueError(f"Unsupported wavelength unit: {unit}")

def _read_two_columns_any(path: str):
    skip = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if _NUM_RE.search(line): break
            skip += 1
    # guess delimiter
    with open(path, "r", errors="ignore") as f:
        for _ in range(skip): next(f, None)
        first = next(f, "").strip()
    delim = "," if ("," in first) else None
    arr = np.loadtxt(path, delimiter=delim, skiprows=skip)
    if arr.ndim == 1:
        x = np.arange(arr.size, dtype=float); y = arr.astype(float)
    else:
        x = arr[:,0].astype(float); y = arr[:,1].astype(float)
    return x, y

def resample_to_common_nm_grid(wl_nm, values, left=0.0, right=0.0, grid_nm=None):
    if grid_nm is None: grid_nm = COMMON_GRID_NM
    wl_nm = np.asarray(wl_nm, float); values = np.asarray(values, float)
    idx = np.argsort(wl_nm)
    wl_nm = wl_nm[idx]; values = values[idx]
    out = np.interp(grid_nm, wl_nm, values, left=left, right=right)
    return grid_nm, out

def _looks_like_percent(y):
    y = np.asarray(y, float)
    if not np.isfinite(y).any(): return False
    return (1.0 <= np.nanmax(y) <= 100.0) and (np.mean(y > 1.0) > 0.2)

# --- Specific loaders ---

def load_tapas_atmosphere(path: str):
    wl, T = _read_two_columns_any(path)
    wl_nm = _to_nm(wl, "nm")
    return wl_nm, np.clip(T, 0.0, 1.0)

def load_fraction_curve(path: str, wav_unit="nm", y_is_percent: bool | None = None):
    wl, y = _read_two_columns_any(path)
    wl_nm = _to_nm(wl, wav_unit)
    if y_is_percent is None: y_is_percent = _looks_like_percent(y)
    y = percent_to_fraction(y) if y_is_percent else np.asarray(y, float)
    return wl_nm, np.clip(y, 0.0, 1.0)

# Solar: irradiance -> photons
_H = 6.62607015e-34
_C = 2.99792458e8

def load_solar_irradiance(path: str, wav_unit="um", E_unit="W/m^2/um", to_photons=True):
    wl, E = _read_two_columns_any(path)
    wl_nm = _to_nm(wl, wav_unit)
    E = np.asarray(E, float)
    eu = E_unit.lower().replace(" ", "")
    if eu in ("w/m^2/um","w_m2_um"):
        E_per_nm = E/1000.0
    elif eu in ("w/m^2/nm","w_m2_nm"):
        E_per_nm = E
    else:
        raise ValueError(f"Unsupported E_unit: {E_unit}")
    if to_photons:
        lam_m = wl_nm*1e-9
        photons = E_per_nm * (lam_m/(_H*_C))
        return wl_nm, photons
    return wl_nm, E_per_nm

# --- Bulk load + interpolators ---

def load_all_curves(base_dir="data_files", paths=None, to_photons=True, grid_nm=None):
    if grid_nm is None: grid_nm = COMMON_GRID_NM
    defaults = {
        "sun":     "solar spectrum - e490_00a_amo.csv",
        "asteroid":"asteroid spectrum - 433 Eros.txt",
        "atm":     "atmospheric transmission - data.txt",
        "qe":      "CCD efficiency - SBIG Aluma CCD77-00 broadband.txt",
        "mirror":  "mirror reflectivity - silver.csv",
    }
    if paths: defaults.update(paths)
    def p(name):
        fp = os.path.join(base_dir, name)
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        return fp

    wl_sun, sun = load_solar_irradiance(p(defaults["sun"]), wav_unit="um", E_unit="W/m^2/um", to_photons=to_photons)
    wl_ast, ast = load_fraction_curve(p(defaults["asteroid"]), wav_unit="um")
    wl_atm, atm = load_tapas_atmosphere(p(defaults["atm"]))
    wl_qe,  qe  = load_fraction_curve(p(defaults["qe"]), wav_unit="nm")
    wl_mr,  mr  = load_fraction_curve(p(defaults["mirror"]), wav_unit="nm")

    lam, sun = resample_to_common_nm_grid(wl_sun, sun, left=0.0, right=0.0, grid_nm=grid_nm)
    _,   ast = resample_to_common_nm_grid(wl_ast, ast, left=0.0, right=0.0, grid_nm=lam)
    _,   atm = resample_to_common_nm_grid(wl_atm, atm, left=0.0, right=0.0, grid_nm=lam)
    _,   qe  = resample_to_common_nm_grid(wl_qe,  qe,  left=0.0, right=0.0, grid_nm=lam)
    _,   mr  = resample_to_common_nm_grid(wl_mr,  mr,  left=0.0, right=0.0, grid_nm=lam)

    curves = {"sun": sun, "asteroid": ast, "atm": atm, "qe": qe, "mirror": mr}
    return lam, curves

def make_curve_interpolators(lam_nm, curves):
    lam_nm = np.asarray(lam_nm, float)
    def mk(key):
        arr = np.asarray(curves[key], float)
        return lambda x: np.interp(np.asarray(x, float), lam_nm, arr, left=0.0, right=0.0)
    return {k: mk(k) for k in ("sun","asteroid","atm","qe","mirror")}
