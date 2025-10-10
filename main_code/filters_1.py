# filters_clean.py
# Purpose: Compute band SNR heatmaps for asteroid spectra with consistent units and a clean pipeline.

import math
import numpy as np
import matplotlib.pyplot as plt
import SMASS as smass

# make project root importable so we can do from data_files import load_data
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # folder that contains data_files/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_files import load_data as ld
print("Using load_data from:", ld.__file__)  # optional sanity print

# ----------------------------
# Load data & build interpolators
# ----------------------------
lam, curves = ld.load_all_curves(base_dir="data_files")
interp = ld.make_curve_interpolators(lam, curves)
f_sun    = interp["sun"]      # photons s^-1 m^-2 nm^-1
f_ast    = interp["asteroid"] # unitless reflectance
f_atm    = interp["atm"]      # unitless
f_qe     = interp["qe"]       # unitless
f_mirror = interp["mirror"]   # unitless

wavelength_nm = lam

# ----------------------------
# Instrument definition
# ----------------------------
def make_instrument(d_m, gain_e_per_photon, s_pix_m, read_noise_e, dark_e_per_pix_s=0.0):
    return {
        "d_m": d_m,
        "gain_e_per_ph": gain_e_per_photon,
        "s_pix_m": s_pix_m,
        "read_noise_e": read_noise_e,
        "dark_e_per_pix_s": dark_e_per_pix_s,
    }

instr = make_instrument(
    d_m=1.0,
    gain_e_per_photon=1.0,
    s_pix_m=5e-6,
    read_noise_e=5.0,
    dark_e_per_pix_s=0.2,
)
t_exp_s = 60.0  # [s]

# ----------------------------
# Per-λ components
# ----------------------------
def spectrum_sun(wavelength_nm):
    return f_sun(wavelength_nm)  # photons s^-1 m^-2 nm^-1

def reflectance_asteroid(wavelength_nm):
    return f_ast(wavelength_nm)  # unitless

def transmission_atm(wavelength_nm):
    return f_atm(wavelength_nm)  # unitless

def transmission_optics(wavelength_nm):
    return f_mirror(wavelength_nm)  # unitless

def ccd_qe(wavelength_nm):
    return f_qe(wavelength_nm)  # unitless

# ----------------------------
# Magnitude-based brightness scaling for the asteroid
# ----------------------------
_H = 6.62607015e-34  # J*s
_C = 2.99792458e8    # m/s

def scale_factor_from_AB_mag(m_AB: float, lambda0_nm: float = 550.0) -> float:
    """Scale K so K*S_sun(λ0)*R(λ0) matches AB magnitude m_AB at λ0."""
    Fnu_AB0 = 3631.0 * 1e-26  # W m^-2 Hz^-1
    lam_m = lambda0_nm * 1e-9
    # photons s^-1 m^-2 nm^-1 at λ0 for AB mag m_AB
    photons_per_nm_target = (10.0 ** (-0.4 * m_AB)) * (Fnu_AB0 / (_H * lam_m)) * 1e-9
    base = spectrum_sun(lambda0_nm) * reflectance_asteroid(lambda0_nm)
    return float(photons_per_nm_target / max(base, 1e-30))

ID_TO_Vmag = {
    "000004": 6.0,   # Vesta (V-type)
    "000006": 8.0,   # Hebe (S-type)
    "000024": 10.0,  # Themis (C-type)
}

# ----------------------------
# Sky background model (AB mag/arcsec^2 -> photons)
# ----------------------------
SKY_SB_MAG_AB = 21.7           # typical dark sky in V, mag/arcsec^2
PLATE_SCALE_AS_PER_PIX = 0.50  # <-- set your true arcsec/pixel if you know it
OMEGA_PIX_AS2 = PLATE_SCALE_AS_PER_PIX ** 2  # arcsec^2 per pixel

def sky_surface_brightness_photons_per_nm(m_AB: float, lambda_nm: np.ndarray) -> np.ndarray:
    """Convert AB mag/arcsec^2 to photons s^-1 m^-2 nm^-1 arcsec^-2."""
    Fnu_AB0 = 3631.0 * 1e-26  # W m^-2 Hz^-1
    lam_m = np.asarray(lambda_nm, float) * 1e-9
    return (10.0 ** (-0.4 * m_AB)) * (Fnu_AB0 / (_H * lam_m)) * 1e-9

def sky_photon_spectrum(wavelength_nm: np.ndarray) -> np.ndarray:
    """Ṡ_sky(λ) photons s^-1 m^-2 nm^-1 arcsec^-2."""
    return sky_surface_brightness_photons_per_nm(SKY_SB_MAG_AB, wavelength_nm)

# ----------------------------
# Per-λ integrands (electrons / nm)
# ----------------------------
def asteroid_flux_raw(wavelength_nm, K_scale: float = 1.0):
    """Ṡ_ast(λ) = K * Ṡ_sun(λ) * R_ast(λ)  [photons s^-1 m^-2 nm^-1]"""
    return K_scale * spectrum_sun(wavelength_nm) * reflectance_asteroid(wavelength_nm)

def _collect_common_factors(instr, t_exp_s):
    area_m2 = math.pi * (instr["d_m"] ** 2) / 4.0  # [m^2]
    gain = instr["gain_e_per_ph"]                  # [e-/photon]
    return area_m2, gain, float(t_exp_s)

def signal_integrand_e_per_nm(wavelength_nm, instr, t_exp_s, K_scale: float = 1.0):
    """Electrons / nm (Eq. 4 integrand)."""
    S_ast = asteroid_flux_raw(wavelength_nm, K_scale)
    Tatm  = transmission_atm(wavelength_nm)
    Topt  = transmission_optics(wavelength_nm)
    QE    = ccd_qe(wavelength_nm)
    area_m2, gain, t_s = _collect_common_factors(instr, t_exp_s)
    return t_s * gain * area_m2 * (S_ast * Tatm * Topt * QE)

def background_integrand_e_per_nm(wavelength_nm, instr, t_exp_s):
    """
    Photon background per-λ from sky surface brightness (per arcsec^2).
    Returns electrons / nm / arcsec^2. Pixel and aperture factors are applied at band level.
    """
    S_sky_as2 = sky_photon_spectrum(wavelength_nm)   # photons s^-1 m^-2 nm^-1 arcsec^-2
    Tatm  = transmission_atm(wavelength_nm)
    Topt  = transmission_optics(wavelength_nm)
    QE    = ccd_qe(wavelength_nm)
    area_m2, gain, t_s = _collect_common_factors(instr, t_exp_s)
    return t_s * gain * area_m2 * (S_sky_as2 * Tatm * Topt * QE)  # e- / nm / arcsec^2

# ----------------------------
# Cumulative integrals (electrons)
# ----------------------------
def cumulative_from_integrand(wavelength_nm, y_e_per_nm):
    dlam_nm = np.gradient(wavelength_nm)
    return np.cumsum(y_e_per_nm * dlam_nm)

# Defer building cumulatives until we know the brightness scale:
gCumSig = None       # electrons (cumulative)
gCumBkg_as2 = None   # electrons / arcsec^2 (cumulative)

def rebuild_cums(K_scale: float):
    """Recompute cumulative signal/background with brightness scaling."""
    global gCumSig, gCumBkg_as2
    sig = signal_integrand_e_per_nm(wavelength_nm, instr, t_exp_s, K_scale)          # e-/nm
    bkg_as2 = background_integrand_e_per_nm(wavelength_nm, instr, t_exp_s)           # e-/nm/arcsec^2
    gCumSig = cumulative_from_integrand(wavelength_nm, sig)                           # e-
    gCumBkg_as2 = cumulative_from_integrand(wavelength_nm, bkg_as2)                   # e-/arcsec^2

# ----------------------------
# Band SNR helpers
# ----------------------------
def band_edges_to_indices(bmin_nm: float, bmax_nm: float):
    i1 = int(np.searchsorted(wavelength_nm, bmin_nm, side="left"))
    i2 = int(np.searchsorted(wavelength_nm, bmax_nm, side="right") - 1)
    i1 = max(0, min(i1, len(wavelength_nm) - 1))
    i2 = max(0, min(i2, len(wavelength_nm) - 1))
    if i2 < i1:
        i1, i2 = i2, i1
    return i1, i2

def snr_for_band(bmin_nm: float, bmax_nm: float, instr, t_exp_s: float, n_pix_aperture: int) -> float:
    """
    S = ∫ signal dλ
    B_sky = (∫ bkg_as2 dλ) * Ω_pix(as^2) * n_pix_aperture
    σ = sqrt(S + B_sky + B_dark + n_pix * R^2)
    """
    i1, i2 = band_edges_to_indices(bmin_nm, bmax_nm)
    S_band = float(gCumSig[i2] - gCumSig[i1])                       # e-
    B_phot_per_as2 = float(gCumBkg_as2[i2] - gCumBkg_as2[i1])       # e- / arcsec^2
    B_sky = B_phot_per_as2 * OMEGA_PIX_AS2 * float(n_pix_aperture)  # e-
    B_dark = instr["dark_e_per_pix_s"] * float(n_pix_aperture) * float(t_exp_s)
    R = float(instr["read_noise_e"])
    sigma = math.sqrt(max(S_band + B_sky + B_dark + n_pix_aperture * (R**2), 0.0))
    return 0.0 if sigma <= 0 else S_band / sigma

# ----------------------------
# Swap in a specific SMASS reflectance
# ----------------------------
def set_reflectance_from_arrays(wl_src_nm, R_src):
    global f_ast
    wl_src_nm = np.asarray(wl_src_nm, float)
    R_src     = np.asarray(R_src, float)
    R_on_grid = np.interp(wavelength_nm, wl_src_nm, R_src)
    f_ast = lambda x_nm: np.interp(np.asarray(x_nm, float), wavelength_nm, R_on_grid)

def load_smass_spectrum_from_file(path_txt: str, id_str: str):
    raw = np.loadtxt(
        path_txt,
        dtype={"names": ("id", "w_um", "R"),
               "formats": ("U16", "f8", "f8")}
    )
    mask = (raw["id"] == id_str)
    wl_nm = raw["w_um"][mask] * 1_000.0  # μm → nm
    R     = raw["R"][mask]
    return wl_nm, R

# ============================
# Robust type-averaged spectra
# ============================
def get_ids_for_type(taxon: str):
    """Try SMASS helpers (if present); else empty list."""
    try:
        if hasattr(smass, "ids_for_type"):
            return list(smass.ids_for_type(taxon))
        if hasattr(smass, "get_ids_for_type"):
            return list(smass.get_ids_for_type(taxon))
    except Exception:
        pass
    return []

def _interp_and_norm_to_grid(wl_src_nm, R_src, grid_nm, norm_nm=550.0):
    wl_src_nm = np.asarray(wl_src_nm, float)
    R_src     = np.asarray(R_src, float)
    Rg = np.interp(grid_nm, wl_src_nm, R_src)
    if norm_nm is not None:
        norm = np.interp(norm_nm, grid_nm, Rg)
        if np.isfinite(norm) and norm > 0:
            Rg = Rg / norm
    return Rg

def robust_average_spectra(id_list, smass_path, grid_nm, norm_nm=550.0, sigma=3.0):
    curves = []
    kept_ids = []
    for aid in id_list:
        try:
            wl_nm, R = load_smass_spectrum_from_file(smass_path, aid)
            Rg = _interp_and_norm_to_grid(wl_nm, R, grid_nm, norm_nm=norm_nm)
            if np.all(~np.isfinite(Rg)):
                continue
            curves.append(Rg)
            kept_ids.append(aid)
        except Exception:
            continue

    if len(curves) == 0:
        raise RuntimeError("No usable spectra for averaging.")

    X = np.vstack(curves)  # (N_spectra, N_wave)
    median = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - median), axis=0)
    thresh = sigma * 1.4826 * mad
    keep = np.abs(X - median) <= thresh

    with np.errstate(invalid="ignore"):
        avg = np.nanmean(np.where(keep, X, np.nan), axis=0)
    avg = np.where(np.isfinite(avg), avg, median)
    return {"avg": avg, "median": median, "mad": mad, "mask": keep, "stack": X, "ids": kept_ids}

def make_type_average(taxon: str, smass_path, grid_nm=wavelength_nm, norm_nm=550.0, sigma=3.0,
                      explicit_ids=None, save_path=None):
    ids = list(explicit_ids) if explicit_ids is not None else get_ids_for_type(taxon)
    if not ids:
        raise RuntimeError(f"No IDs provided/found for type '{taxon}'. "
                           f"Pass explicit_ids=['000004', ...] if SMASS helper can't list them.")
    res = robust_average_spectra(ids, smass_path, grid_nm, norm_nm=norm_nm, sigma=sigma)
    if save_path:
        arr = np.column_stack([grid_nm, res["avg"]])
        np.savetxt(save_path, arr, delimiter=",",
                   header=f"wavelength_nm, reflectance_avg_{taxon} (normalized at {norm_nm} nm)",
                   comments="")
    return res

def set_reflectance_to_type_average(taxon: str, smass_path, norm_nm=550.0, sigma=3.0,
                                    explicit_ids=None, save_path=None, also_plot=False):
    res = make_type_average(taxon, smass_path, grid_nm=wavelength_nm,
                            norm_nm=norm_nm, sigma=sigma,
                            explicit_ids=explicit_ids, save_path=save_path)
    set_reflectance_from_arrays(wavelength_nm, res["avg"])
    if also_plot:
        plt.figure(figsize=(7, 4))
        for row in res["stack"]:
            plt.plot(wavelength_nm, row, alpha=0.15, lw=1)
        plt.plot(wavelength_nm, res["avg"], lw=2, label=f"{taxon}-avg")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance (norm @ 550 nm)")
        plt.title(f"{taxon}-type: per-object curves (faint) and robust average")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.show()
    return res

# ----------------------------
# Heatmaps
# ----------------------------
def plot_snr_heatmap_for_asteroid(id_str, smass_path, instr, t_exp_s, n_pix_aperture,
                                  min_nm=400, max_nm=900, step_nm=10):
    wl_nm, R = load_smass_spectrum_from_file(smass_path, id_str)
    set_reflectance_from_arrays(wl_nm, R)
    mV = ID_TO_Vmag.get(id_str, 10.0)
    K  = scale_factor_from_AB_mag(mV, lambda0_nm=550.0)
    rebuild_cums(K)
    L1_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L2_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L1, L2 = np.meshgrid(L1_vals, L2_vals)
    mask = L2 > L1
    SNR_vals = np.full_like(L1, np.nan, dtype=float)
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            if mask[i, j]:
                SNR_vals[i, j] = snr_for_band(float(L1[i, j]), float(L2[i, j]),
                                               instr, t_exp_s, n_pix_aperture)
    plt.figure(figsize=(7, 6))
    pc = plt.pcolormesh(L1, L2, SNR_vals, shading="auto")
    plt.colorbar(pc, label="SNR")
    plt.xlabel("Band min λ (nm)")
    plt.ylabel("Band max λ (nm)")
    plt.title(f"SNR heatmap for asteroid {id_str}")
    plt.show()

def plot_snr_heatmap_current(label, instr, t_exp_s, n_pix_aperture,
                             min_nm=400, max_nm=900, step_nm=10):
    """Plot SNR heatmap using the CURRENT reflectance (already installed in f_ast)."""
    L1_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L2_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L1, L2 = np.meshgrid(L1_vals, L2_vals)
    mask = L2 > L1
    SNR_vals = np.full_like(L1, np.nan, dtype=float)
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            if mask[i, j]:
                SNR_vals[i, j] = snr_for_band(float(L1[i, j]), float(L2[i, j]),
                                               instr, t_exp_s, n_pix_aperture)
    plt.figure(figsize=(7, 6))
    pc = plt.pcolormesh(L1, L2, SNR_vals, shading="auto")
    plt.colorbar(pc, label="SNR")
    plt.xlabel("Band min λ (nm)")
    plt.ylabel("Band max λ (nm)")
    plt.title(f"SNR heatmap for {label}")
    plt.show()

# ----------------------------
# Driver
# ----------------------------
tax_path = "/Users/karakanetis/Documents/GitHub/optimized-filters/main_code/taxonomy.pds.table.txt"
taxonomy_map = ld.load_taxonomy_table(tax_path)
print("taxonomy file:", tax_path, "| entries:", len(taxonomy_map))
from collections import Counter
print("type counts (first 10):", Counter(taxonomy_map.values()).most_common(10))

if __name__ == "__main__":
    smass_path = "/Users/karakanetis/Documents/GitHub/optimized-filters/main_code/smass2_all_spfit.txt"
    n_pix_aperture = 300  # adjust to your seeing/aperture

    # Build ID lists per taxon from the taxonomy table (include subclasses like 'S0', 'Sv', etc.)
    def ids_for_type(t):
        T = t.upper()
        return [aid for aid, cls in taxonomy_map.items() if str(cls).upper().startswith(T)]

    type_sets = {
        "C": ids_for_type("C"),
        "S": ids_for_type("S"),
        "V": ids_for_type("V"),
    }

    # --- Optionally precompute averages (no plotting here) ---
    mV_by_type = {"C": 10.0, "S": 8.5, "V": 6.5}
    for taxon, ids in type_sets.items():
        if not ids:
            print(f"[{taxon}] skipped: no IDs found in taxonomy file.")
            continue
        try:
            # Install the average reflectance for this taxon (no plot here)
            set_reflectance_to_type_average(
                taxon, smass_path,
                norm_nm=550.0, sigma=3.0,
                explicit_ids=ids, save_path=None, also_plot=False
            )
        except RuntimeError as e:
            print(f"[{taxon}] skipped: {e}")

    # --- One SNR 2-D plot for a C, S, and V *single object* ---
    print("\nNow plotting the three specific objects (C:000024, S:000006, V:000004)…")
    for aid, label in (("000024", "C-type example"),
                       ("000006", "S-type example"),
                       ("000004", "V-type example")):
        plot_snr_heatmap_for_asteroid(aid, smass_path, instr, t_exp_s, n_pix_aperture)

    # --- SNR 2-D plots for the *average* C, S, V spectra (once) ---
    print("\nSNR heatmaps for type-AVERAGE spectra (C, S, V)…")
    for taxon in ("C", "S", "V"):
        ids = type_sets.get(taxon, [])
        if not ids:
            print(f"[{taxon}] skipped: no IDs found in taxonomy table.")
            continue

        # Rebuild the average and install it (ensures we're using the right reflectance)
        set_reflectance_to_type_average(
            taxon, smass_path,
            norm_nm=550.0, sigma=3.0,
            explicit_ids=ids, save_path=None, also_plot=False
        )

        # Brightness scaling for a representative magnitude of the class
        K = scale_factor_from_AB_mag(mV_by_type.get(taxon, 10.0), lambda0_nm=550.0)
        rebuild_cums(K)

        # Plot once per average
        plot_snr_heatmap_current(
            label=f"{taxon}-avg",
            instr=instr,
            t_exp_s=t_exp_s,
            n_pix_aperture=n_pix_aperture,
            min_nm=400, max_nm=900, step_nm=10
        )
