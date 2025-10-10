# filters_clean.py
# Purpose: Compute band SNR heatmaps for asteroid spectra with consistent units and a clean pipeline.

import math
import numpy as np
import matplotlib.pyplot as plt

import load_data as ld
import SMASS as smass

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
    """
    Convert AB mag/arcsec^2 to photons s^-1 m^-2 nm^-1 arcsec^-2, as a weak 1/λ spectrum.
    """
    Fnu_AB0 = 3631.0 * 1e-26  # W m^-2 Hz^-1
    lam_m = np.asarray(lambda_nm, float) * 1e-9
    # photons per s per m^2 per nm per arcsec^2
    return (10.0 ** (-0.4 * m_AB)) * (Fnu_AB0 / (_H * lam_m)) * 1e-9

def sky_photon_spectrum(wavelength_nm: np.ndarray) -> np.ndarray:
    """
    Returns Ṡ_sky(λ) in photons s^-1 m^-2 nm^-1 arcsec^-2 (surface brightness).
    We multiply by pixel solid angle and number of pixels at the BAND level.
    """
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

# ----------------------------
# Heatmap plot for a given asteroid ID
# ----------------------------
def plot_snr_heatmap_for_asteroid(id_str, smass_path, instr, t_exp_s, n_pix_aperture,
                                  min_nm=400, max_nm=900, step_nm=10):
    wl_nm, R = load_smass_spectrum_from_file(smass_path, id_str)
    set_reflectance_from_arrays(wl_nm, R)

    # magnitude → scale
    mV = ID_TO_Vmag.get(id_str, 10.0)
    K  = scale_factor_from_AB_mag(mV, lambda0_nm=550.0)
    rebuild_cums(K)   # builds gCumSig (e-) and gCumBkg_as2 (e-/arcsec^2)

    # grid
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

# ----------------------------
# Driver
# ----------------------------
if __name__ == "__main__":
    smass_path = "/Users/karakanetis/Documents/GitHub/optimized-filters/main_code/smass2_all_spfit.txt"
    n_pix_aperture = 300  # try something realistic; tweak with your seeing & aperture radius

    for aid in ("000024", "000006", "000004"):  # C, S, V
        plot_snr_heatmap_for_asteroid(aid, smass_path, instr, t_exp_s, n_pix_aperture)
