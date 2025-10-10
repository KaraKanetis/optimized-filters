import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load data helpers
# -------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from data_files import load_data as ld

print("Using load_data from:", ld.__file__)

wavelength_nm, curves = ld.load_all_curves(base_dir="data_files")
interp = ld.make_curve_interpolators(wavelength_nm, curves)
f_sun    = interp["sun"]        # photons s^-1 m^-2 nm^-1
f_ast    = interp["asteroid"]   # unitless reflectance (will replace with 000004)
f_atm    = interp["atm"]        # unitless
f_qe     = interp["qe"]         # e- per photon
f_mirror = interp["mirror"]     # unitless

def make_instrument(d_m, s_pix_m, read_noise_e, dark_e_per_pix_s=0.0):
    return {
        "d_m": float(d_m),
        "s_pix_m": float(s_pix_m),
        "read_noise_e": float(read_noise_e),
        "dark_e_per_pix_s": float(dark_e_per_pix_s),
    }

instrument = make_instrument(
    d_m=1.0,           # telescope diameter [m]
    s_pix_m=5e-6,      # pixel pitch [m]
    read_noise_e=5.0,  # electrons RMS / pixel
    dark_e_per_pix_s=0.2
)

exposure_time_s = 600.0
plate_scale_arcsec_per_pix = 0.50
solid_angle_pix_arcsec2 = plate_scale_arcsec_per_pix ** 2

# ===== 3) Sky model (AB mag / arcsec^2) =====
SKY_AB_MAG_PER_ARCSEC2 = 21.7  # typical dark sky in V

# Physics constants
H_PLANCK = 6.62607015e-34  # J*s

def sky_surface_brightness_photons_per_nm(m_AB, lambda_nm):
    """AB mag/arcsec^2 -> photons s^-1 m^-2 nm^-1 arcsec^-2."""
    Fnu_AB0_W_m2_Hz = 3631.0 * 1e-26
    lam_m = np.asarray(lambda_nm, float) * 1e-9
    return (10.0 ** (-0.4 * m_AB)) * (Fnu_AB0_W_m2_Hz / (H_PLANCK * lam_m)) * 1e-9

def sky_photon_spectrum(lambda_nm):
    return sky_surface_brightness_photons_per_nm(SKY_AB_MAG_PER_ARCSEC2, lambda_nm)

# ===== 4) Install asteroid 000004 reflectance =====
def set_reflectance_from_arrays(wl_src_nm, R_src):
    """Replace f_ast with an interpolator defined on 'wavelength_nm' grid."""
    global f_ast
    wl_src_nm = np.asarray(wl_src_nm, float)
    R_src = np.asarray(R_src, float)
    R_on_grid = np.interp(wavelength_nm, wl_src_nm, R_src)

    def _interp(x_nm):
        return np.interp(np.asarray(x_nm, float), wavelength_nm, R_on_grid)

    f_ast = _interp

def load_smass_spectrum_from_file(path_txt, id_str):
    """
    Read one object from the SMASS table:
      columns: id, wavelength_um, reflectance
    Returns (wavelength_nm, reflectance).
    """
    raw = np.loadtxt(
        path_txt,
        dtype={"names": ("id", "w_um", "R"), "formats": ("U16", "f8", "f8")}
    )
    mask = (raw["id"] == id_str)
    wl_nm = raw["w_um"][mask] * 1_000.0  # um -> nm
    R = raw["R"][mask]
    return wl_nm, R

# ===== 5) AB magnitude normalization (sets overall brightness K) =====
def scale_factor_from_AB_mag(m_AB, lambda0_nm=550.0):
    """
    Choose K so that K * f_sun(λ0) * R(λ0) matches the AB photon flux at λ0.
    Only affects absolute counts; spectral shape stays the same.
    """
    Fnu_AB0_W_m2_Hz = 3631.0 * 1e-26
    lam_m = float(lambda0_nm) * 1e-9
    photons_per_nm_target = (10.0 ** (-0.4 * m_AB)) * (Fnu_AB0_W_m2_Hz / (H_PLANCK * lam_m)) * 1e-9
    base_photons_per_nm = float(f_sun(lambda0_nm) * f_ast(lambda0_nm))
    return photons_per_nm_target / max(base_photons_per_nm, 1e-30)

# ===== 6) Per-wavelength electrons integrands =====
def _collect_area_and_time(instr, t_exp_s):
    area_m2 = math.pi * (instr["d_m"] ** 2) / 4.0
    return area_m2, float(t_exp_s)

def asteroid_flux_raw(lambda_nm, K_scale=1.0):
    """photons s^-1 m^-2 nm^-1 BEFORE atm/optics/QE."""
    return K_scale * f_sun(lambda_nm) * f_ast(lambda_nm)

def signal_integrand_e_per_nm(lambda_nm, instr, t_exp_s, K_scale=1.0):
    S_ast_ph_nm = asteroid_flux_raw(lambda_nm, K_scale)
    T_atm = f_atm(lambda_nm)
    T_opt = f_mirror(lambda_nm)
    QE = f_qe(lambda_nm)
    area_m2, time_s = _collect_area_and_time(instr, t_exp_s)
    return time_s * area_m2 * (S_ast_ph_nm * T_atm * T_opt * QE)

def background_integrand_e_per_nm(lambda_nm, instr, t_exp_s):
    S_sky_ph_nm_as2 = sky_photon_spectrum(lambda_nm)  # per arcsec^2
    T_atm = f_atm(lambda_nm)
    T_opt = f_mirror(lambda_nm)
    QE = f_qe(lambda_nm)
    area_m2, time_s = _collect_area_and_time(instr, t_exp_s)
    return time_s * area_m2 * (S_sky_ph_nm_as2 * T_atm * T_opt * QE)  # per arcsec^2

# ===== 7) Cumulative electrons vs wavelength =====
gCumSig_e = None            # electrons (cumulative)
gCumBkg_e_per_as2 = None    # electrons per arcsec^2 (cumulative)

def cumulative_from_integrand(lambda_nm, y_e_per_nm):
    dlam = np.gradient(lambda_nm)
    return np.cumsum(y_e_per_nm * dlam)

def rebuild_cums(K_scale):
    global gCumSig_e, gCumBkg_e_per_as2
    sig_e_per_nm = signal_integrand_e_per_nm(wavelength_nm, instrument, exposure_time_s, K_scale)
    bkg_e_per_nm_as2 = background_integrand_e_per_nm(wavelength_nm, instrument, exposure_time_s)
    gCumSig_e = cumulative_from_integrand(wavelength_nm, sig_e_per_nm)
    gCumBkg_e_per_as2 = cumulative_from_integrand(wavelength_nm, bkg_e_per_nm_as2)

# ===== 8) Band helpers and SNR =====
def _band_edges_to_indices(bmin_nm, bmax_nm):
    i1 = int(np.searchsorted(wavelength_nm, bmin_nm, side="left"))
    i2 = int(np.searchsorted(wavelength_nm, bmax_nm, side="right") - 1)
    i1 = max(0, min(i1, len(wavelength_nm) - 1))
    i2 = max(0, min(i2, len(wavelength_nm) - 1))
    if i2 < i1:
        i1, i2 = i2, i1
    return i1, i2

def snr_for_band(bmin_nm, bmax_nm, instr, t_exp_s, n_pix_aperture):
    assert gCumSig_e is not None and gCumBkg_e_per_as2 is not None, "Call rebuild_cums(K_scale) first."
    i1, i2 = _band_edges_to_indices(bmin_nm, bmax_nm)

    # signal electrons in band
    S_band_e = float(gCumSig_e[i2] - gCumSig_e[i1])

    # background electrons in band (sky per arcsec^2 → per pixel → aperture)
    B_phot_e_per_as2 = float(gCumBkg_e_per_as2[i2] - gCumBkg_e_per_as2[i1])
    B_sky_e = B_phot_e_per_as2 * solid_angle_pix_arcsec2 * float(n_pix_aperture)

    # dark + read noise
    B_dark_e = instr["dark_e_per_pix_s"] * float(n_pix_aperture) * float(t_exp_s)
    read_rms_e = instr["read_noise_e"]
    noise_e = math.sqrt(max(S_band_e + B_sky_e + B_dark_e + n_pix_aperture * (read_rms_e ** 2), 0.0))

    return 0.0 if noise_e <= 0 else S_band_e / noise_e

# ===== 9) Run for asteroid 000004 and print SNRs =====
if __name__ == "__main__":
    # path to SMASS table
    smass_path = "/Users/karakanetis/Documents/GitHub/optimized-filters/main_code/smass2_all_spfit.txt"

    # load asteroid 000004 reflectance and install it
    asteroid_id = "000004"
    wl_src_nm, R_src = load_smass_spectrum_from_file(smass_path, asteroid_id)
    set_reflectance_from_arrays(wl_src_nm, R_src)

    # choose an apparent magnitude for absolute scaling (adjust if you have a better value)
    V_mag_assumed = 6.5
    K_scale = scale_factor_from_AB_mag(V_mag_assumed, lambda0_nm=550.0)

    # build cumulative signal/background
    rebuild_cums(K_scale)

    # example bands and aperture
    band1_edges_nm = (450.0, 650.0)
    band2_edges_nm = (730.0, 740.0)
    aperture_pixels = 300

    # compute SNR in each band and print
    snr_band1 = snr_for_band(band1_edges_nm[0], band1_edges_nm[1], instrument, exposure_time_s, aperture_pixels)
    snr_band2 = snr_for_band(band2_edges_nm[0], band2_edges_nm[1], instrument, exposure_time_s, aperture_pixels)

    print(f"Asteroid {asteroid_id}")
    print(f"Band 1 {band1_edges_nm[0]:.0f}-{band1_edges_nm[1]:.0f} nm  -> SNR = {snr_band1:.3f}")
    print(f"Band 2 {band2_edges_nm[0]:.0f}-{band2_edges_nm[1]:.0f} nm  -> SNR = {snr_band2:.3f}")

def plot_snr_map_for_current(label, instr, t_exp_s, n_pix_aperture,
                             min_nm=400, max_nm=900, step_nm=10):
    """Plot SNR for all top-hat bands (λ_max > λ_min) using the current reflectance f_ast."""
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
    plt.title(f"SNR for {label}")
    plt.tight_layout()
    plt.show(block=False); plt.pause(0.05)

def debug_band_terms(bmin_nm, bmax_nm, instr, t_exp_s, n_pix_aperture):
    assert gCumSig_e is not None and gCumBkg_e_per_as2 is not None
    i1 = int(np.searchsorted(wavelength_nm, bmin_nm, side="left"))
    i2 = int(np.searchsorted(wavelength_nm, bmax_nm, side="right") - 1)
    i1 = max(0, min(i1, len(wavelength_nm)-1))
    i2 = max(0, min(i2, len(wavelength_nm)-1))
    if i2 < i1: i1, i2 = i2, i1

    S_e = float(gCumSig_e[i2] - gCumSig_e[i1])
    B_ph_as2 = float(gCumBkg_e_per_as2[i2] - gCumBkg_e_per_as2[i1])
    B_sky_e = B_ph_as2 * (plate_scale_arcsec_per_pix**2) * float(n_pix_aperture)
    B_dark_e = instr["dark_e_per_pix_s"] * float(n_pix_aperture) * float(t_exp_s)
    RN_e2 = float(n_pix_aperture) * (instr["read_noise_e"]**2)
    sigma = math.sqrt(max(S_e + B_sky_e + B_dark_e + RN_e2, 0.0))
    snr = 0.0 if sigma <= 0 else S_e / sigma

    print(f"[{bmin_nm:.0f}-{bmax_nm:.0f} nm] S={S_e:.3e}  Bsky={B_sky_e:.3e}  Bdark={B_dark_e:.3e}  n*RN^2={RN_e2:.3e}  SNR={snr:.3f}")

# print
if __name__ == "__main__":
    smass_path = "/Users/karakanetis/Documents/GitHub/optimized-filters/main_code/smass2_all_spfit.txt"
    asteroid_id = "000004"
    wl_src_nm, R_src = load_smass_spectrum_from_file(smass_path, asteroid_id)
    set_reflectance_from_arrays(wl_src_nm, R_src)

# after set_reflectance_from_arrays(...)
V_mag_assumed = 6.5   # use a realistic value; AB ≈ V near 550 nm is fine for this purpose
K_scale = scale_factor_from_AB_mag(V_mag_assumed, lambda0_nm=550.0)
rebuild_cums(K_scale)

band1_edges_nm = (450.0, 650.0)
band2_edges_nm = (730.0, 740.0)
aperture_pixels = 300

print("Asteroid 000004 with K =", K_scale)
debug_band_terms(band1_edges_nm[0], band1_edges_nm[1], instrument, exposure_time_s, aperture_pixels)
debug_band_terms(band2_edges_nm[0], band2_edges_nm[1], instrument, exposure_time_s, aperture_pixels)


    # 2D SNR map for asteroid 000004 
plot_snr_map_for_current(
        label=f"asteroid {asteroid_id}",
        instr=instrument,
        t_exp_s=exposure_time_s,
        n_pix_aperture=aperture_pixels,
        min_nm=400,
        max_nm=900,
        step_nm=10)
plt.show()
