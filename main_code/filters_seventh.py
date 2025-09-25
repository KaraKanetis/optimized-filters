import numpy as np
import matplotlib.pyplot as plt
import data_files.load_data as ld
import math

# ----------------------------
# Load curves
# ----------------------------
wavelength_grid_nm, curves = ld.load_all_curves(base_dir="data_files")
interp = ld.make_curve_interpolators(wavelength_grid_nm, curves)
f_sun    = interp["sun"]
f_ast    = interp["asteroid"]
f_atm    = interp["atm"]
f_qe     = interp["qe"]
f_mirror = interp["mirror"]

COMMON_GRID_NM = np.arange(300.0, 1201.0, 1.0)  # nm

# ----------------------------
# Instrument & bands helpers
# ----------------------------
def make_instrument(d_m, gain_e_per_ph, s_pix_m, read_noise_e, dark_e_per_pix_s=0.0):
    """
    d_m:                 aperture diameter [m]
    gain_e_per_ph:       electrons per photon (ONLY use if QE doesn't already include it)
    s_pix_m:             pixel pitch [m]
    read_noise_e:        read noise per pixel [e- RMS]
    dark_e_per_pix_s:    dark current [e- / pixel / s]
    """
    return {
        "d_m": float(d_m),
        "gain_e_per_ph": float(gain_e_per_ph),
        "s_pix_m": float(s_pix_m),
        "read_noise_e": float(read_noise_e),
        "dark_e_per_pix_s": float(dark_e_per_pix_s),
    }

def make_bands(band_min_nm, band_max_nm, band_width_nm=None):
    return {
        "band_min_nm": float(band_min_nm),
        "band_max_nm": float(band_max_nm),
        "band_width_nm": None if band_width_nm is None else float(band_width_nm),
    }

# --------------------------------
# Physics functions (all λ in nm)
# --------------------------------
def spectrum_sun(wavelength_nm):
    """Solar photon spectrum vs wavelength_nm [photons / s / m² / nm]."""
    return f_sun(wavelength_nm)

def reflectance_asteroid(wavelength_nm):
    """Asteroid reflectance (unitless)."""
    return f_ast(wavelength_nm)

def transmission_atm(wavelength_nm):
    """Atmospheric transmission (0..1)."""
    return f_atm(wavelength_nm)

def transmission_optics(wavelength_nm):
    """Optics throughput (0..1), here: mirror reflectivity."""
    return f_mirror(wavelength_nm)

def ccd_qe(wavelength_nm):
    """CCD quantum efficiency [e- / photon] (0..1)."""
    return f_qe(wavelength_nm)

def transmission_filter_placeholder(wavelength_nm, band_min_nm, band_max_nm):
    """Top-hat filter transmission: 1 inside [band_min_nm, band_max_nm], else 0."""
    wavelength_nm = np.asarray(wavelength_nm, float)
    return ((wavelength_nm >= band_min_nm) & (wavelength_nm <= band_max_nm)).astype(float)

def sky_photon_spectrum(wavelength_nm):
    """Background photon spectrum [photons / s / m² / nm] (placeholder flat)."""
    return 1e2 * np.ones_like(wavelength_nm, dtype=float)

def asteroid_flux_raw(wavelength_nm):
    """Asteroid photon spectrum density ≈ Sun × Reflectance [photons / s / m² / nm]."""
    return spectrum_sun(wavelength_nm) * reflectance_asteroid(wavelength_nm)

# --------------------------------------------
# Per-wavelength integrands and band integrals
# --------------------------------------------
def _collecting_area_m2(instr):
    return math.pi * (instr["d_m"] ** 2) / 4.0

def signal_integrand(wavelength_nm, band_min_nm, band_max_nm, instr, t_exp_s):
    """
    Per-wavelength contribution to detected signal [e- per nm].
    """
    wavelength_nm = np.asarray(wavelength_nm, float)
    S_ast_ph_per_s_m2_nm = asteroid_flux_raw(wavelength_nm)      # photons / s / m^2 / nm
    T_atm   = transmission_atm(wavelength_nm)                    # unitless
    T_opt   = transmission_optics(wavelength_nm)                 # unitless
    T_filt  = transmission_filter_placeholder(wavelength_nm, band_min_nm, band_max_nm)  # unitless
    QE_eph  = ccd_qe(wavelength_nm)                              # e- / photon
    gain_e_per_ph = instr["gain_e_per_ph"]                       # note: usually 1.0 if QE already in e-/photon

    area_m2 = _collecting_area_m2(instr)
    # electrons per nm = photons/s/m^2/nm * s * m^2 * (unitless) * (e-/ph) * (optional extra gain_e_per_ph)
    return t_exp_s * area_m2 * (S_ast_ph_per_s_m2_nm * T_atm * T_opt * T_filt * QE_eph * gain_e_per_ph)

def detected_signal_band(wavelength_nm, band_min_nm, band_max_nm, instr, t_exp_s):
    """Total detected signal in [band_min_nm, band_max_nm] [electrons]."""
    integrand_e_per_nm = signal_integrand(wavelength_nm, band_min_nm, band_max_nm, instr, t_exp_s)
    return float(np.trapz(integrand_e_per_nm, wavelength_nm))

def background_integrand(wavelength_nm, band_min_nm, band_max_nm, instr, t_exp_s):
    """
    Per-wavelength sky background contribution [e- per nm per pixel] (placeholder).
    NOTE: Real sky should include pixel solid angle; this is a simplified placeholder.
    """
    wavelength_nm = np.asarray(wavelength_nm, float)
    S_sky_ph_per_s_m2_nm = sky_photon_spectrum(wavelength_nm)    # photons / s / m^2 / nm
    T_atm  = transmission_atm(wavelength_nm)
    T_opt  = transmission_optics(wavelength_nm)
    T_filt = transmission_filter_placeholder(wavelength_nm, band_min_nm, band_max_nm)
    QE_eph = ccd_qe(wavelength_nm)
    gain_e_per_ph = instr["gain_e_per_ph"]
    area_m2 = _collecting_area_m2(instr)

    # electrons per nm per pixel (pixel factor applied later when summing over aperture)
    return t_exp_s * area_m2 * (S_sky_ph_per_s_m2_nm * T_atm * T_opt * T_filt * QE_eph * gain_e_per_ph)

def detected_background_band(wavelength_nm, band_min_nm, band_max_nm, instr, t_exp_s, n_pix_aperture):
    """
    Total background [electrons] in band: (sky per pixel + dark per pixel) summed over aperture pixels.
    """
    b_int_e_per_nm_perpix = background_integrand(wavelength_nm, band_min_nm, band_max_nm, instr, t_exp_s)
    B_sky_perpix_e = float(np.trapz(b_int_e_per_nm_perpix, wavelength_nm))           # e- per pixel
    B_sky_total_e  = B_sky_perpix_e * float(n_pix_aperture)                          # sum over pixels
    B_dark_total_e = instr["dark_e_per_pix_s"] * float(n_pix_aperture) * float(t_exp_s)
    return B_sky_total_e + B_dark_total_e

# --------------------------------------------
# Uncertainty and color helpers
# --------------------------------------------
def delta_S(S_e, B_e, instr, n_pix_aperture):
    """
    Total noise [e- RMS]: sqrt( S + B + N_pix * R^2 ).
    """
    R_e = float(instr["read_noise_e"])
    var_e2 = float(S_e) + float(B_e) + float(n_pix_aperture) * (R_e ** 2)
    return float(np.sqrt(max(var_e2, 0.0)))

def color_mag(S1_e, dS1_e, S2_e, dS2_e):
    """Color (m2 - m1) [mag] and its uncertainty [mag]."""
    eps = 1e-30
    c_mag = 2.5 * np.log10(max(S1_e, eps) / max(S2_e, eps))
    var_rel = 0.0
    if S1_e > 0:
        var_rel += (dS1_e / S1_e) ** 2
    if S2_e > 0:
        var_rel += (dS2_e / S2_e) ** 2
    dc_mag = (2.5 / np.log(10.0)) * np.sqrt(var_rel)
    return c_mag, dc_mag

# --------------------------------------------
# Proper cumulative integral from integrand
# --------------------------------------------
def cumulative_signal_from_integrand(wavelength_nm, integrand_e_per_nm):
    """Cumulative electrons integrated from min(wavelength_nm) up to each point."""
    wavelength_nm = np.asarray(wavelength_nm, float)
    dlam_nm = np.gradient(wavelength_nm)  # nm
    return np.cumsum(integrand_e_per_nm * dlam_nm)

# ===========================
# TEST / PLOTS
# ===========================
# wavelength grid for everything (match the common grid from loaders)
wavelength_nm = wavelength_grid_nm  # e.g., 1-nm grid from 300–1200 nm
# Or subset:
# wavelength_nm = np.arange(400.0, 901.0, 1.0)

# instrument & exposure
instr = make_instrument(
    d_m=1.0,
    gain_e_per_ph=1.0,   # keep at 1.0 if QE is already e-/photon
    s_pix_m=5e-6,
    read_noise_e=5.0,
    dark_e_per_pix_s=0.2,
)
t_exp_s = 60.0  # seconds

# two example bands [nm]
band1_nm = (450.0, 650.0)
band2_nm = (730.0, 740.0)

# aperture size [pixels]
n_pix_aperture = 50

# ---- Plot transmissions and QE ----
Tatm = transmission_atm(wavelength_nm)
Topt = transmission_optics(wavelength_nm)
Tf   = transmission_filter_placeholder(wavelength_nm, 460.0, 470.0)  # simple demo filter
QE   = ccd_qe(wavelength_nm)

plt.figure(figsize=(8,5))
plt.plot(wavelength_nm, Tatm, label="Atmosphere T(λ)")
plt.plot(wavelength_nm, Topt, label="Optics throughput")
plt.plot(wavelength_nm, Tf,   label="Filter 460–470 nm")
plt.plot(wavelength_nm, QE,   label="CCD QE (e⁻/photon)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission / QE (unitless)")
plt.title("Measured transmissions & CCD QE")
plt.legend(); plt.grid(True); plt.show()

# ---- Per-λ detected signal (integrand) for band1 ----
signal_pw_e_per_nm = signal_integrand(wavelength_nm, band1_nm[0], band1_nm[1], instr, t_exp_s)

plt.figure(figsize=(8,5))
plt.plot(wavelength_nm, signal_pw_e_per_nm, label=f"per-λ signal in [{band1_nm[0]:.0f}, {band1_nm[1]:.0f}] nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Electrons per nm")
plt.title("Per-λ detected signal (integrand)")
plt.legend(); plt.grid(True); plt.show()

# ---- Total band signals (scalars) and color demo ----
S1_e = detected_signal_band(wavelength_nm, band1_nm[0], band1_nm[1], instr, t_exp_s)
S2_e = detected_signal_band(wavelength_nm, band2_nm[0], band2_nm[1], instr, t_exp_s)
B1_e = detected_background_band(wavelength_nm, band1_nm[0], band1_nm[1], instr, t_exp_s, n_pix_aperture)
B2_e = detected_background_band(wavelength_nm, band2_nm[0], band2_nm[1], instr, t_exp_s, n_pix_aperture)

def snr_from_signal_background(S_e, B_e, instr, n_pix):
    sigma_e = delta_S(S_e, B_e, instr, n_pix)
    return float(S_e / sigma_e) if sigma_e > 0 else np.nan

SNR1 = snr_from_signal_background(S1_e, B1_e, instr, n_pix_aperture)

dS1_e = delta_S(S1_e, B1_e, instr, n_pix_aperture)
dS2_e = delta_S(S2_e, B2_e, instr, n_pix_aperture)
c_mag, dc_mag = color_mag(S1_e, dS1_e, S2_e, dS2_e)

print("Band 1:", band1_nm, f"  S1 = {S1_e:.3e} e-   B1 = {B1_e:.3e} e-   ΔS1 = {dS1_e:.3e} e-")
print("Band 2:", band2_nm, f"  S2 = {S2_e:.3e} e-   B2 = {B2_e:.3e} e-   ΔS2 = {dS2_e:.3e} e-")
print(f"Color (m2 - m1) [mag]: {c_mag:.4f}   Uncertainty [mag]: {dc_mag:.4f}")
print(f"SNR (band1): {SNR1:.2f}")

# Dark current contribution
Bdark_e = instr["dark_e_per_pix_s"] * n_pix_aperture * t_exp_s
print("Dark current contribution (per band):", f"{Bdark_e:.3e}", "electrons")

# ---- Sun, Reflectance, Asteroid flux ----
Ssun_ph = spectrum_sun(wavelength_nm)
plt.figure(figsize=(8,5))
plt.plot(wavelength_nm, Ssun_ph)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Solar photons / s / m² / nm")
plt.title("Solar spectrum S_sun(λ)")
plt.grid(True); plt.show()

Rast = reflectance_asteroid(wavelength_nm)
plt.figure(figsize=(8,5))
plt.plot(wavelength_nm, Rast)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (unitless)")
plt.title("Asteroid reflectance R_ast(λ)")
plt.grid(True); plt.show()

S_ast_ph = asteroid_flux_raw(wavelength_nm)
plt.figure(figsize=(8,5))
plt.plot(wavelength_nm, S_ast_ph)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Photons / s / m² / nm")
plt.title("Asteroid photon spectrum Ṡ_ast(λ) = S_sun × R_ast")
plt.grid(True); plt.show()

# -- background per wavelength (using placeholder sky)
b_pw_e_per_nm_perpix = background_integrand(wavelength_nm, band1_nm[0], band1_nm[1], instr, t_exp_s)
plt.figure(figsize=(8,5))
plt.plot(wavelength_nm, b_pw_e_per_nm_perpix, label="per-λ background (sky, per pixel)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Electrons per nm per pixel")
plt.title(f"Per-λ background inside [{band1_nm[0]:.0f}, {band1_nm[1]:.0f}] nm")
plt.legend(); plt.grid(True); plt.show()

# ----------------------------
# SNR map vs band edges (λ2 > λ1)
# ----------------------------
def snr_for_band_edges(bmin_nm, bmax_nm):
    S_e = detected_signal_band(wavelength_nm, bmin_nm, bmax_nm, instr, t_exp_s)
    B_e = detected_background_band(wavelength_nm, bmin_nm, bmax_nm, instr, t_exp_s, n_pix_aperture)
    return snr_from_signal_background(S_e, B_e, instr, n_pix_aperture)

# choose a valid subrange inside your grid (e.g., 400–900 nm)
L1_vals_nm = np.linspace(400.0, 880.0, 121)
L2_vals_nm = np.linspace(410.0, 900.0, 121)
L1_mesh_nm, L2_mesh_nm = np.meshgrid(L1_vals_nm, L2_vals_nm)

SNR_vals = np.full_like(L1_mesh_nm, np.nan, dtype=float)
valid = L2_mesh_nm > L1_mesh_nm
# vectorized-ish fill
it = np.nditer(valid, flags=['multi_index'])
while not it.finished:
    i, j = it.multi_index
    if valid[i, j]:
        SNR_vals[i, j] = snr_for_band_edges(L1_mesh_nm[i, j], L2_mesh_nm[i, j])
    it.iternext()

plt.figure(figsize=(7,6))
pc = plt.pcolormesh(L1_mesh_nm, L2_mesh_nm, SNR_vals, shading="auto", cmap="viridis")
plt.colorbar(pc, label="SNR (electrons-based)")
plt.xlabel("Band min (nm)")
plt.ylabel("Band max (nm)")
plt.title("SNR vs band edges (band_max_nm > band_min_nm)")
plt.show()
