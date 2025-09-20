# reading files, no more dummy numbers
# removing the unit conversion from here and putting it into the file reader

import numpy as np
import matplotlib.pyplot as plt
import data_files.load_data as ld
import math

# read curves once
lam, curves = ld.load_all_curves(base_dir="data_files")
interp = ld.make_curve_interpolators(lam, curves)
f_sun    = interp["sun"]
f_ast    = interp["asteroid"]
f_atm    = interp["atm"]
f_qe     = interp["qe"]
f_mirror = interp["mirror"]

COMMON_GRID_NM = np.arange(300.0, 1201.0, 1.0)

# ----------------------------
# Instrument & bands helpers
# ----------------------------
def make_instrument(d_m, gain_e_per_ph, s_pix_m, read_noise_e, dark_e_per_pix_s=0.0):
    return {
        "d": d_m,
        "gain": gain_e_per_ph,
        "s_pix": s_pix_m,
        "read_noise": read_noise_e,  
        "dark_e_per_pix_s": dark_e_per_pix_s,
    }

def make_bands(V_min_nm, V_max_nm, V_width_nm):
    return {
        "V_min": V_min_nm,
        "V_max": V_max_nm,
        "V_width": V_width_nm,
    }

# --------------------------------
# Physics functions
# --------------------------------
def spectrum_sun(wavelength_nm):
    """Solar photon spectrum on nm grid (from file)."""
    return f_sun(wavelength_nm)

def reflectance_asteroid(wavelength_nm):
    """Asteroid reflectance (from file)."""
    return f_ast(wavelength_nm)

def transmission_atm(wavelength_nm):
    """Atmospheric transmission (from file)."""
    return f_atm(wavelength_nm)

def transmission_optics(wavelength_nm):
    """Optics throughput (here: mirror reflectivity from file)."""
    return f_mirror(wavelength_nm)

def ccd_qe(wavelength_nm):
    """CCD QE (from file)."""
    return f_qe(wavelength_nm)

def transmission_filter_placeholder(wavelength_nm, wavelength1_nm, wavelength2_nm):
    """Tophat filter: 1 inside [λ1, λ2], else 0."""
    wavelength = np.asarray(wavelength_nm, float)
    return ((wavelength >= wavelength1_nm) & (wavelength <= wavelength2_nm)).astype(float)

def sky_photon_spectrum(wavelength_nm):
    """Background photon spectrum (placeholder; keep simple for now)."""
    # background placeholder, need spectrum?
    return 1e2 * np.ones_like(wavelength_nm, dtype=float)

def asteroid_flux_raw(wavelength_nm):
    """Asteroid photon spectrum ≈ Sun × Reflectance (now from real curves)."""
    return spectrum_sun(wavelength_nm) * reflectance_asteroid(wavelength_nm)

# --------------------------------------------
# Per-wavelength integrands and band integrals
# --------------------------------------------
def signal_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s):
    """
    Per-wavelength contribution to detected signal (Eq. 4 BEFORE integrating).
    Returns array same shape as wavelength_nm.
    """
    wavelength = np.asarray(wavelength_nm, float)
    S_ast = asteroid_flux_raw(wavelength)
    Tatm  = transmission_atm(wavelength)
    Topt  = transmission_optics(wavelength)
    Tf    = transmission_filter_placeholder(wavelength, wavelength1_nm, wavelength2_nm)
    QE    = ccd_qe(wavelength)

    area_m2 = np.pi * (instr["d"] ** 2) / 4.0
    return t_exp_s * instr["gain"] * area_m2 * (S_ast * Tatm * Topt * Tf * QE)

def detected_signal_band(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s):
    """Total detected signal (electrons) in band [λ1, λ2] via trapezoidal integral over λ."""
    integrand = signal_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s)
    return float(np.trapz(integrand, wavelength_nm))

def background_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s):
    """Per-wavelength background contribution (placeholder sky × optics × QE × filter)."""
    wavelength = np.asarray(wavelength_nm, float)
    Ssky = sky_photon_spectrum(wavelength)
    Tatm = transmission_atm(wavelength)
    Topt = transmission_optics(wavelength)
    Tf   = transmission_filter_placeholder(wavelength, wavelength1_nm, wavelength2_nm)
    QE   = ccd_qe(wavelength)
    area_m2 = np.pi * (instr["d"] ** 2) / 4.0
    return t_exp_s * instr["gain"] * area_m2 * (Ssky * Tatm * Topt * Tf * QE)

def detected_background_band(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s, n_pix_aperture):
    """Total background (electrons) in band [λ1, λ2] (sky + dark current)."""
    b_int = background_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s)
    B_photon = float(np.trapz(b_int, wavelength_nm))
    B_dark   = instr["dark_e_per_pix_s"] * float(n_pix_aperture) * float(t_exp_s)
    return B_photon + B_dark

# --------------------------------------------
# Uncertainty and color helpers
# --------------------------------------------
def delta_S(S, B, instr, n_pix_aperture):
    """
    Shot noise (S + B) plus read noise per pixel × number of pixels (quadrature).
    Eq. (7)
    """
    R = float(instr["read_noise"])   # <-- FIX: correct key
    var = float(S) + float(B) + float(n_pix_aperture) * (R**2)
    return float(np.sqrt(max(var, 0.0)))

def color_mag(S1, dS1, S2, dS2):
    """Color (m2 - m1) and its uncertainty (Eq. 8)."""
    eps = 1e-30
    c = 2.5 * np.log10(max(S1, eps) / max(S2, eps))
    var = 0.0
    if S1 > 0:
        var += (dS1 / S1) ** 2
    if S2 > 0:
        var += (dS2 / S2) ** 2
    dc = (2.5 / np.log(10.0)) * np.sqrt(var)
    return c, dc

# --------------------------------------------
# Proper cumulative integral from integrand
# --------------------------------------------
def cumulative_signal_from_integrand(wavelength_nm, integrand):
    """Cumulative integral from min(λ) up to each λ_i."""
    wavelength = np.asarray(wavelength_nm, float)
    dlam = np.gradient(wavelength)            # Δλ at each point
    return np.cumsum(integrand * dlam)

# ===========================
# TEST / PLOTS
# ===========================
# wavelength grid for everything (match the common grid)
wavelength = lam  # use the same grid the loaders produced (1-nm from 300–1200)
# To use 400–900 only:
# wavelength = np.arange(400.0, 901.0, 1.0)

# instrument & exposure
instr = make_instrument(
    d_m=1.0,
    gain_e_per_ph=1.0,
    s_pix_m=5e-6,
    read_noise_e=5.0,
    dark_e_per_pix_s=0.2,
)
t_exp = 60.0  # seconds

# two example bands
band1 = (450, 650)
band2 = (730, 740)

# aperture size
n_pix_aperture = 50

# ---- Plot transmissions and QE ----
Tatm = transmission_atm(wavelength)
Topt = transmission_optics(wavelength)
Tf   = transmission_filter_placeholder(wavelength, 460, 470)  # simple demo filter
QE   = ccd_qe(wavelength)

plt.figure(figsize=(8,5))
plt.plot(wavelength, Tatm, label="Atmosphere T(λ)")
plt.plot(wavelength, Topt, label="Optics throughput")
plt.plot(wavelength, Tf,   label="Filter 460–470 nm")
plt.plot(wavelength, QE,   label="CCD QE")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission / QE")
plt.title("Measured transmissions & CCD QE")
plt.legend(); plt.grid(True); plt.show()

# ---- Per-λ detected signal (integrand) for band1 ----
signal_pw = signal_integrand(wavelength, band1[0], band1[1], instr, t_exp)

plt.figure(figsize=(8,5))
plt.plot(wavelength, signal_pw, label=f"per-λ signal in [{band1[0]}, {band1[1]}] nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Electrons per nm (scaled)")
plt.title("Per-λ detected signal (integrand of Eq. 4)")
plt.legend(); plt.grid(True); plt.show()

# ---- Cumulative signal curve from the per-λ integrand ----
# cum_sig = cumulative_signal_from_integrand(wavelength, signal_pw)

# plt.figure(figsize=(8,5))
# plt.plot(wavelength, cum_sig, label="cumulative signal up to λ")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Integrated electrons up to λ")
# plt.title("Cumulative detected signal vs wavelength")
# plt.legend(); plt.grid(True); plt.show()


# #signal to noise ratio per wavelenght using bands
# signal_pw = signal_integrand(wavelength, band1[0], band1[1], instr, t_exp)
# background_pw = background_integrand(wavelength, band1[0], band1[1], instr, t_exp)

# # EQN 7: sqrt (B+S+ (expanded version) + aperature*R^2)
# R = instr["read_noise"]
# noise_pw = np.sqrt(signal_pw + background_pw + n_pix_aperture * (R**2))

# # eqn SNR = signal/noise dividing
# SNR_pw = signal_pw / noise_pw

# plt.figure(figsize=(8,5))
# plt.plot(wavelength, SNR_pw, label=f"SNR per wavelength in [{band1[0]}, {band1[1]}] nm")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("SNR per nm")
# plt.title("Signal to Noise Ratio vs Wavelength")
# plt.legend(); plt.grid(True); plt.show()

# ---- Total band signals (scalars) and color demo ----
S1 = detected_signal_band(wavelength, band1[0], band1[1], instr, t_exp)
S2 = detected_signal_band(wavelength, band2[0], band2[1], instr, t_exp)
B1 = detected_background_band(wavelength, band1[0], band1[1], instr, t_exp, n_pix_aperture)
B2 = detected_background_band(wavelength, band2[0], band2[1], instr, t_exp, n_pix_aperture)
SNR = S1 / math.sqrt(S1+B1)


dS1 = delta_S(S1, B1, instr, n_pix_aperture)
dS2 = delta_S(S2, B2, instr, n_pix_aperture)
c, dc = color_mag(S1, dS1, S2, dS2)

print("Band 1:", band1, "  S1 =", S1, "  B1 =", B1, "  ΔS1 =", dS1)
print("Band 2:", band2, "  S2 =", S2, "  B2 =", B2, "  ΔS2 =", dS2)
print("Color (m2 - m1) [mag]:", c, "  Uncertainty [mag]:", dc)
print("SNR", SNR)

# Dark current contribution
Bdark = instr["dark_e_per_pix_s"] * n_pix_aperture * t_exp
print("Dark current contribution (per band):", Bdark, "electrons")

# ---- Sun, Reflectance, Asteroid flux ----
Ssun = spectrum_sun(wavelength)
plt.figure(figsize=(8,5))
plt.plot(wavelength, Ssun)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Solar photons / s / m² / nm (scaled)")
plt.title("Solar spectrum S_sun(λ) (from file)")
plt.grid(True); plt.show()

Rast = reflectance_asteroid(wavelength)
plt.figure(figsize=(8,5))
plt.plot(wavelength, Rast)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (unitless)")
plt.title("Asteroid reflectance R_ast(λ) (from file)")
plt.grid(True); plt.show()

S_ast = asteroid_flux_raw(wavelength)
plt.figure(figsize=(8,5))
plt.plot(wavelength, S_ast)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Asteroid photons (scaled)")
plt.title("Asteroid photon spectrum Ṡ_ast(λ) = S_sun × R_ast")
plt.grid(True); plt.show()

# -- background per wavelength (using placeholder sky)
b_pw = background_integrand(wavelength, band1[0], band1[1], instr, t_exp)
plt.figure(figsize=(8,5))
plt.plot(wavelength, b_pw, label="per-λ background (sky)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Electrons per nm (scaled)")
plt.title(f"Per-λ background inside [{band1[0]}, {band1[1]}] nm")
plt.legend(); plt.grid(True); plt.show()




# 
def SNR(lam1, lam2):
    S1 = detected_signal_band(wavelength, lam1, lam2, instr, t_exp)
    B1 = detected_background_band(wavelength, lam1, lam2, instr, t_exp, n_pix_aperture)
    SNR = S1 / math.sqrt(S1+B1)
    return SNR

# Define ranges
L1_vals = np.linspace(0, 100, 200)
L2_vals = np.linspace(0, 100, 200)

# Create grid
L1, L2 = np.meshgrid(L1_vals, L2_vals)

# Mask out invalid region (L2 <= L1)
mask = L2 > L1

# Compute SNR only where valid
SNR_vals = np.full_like(L1, np.nan, dtype=float)  # fill with NaN
SNR_vals[mask] = SNR(L1[mask], L2[mask])

# Plot
plt.figure(figsize=(7,6))
c = plt.pcolormesh(L1, L2, SNR_vals, shading="auto", cmap="viridis")
plt.colorbar(c, label="SNR")
plt.xlabel("L1")
plt.ylabel("L2")
plt.title("SNR vs L1 and L2 (L2 > L1)")
plt.show()

# random to show commit

