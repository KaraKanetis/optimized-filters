#so, since I met with Rob, I forgot how messy this code was and completely forgot 
#how/what I was going to fix... asked AI to look at my code and clean it up without
#changing my base structure/code, also to comment where/when it made changes
#also asked it to stylize my code so its easier to navigate/read sections

#Whats changed in this version
# - Added Dark Current
# edited Delta s to include n_pix in the uncertainty (since wed added dark current)

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Instrument & bands helpers
# ----------------------------
def make_instrument(d_m, gain_e_per_ph, s_pix_m, read_noise_e):
    return {
        "d": d_m,                # telescope diameter [m]
        "gain": gain_e_per_ph,   # electrons per photon
        "s_pix": s_pix_m,        # pixel size [m]  (unused for now)
        "read_noise": read_noise_e,  # electrons RMS (unused for now)
        "dark_e_per_pix_s": dark_e_per_pix_s,
    }

def make_bands(V_min_nm, V_max_nm, V_width_nm):
    return {
        "V_min": V_min_nm,
        "V_max": V_max_nm,
        "V_width": V_width_nm,
    }


# ----------------------------
# Toy physics functions
# ----------------------------
def spectrum_sun(wavelength_nm):
    """Toy solar spectrum ~ 1/λ (arbitrary scale)."""
    wavelength = np.asarray(wavelength_nm, float)
    return 1e6 / np.clip(wavelength, 1.0, None)

def reflectance_asteroid(wavelength_nm):
    """Toy asteroid reflectance with slight red slope (unitless)."""
    wavelength = np.asarray(wavelength_nm, float)
    return 0.95 + 0.0002 * (wavelength - 550.0)

def transmission_atm(wavelength_nm):
    """Toy atmosphere: perfectly transparent (1.0)."""
    return np.ones_like(wavelength_nm, dtype=float)

def transmission_optics(wavelength_nm):
    """Toy optics: flat 90% throughput."""
    return 0.9 * np.ones_like(wavelength_nm, dtype=float)

def ccd_qe(wavelength_nm):
    """Toy CCD QE: Gaussian-shaped sensitivity centered near 650 nm."""
    wavelength = np.asarray(wavelength_nm, float)
    return np.exp(-((wavelength - 650.0) / 200.0) ** 2)

def transmission_filter_placeholder(wavelength_nm, wavelength1_nm, wavelength2_nm):
    """Tophat filter: 1 inside [λ1, λ2], else 0."""
    wavelength = np.asarray(wavelength_nm, float)
    return ((wavelength >= wavelength1_nm) & (wavelength <= wavelength2_nm)).astype(float)

def sky_photon_spectrum(wavelength_nm):
    """Toy sky background: constant level."""
    return 1e2 * np.ones_like(wavelength_nm, dtype=float)

def asteroid_flux_raw(wavelength_nm):
    """Unscaled asteroid photon spectrum ≈ Sun × Reflectance (toy)."""
    return spectrum_sun(wavelength_nm) * reflectance_asteroid(wavelength_nm)


# --------------------------------------------
# Per-wavelength integrands and band integrals
# --------------------------------------------
def signal_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s):
    """
    Per-wavelength contribution to detected signal (eq. 4 BEFORE integrating).
    Returns array same shape as wavelength_nm.
    """
    wavelength = np.asarray(wavelength_nm, float)
    S_ast = asteroid_flux_raw(wavelength)     # Sun × Reflectance (toy)
    Tatm  = transmission_atm(wavelength)
    Topt  = transmission_optics(wavelength)
    Tf    = transmission_filter_placeholder(wavelength, wavelength1_nm, wavelength2_nm)
    QE    = ccd_qe(wavelength)

    area_m2 = np.pi * (instr["d"] ** 2) / 4.0
    return t_exp_s * instr["gain"] * area_m2 * (S_ast * Tatm * Topt * Tf * QE)

def detected_signal_band(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s):
    """
    Total detected signal (electrons) in band [λ1, λ2] via trapezoidal integral over λ.
    """
    integrand = signal_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s)
    return float(np.trapz(integrand, wavelength_nm))

def background_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s):
    """Per-wavelength background contribution (toy)."""
    wavelength = np.asarray(wavelength_nm, float)
    Ssky = sky_photon_spectrum(wavelength)
    Tatm = transmission_atm(wavelength)
    Topt = transmission_optics(wavelength)
    Tf   = transmission_filter_placeholder(wavelength, wavelength1_nm, wavelength2_nm)
    QE   = ccd_qe(wavelength)
    area_m2 = np.pi * (instr["d"] ** 2) / 4.0
    return t_exp_s * instr["gain"] * area_m2 * (Ssky * Tatm * Topt * Tf * QE)

def detected_background_band(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s):
    """Total background (electrons) in band [λ1, λ2]."""
    integrand = background_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s)
    B_photon = float(np.trapz(integrand, wavelength_nm))

    B_dark= instr["dark_e_per_pix_s"] * float(n_pix_aperature) * float(t_exp_s)
    return B_photon + B_dark


# --------------------------------------------
# Uncertainty and color helpers
# --------------------------------------------
def delta_S(S, B):
    """ΔS = sqrt(S + B)  (paper eq. 7)"""
    return float(np.sqrt(max(S + B, 0.0)))

def color_mag(S1, dS1, S2, dS2):
    """
    Color (m2 - m1) = 2.5 log10(S1/S2) and its uncertainty (paper eq. 8).
    """
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
    """
    Cumulative integral from min(λ) up to each λ_i:
    cum[i] = sum_{j<=i} integrand[j] * Δλ_j
    """
    wavelength = np.asarray(wavelength_nm, float)
    dlam = np.gradient(wavelength)            # Δλ at each point
    return np.cumsum(integrand * dlam)        # elementwise multiply then cumsum


# ===========================
# TEST / PLOTS
# ===========================

# wavelength grid for everything
wavelength = np.arange(400.0, 901.0, 1.0)   # 400–900 nm (1 nm step)

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
band1 = (450, 650)   # wide band just for visualization
band2 = (730, 740)   # narrow red band

# aperature size
n_pix_aperature = 50

# ---- Plot transmissions and QE (as you had) ----
Tatm = transmission_atm(wavelength)
Topt = transmission_optics(wavelength)
Tf   = transmission_filter_placeholder(wavelength, 460, 470)  # example filter
QE   = ccd_qe(wavelength)

plt.figure(figsize=(8,5))
plt.plot(wavelength, Tatm, label="atm transmission")
plt.plot(wavelength, Topt, label="optics transmission")
plt.plot(wavelength, Tf,   label="Filter 460–470 nm")
plt.plot(wavelength, QE,   label="CCD QE")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission / QE")
plt.title("Toy transmissions & CCD QE")
plt.legend(); plt.grid(True); plt.show()

# ---- Per-λ detected signal (integrand) for band1 ----
signal_pw = signal_integrand(wavelength, band1[0], band1[1], instr, t_exp)

plt.figure(figsize=(8,5))
plt.plot(wavelength, signal_pw, label=f"per-λ signal in [{band1[0]}, {band1[1]}] nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Electrons per nm (scaled)")
plt.title("Per-λ detected signal (integrand of eq. 4)")
plt.legend(); plt.grid(True); plt.show()

# ---- Cumulative signal curve from the per-λ integrand ----
cum_sig = cumulative_signal_from_integrand(wavelength, signal_pw)

plt.figure(figsize=(8,5))
plt.plot(wavelength, cum_sig, label="cumulative signal up to λ")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Integrated electrons up to λ")
plt.title("Cumulative detected signal vs wavelength")
plt.legend(); plt.grid(True); plt.show()

# ---- Total band signals (scalars) and color demo ----
S1 = detected_signal_band(wavelength, band1[0], band1[1], instr, t_exp)
S2 = detected_signal_band(wavelength, band2[0], band2[1], instr, t_exp)
B1 = detected_background_band(wavelength, band1[0], band1[1], instr, t_exp, n_pix_aperature)
B2 = detected_background_band(wavelength, band2[0], band2[1], instr, t_exp, n_pix_aperature)

dS1 = delta_S(S1, B1)
dS2 = delta_S(S2, B2)
c, dc = color_mag(S1, dS1, S2, dS2)

print("Band 1:", band1, "  S1 =", S1, "  B1 =", B1, "  ΔS1 =", dS1)
print("Band 2:", band2, "  S2 =", S2, "  B2 =", B2, "  ΔS2 =", dS2)
print("Color (m2 - m1) [mag]:", c, "  Uncertainty [mag]:", dc)

# optional: if you want to see how much of B is from dark current specifically
Bdark1 = instr["dark_e_per_pix_s"] * n_pix_aperature * t_exp
Bdark2 = instr["dark_e_per_pix_s"] * n_pix_aperature * t_exp
print("Dark current contribution (both bands):", Bdark1, "electrons")

# ---- Extra: Sun, Reflectance, Asteroid flux (your earlier visuals) ----
Ssun = spectrum_sun(wavelength)
plt.figure(figsize=(8,5))
plt.plot(wavelength, Ssun)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Solar spectrum (arb. units)")
plt.title("Toy solar spectrum S_sun(λ)")
plt.grid(True); plt.show()

Rast = reflectance_asteroid(wavelength)
plt.figure(figsize=(8,5))
plt.plot(wavelength, Rast)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (unitless)")
plt.title("Asteroid reflectance R_ast(λ) (toy)")
plt.grid(True); plt.show()

S_ast = asteroid_flux_raw(wavelength)
plt.figure(figsize=(8,5))
plt.plot(wavelength, S_ast)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Asteroid flux (arb. units)")
plt.title("Asteroid photon spectrum Ṡ_ast(λ) ≈ S_sun × R_ast (toy)")
plt.grid(True); plt.show()
