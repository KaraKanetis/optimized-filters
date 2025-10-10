# reading files, no more dummy numbers
# removing the unit conversion from here and putting it into the file reader

import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import math
import SMASS as smass



data = smass.load_smass2()
#wavelength_nm, reflectance = smass.plot_smass2_spectrum('000004')
smass.plot_all_smass2_spectra_of_type( 'V' )

# read curves once
lam, curves = ld.load_all_curves(base_dir="data_files")
interp = ld.make_curve_interpolators(lam, curves)
f_sun    = interp["sun"]
f_ast    = interp["asteroid"]
f_atm    = interp["atm"]
f_qe     = interp["qe"]
f_mirror = interp["mirror"]

COMMON_GRID_NM = np.arange(300.0, 1201.0, 1.0)


wavelength = lam  # use the same grid the loaders produced (1-nm from 300–1200)
#To use 400–900 only:
min_wavelength_nm = 400
max_wavelength_nm = 900
step_wavelength_nm = 10
wavelength_nm = np.arange( min_wavelength_nm, max_wavelength_nm, 1.0)
wavelength_index = np.array( len(wavelength_nm) )

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

# def make_bands(V_min_nm, V_max_nm, V_width_nm):
#     return {
#         "V_min": V_min_nm,
#         "V_max": V_max_nm,
#         "V_width": V_width_nm,
#     }

# instrument & exposure
instr = make_instrument(
    d_m=1.0,
    gain_e_per_ph=1.0,
    s_pix_m=5e-6,
    read_noise_e=5.0,
    dark_e_per_pix_s=0.2,
)
t_exp_s = 60.0  # seconds

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
def signal(wavelength_nm,instr, t_exp_s):
    """
    Per-wavelength contribution to detected signal (Eq. 4 BEFORE integrating).
    Returns array same shape as wavelength_nm.
    """
    #wavelength = np.asarray(wavelength_nm, float)
    S_ast = asteroid_flux_raw(wavelength_nm)
    Tatm  = transmission_atm(wavelength_nm)
    Topt  = transmission_optics(wavelength_nm)
    # Tf    = transmission_filter_placeholder(wavelength_nm, wavelength1_nm, wavelength2_nm)
    QE    = ccd_qe(wavelength_nm)

    area_m2 = np.pi * (instr["d"] ** 2) / 4.0
    # return t_exp_s * instr["gain"] * area_m2 * (S_ast * Tatm * Topt * Tf * QE)
    return t_exp_s * instr["gain"] * area_m2 * (S_ast * Tatm * Topt * QE)


# def detected_signal_band(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s):
#     """Total detected signal (electrons) in band [λ1, λ2] via trapezoidal integral over λ."""
#     integrand = signal(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s)
#     return float(np.trapz(integrand, wavelength_nm))

def background(wavelength_nm, instr, t_exp_s):
    """Per-wavelength background contribution (placeholder sky × optics × QE × filter)."""
    wavelength = np.asarray(wavelength_nm, float)
    Ssky = sky_photon_spectrum(wavelength)
    Tatm = transmission_atm(wavelength)
    Topt = transmission_optics(wavelength)
    # Tf   = transmission_filter_placeholder(wavelength, wavelength1_nm, wavelength2_nm)
    QE   = ccd_qe(wavelength)
    area_m2 = np.pi * (instr["d"] ** 2) / 4.0
    # return t_exp_s * instr["gain"] * area_m2 * (Ssky * Tatm * Topt * Tf * QE)
    return t_exp_s * instr["gain"] * area_m2 * (Ssky * Tatm * Topt * QE)

# def detected_background_band(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s, n_pix_aperture):
#     """Total background (electrons) in band [λ1, λ2] (sky + dark current)."""
#     b_int = background_integrand(wavelength_nm, wavelength1_nm, wavelength2_nm, instr, t_exp_s)
#     B_photon = float(np.trapz(b_int, wavelength_nm))
#     B_dark   = instr["dark_e_per_pix_s"] * float(n_pix_aperture) * float(t_exp_s)
#     return B_photon + B_dark

# --------------------------------------------
# Uncertainty and color helpers
# # --------------------------------------------
# def delta_S(S, B, instr, n_pix_aperture):
#     """
#     Shot noise (S + B) plus read noise per pixel × number of pixels (quadrature).
#     Eq. (7)
#     """
#     R = float(instr["read_noise"])   # <-- FIX: correct key
#     var = float(S) + float(B) + float(n_pix_aperture) * (R**2)
#     return float(np.sqrt(max(var, 0.0)))

# def color_mag(S1, dS1, S2, dS2):
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
def cumulative_signal(wavelength_nm,instr,t_exp_s):
    """Cumulative integral from min(λ) up to each λ_i."""
    S = signal(wavelength_nm,instr,t_exp_s)
    dlam_nm = np.gradient(wavelength_nm)            # Δλ at each point
    return np.cumsum( S * dlam_nm )

def cumulative_background(wavelength_nm,instr,t_exp_s):
    """Cumulative integral from min(λ) up to each λ_i."""
    B = background(wavelength_nm,instr,t_exp_s)
    dlam_nm = np.gradient(wavelength_nm)            # Δλ at each point
    return np.cumsum( B * dlam_nm )

gCumSig   = cumulative_signal(    wavelength_nm,instr,t_exp_s)

gCumBkgnd = cumulative_background(wavelength_nm,instr,t_exp_s)

#<<<<<<< HEAD:filters_rob.py
def SNR( lambda1_nm, lambda2_nm ):
    index1 = np.int32( ( lambda1_nm - min_wavelength_nm ) / step_wavelength_nm )
    index2 = np.int32( ( lambda2_nm - min_wavelength_nm ) / step_wavelength_nm )
    print(" Index 1/2, Lam1/2: ", index1, index2, lambda1_nm, lambda2_nm)
#=======
def SNR( lamba1_nm, lamba2_nm ):
    index1 = np.int32( ( lamba1_nm - min_wavelength_nm ) / step_wavelength_nm )
    index2 = np.int32( ( lamba2_nm - min_wavelength_nm ) / step_wavelength_nm )
    print(" Index 1/2, Lam1/2: ", index1, index2, lamba1_nm, lamba2_nm)
    
#>>>>>>> 5a865e6f79682b24b4275fff7ba56037a039df23:main_code/filters_rob.py
    signal     =   gCumSig[index2] -   gCumSig[index1]
    background = gCumBkgnd[index2] - gCumBkgnd[index1]
    return signal / np.sqrt( signal + background )

# ===========================
# TEST / PLOTS
# ===========================
# wavelength grid for everything (match the common grid)
# wavelength = lam  # use the same grid the loaders produced (1-nm from 300–1200)
# #To use 400–900 only:
# min_wavelength_nm = 400
# max_wavelength_nm = 900
# wavelength_nm = np.arange( min_wavelength_nm, max_wavelength_nm, 1.0)



# two example bands
band1 = (450, 650)
band2 = (730, 740)

# aperture size
n_pix_aperture = 50

# # ---- Plot transmissions and QE ----
# Tatm = transmission_atm(wavelength)
# Topt = transmission_optics(wavelength)
# Tf   = transmission_filter_placeholder(wavelength, 460, 470)  # simple demo filter
# QE   = ccd_qe(wavelength)

# plt.figure(figsize=(8,5))
# plt.plot(wavelength, Tatm, label="Atmosphere T(λ)")
# plt.plot(wavelength, Topt, label="Optics throughput")
# plt.plot(wavelength, Tf,   label="Filter 460–470 nm")
# plt.plot(wavelength, QE,   label="CCD QE")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Transmission / QE")
# plt.title("Measured transmissions & CCD QE")
# plt.legend(); plt.grid(True); plt.show()

# # ---- Per-λ detected signal (integrand) for band1 ----
# signal_pw = signal(wavelength, instr, t_exp_s)

# plt.figure(figsize=(8,5))
# plt.plot(wavelength, signal_pw, label=f"per-λ signal in [{band1[0]}, {band1[1]}] nm")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Electrons per nm (scaled)")
# plt.title("Per-λ detected signal (integrand of Eq. 4)")
# plt.legend(); plt.grid(True); plt.show()

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

# # ---- Total band signals (scalars) and color demo ----
# S1 = detected_signal_band(wavelength, band1[0], band1[1], instr, t_exp_s)
# S2 = detected_signal_band(wavelength, band2[0], band2[1], instr, t_exp)
# B1 = detected_background_band(wavelength, band1[0], band1[1], instr, t_exp, n_pix_aperture)
# B2 = detected_background_band(wavelength, band2[0], band2[1], instr, t_exp, n_pix_aperture)
# SNR = S1 / math.sqrt(S1+B1)


# dS1 = delta_S(S1, B1, instr, n_pix_aperture)
# dS2 = delta_S(S2, B2, instr, n_pix_aperture)
# c, dc = color_mag(S1, dS1, S2, dS2)

# print("Band 1:", band1, "  S1 =", S1, "  B1 =", B1, "  ΔS1 =", dS1)
# print("Band 2:", band2, "  S2 =", S2, "  B2 =", B2, "  ΔS2 =", dS2)
# print("Color (m2 - m1) [mag]:", c, "  Uncertainty [mag]:", dc)
# print("SNR", SNR)

# # Dark current contribution
# Bdark = instr["dark_e_per_pix_s"] * n_pix_aperture * t_exp_s
# print("Dark current contribution (per band):", Bdark, "electrons")

# # ---- Sun, Reflectance, Asteroid flux ----
# Ssun = spectrum_sun(wavelength)
# plt.figure(figsize=(8,5))
# plt.plot(wavelength, Ssun)
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Solar photons / s / m² / nm (scaled)")
# plt.title("Solar spectrum S_sun(λ) (from file)")
# plt.grid(True); plt.show()

# Rast = reflectance_asteroid(wavelength)
# plt.figure(figsize=(8,5))
# plt.plot(wavelength, Rast)
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Reflectance (unitless)")
# plt.title("Asteroid reflectance R_ast(λ) (from file)")
# plt.grid(True); plt.show()

# S_ast = asteroid_flux_raw(wavelength)
# plt.figure(figsize=(8,5))
# plt.plot(wavelength, S_ast)
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Asteroid photons (scaled)")
# plt.title("Asteroid photon spectrum Ṡ_ast(λ) = S_sun × R_ast")
# plt.grid(True); plt.show()

# # -- background per wavelength (using placeholder sky)
# b_pw = background(wavelength, instr, t_exp_s)
# plt.figure(figsize=(8,5))
# plt.plot(wavelength, b_pw, label="per-λ background (sky)")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Electrons per nm (scaled)")
# plt.title(f"Per-λ background inside [{band1[0]}, {band1[1]}] nm")
# plt.legend(); plt.grid(True); plt.show()




# # 
# def SNR(lambda1_nm, lambda2_nm):
#     S1 = detected_signal_band(wavelength, lambda1_nm, lambda2_nm, instr, t_exp)
#     B1 = detected_background_band(wavelength, lambda1_nm, lambda2_nm, instr, t_exp, n_pix_aperture)
#     SNR = S1 / math.sqrt(S1+B1)
#     return SNR

# Define ranges
L1_vals = np.arange(min_wavelength_nm, max_wavelength_nm+1, step_wavelength_nm)
L2_vals = np.arange(min_wavelength_nm, max_wavelength_nm+1, step_wavelength_nm)

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
plt.xlabel("min wavelength")
plt.ylabel("max wavelength")
plt.title("SNR vs band min and max wavelength")
plt.show()

#plot 3 together
wavelength_s, reflectance_s = smass.plot_smass2_spectrum('000003')  
wavelength_v, reflectance_v = smass.plot_smass2_spectrum('000004')  
wavelength_c, reflectance_c = smass.plot_smass2_spectrum('000010')  

# plt.figure(figsize=(8,5))
# plt.plot(wavelength_s, reflectance_s, label="00003")
# plt.plot(wavelength_v, reflectance_v, label="00004")
# plt.plot(wavelength_c, reflectance_c, label="000010")

# plt.xlabel("Wavelength")
# plt.ylabel("Reflectance")
# plt.title("S, V, C types")
# plt.legend()
# plt.grid(True)
# plt.show()

# maybe this fixes my issue of the plots all looking the same?
wavelength_nm = lam 



# change f_ast to this new data
# wl_src... is wavelengths
# r_src is reflectance
# asked Chat GPT to look at this,
# pasted in its answer to try to see if it will fix?
def set_reflectance_from_arrays(wl_src_nm, R_src):
    """Override f_ast so the pipeline uses this SMASS reflectance."""
    global f_ast
    wl_src_nm = np.asarray(wl_src_nm, float)
    R_src     = np.asarray(R_src, float)
    # interpolate the SMASS curve onto the SAME grid you integrate on
    R_on_grid = np.interp(wavelength_nm, wl_src_nm, R_src)
    # new asteroid reflectance function used by signal()
    f_ast = lambda x_nm: np.interp(np.asarray(x_nm, float), wavelength_nm, R_on_grid)


# chose one asteroid
def load_smass_spectrum_from_file(path_txt, id_str):
    #looks at format
    # reminder: 000001 0.44 0.9281 = ID, wavelength in microns, reflectance
    raw = np.loadtxt(path_txt, dtype={"names": ("id","w","r"),
                                      "formats": ("U16","f8","f8")})
    mask = (raw["id"] == id_str)
    wl_um = raw["w"][mask] # wavelength in microns
    R = raw["r"][mask] # reflectance
    wl_nm = wl_um * 1000.0 # convert um → nm
    return wl_nm, R

# Find SNR for one band
# looping through each one to creat ea full grid/plot
# the left and right should find that min/max
def snr_band(bmin_nm, bmax_nm):
    i1 = int(np.searchsorted(wavelength_nm, bmin_nm, side="left"))
    i2 = int(np.searchsorted(wavelength_nm, bmax_nm, side="right") - 1)

    # total signal and background
    S_band = float(gCumSig[i2] - gCumSig[i1])
    B_band = float(gCumBkgnd[i2] - gCumBkgnd[i1])

    # add dark current from earlier
    B_dark = instr["dark_e_per_pix_s"] * n_pix_aperture * t_exp_s
    B_band += B_dark

    #noise = sqrt(signal+background+read noise)
    R = instr["read_noise"]
    sigma = math.sqrt(S_band + B_band + n_pix_aperture * (R**2))
    return S_band/sigma


# gonna wrap this in a function so I can input multiple asteroids
def plot_snr(id_str, smass_path, instr, t_exp_s, n_pix_aperture):
    # finds asteroid from the file
    wl_nm, R = load_smass_spectrum_from_file(smass_path, id_str)

    # swap the reflectance
    set_reflectance_from_arrays(wl_nm, R)

    # 3remake the cum with new data
    global gCumSig, gCumBkgnd
    gCumSig   = cumulative_signal(wavelength_nm, instr, t_exp_s)
    gCumBkgnd = cumulative_background(wavelength_nm, instr, t_exp_s)

    # the ranges like from earlier
    min_wavelength_nm = 400
    max_wavelength_nm = 900
    step_wavelength_nm = 10
    L1_vals = np.arange(min_wavelength_nm, max_wavelength_nm+1, step_wavelength_nm)
    L2_vals = np.arange(min_wavelength_nm, max_wavelength_nm+1, step_wavelength_nm)
    L1, L2 = np.meshgrid(L1_vals, L2_vals)
    mask = L2 > L1

    # This basic loop fills in the grid pretty much
    SNR_vals = np.full_like(L1, np.nan, dtype=float)
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            if mask[i, j]:
                SNR_vals[i, j] = snr_band(float(L1[i, j]), float(L2[i, j]))

    # basic plot, copied from above
    plt.figure(figsize=(7,6))
    c = plt.pcolormesh(L1, L2, SNR_vals, shading="auto", cmap="viridis")
    plt.colorbar(c, label="SNR")
    plt.xlabel("min wavelength (nm)")
    plt.ylabel("max wavelength (nm)")
    plt.title(f"SNR vs band min/max")
    plt.show()

#pick the asteroid
smass_path = "/Users/karakanetis/Documents/GitHub/optimized-filters/main_code/smass2_all_spfit.txt"

plot_snr("000004", smass_path, instr, t_exp_s, n_pix_aperture)  # Vesta
plot_snr("000001", smass_path, instr, t_exp_s, n_pix_aperture)  # Ceres
plot_snr("000003", smass_path, instr, t_exp_s, n_pix_aperture)  # Juno
