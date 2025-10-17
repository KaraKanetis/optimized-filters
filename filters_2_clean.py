# Main code from now on
import numpy as np
import matplotlib.pyplot as plt
import math

# changed from github files to my local files
import importlib.util

LD_PATH = "/Users/karakanetis/Documents/spyder_transfer/data_files/load_data.py"
spec = importlib.util.spec_from_file_location("load_data_local", LD_PATH)
ld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ld)
print("USING load_data FROM:", ld.__file__)

# ----------------------------
# Variables
# ----------------------------
LAM_MIN_NM = 440.0   # Λ_min
LAM_MAX_NM = 920.0   # Λ_max

# test bands
BAND1 = (450.0, 650.0)
BAND2 = (730.0, 740.0)

# for wavelength plots
min_wavelength_nm = 400
max_wavelength_nm = 900
step_wavelength_nm = 10

# --------------------------------
# Load inputs from files
# --------------------------------
lam_full, curves = ld.load_all_curves(base_dir="data_files")
interp = ld.make_curve_interpolators(lam_full, curves)

f_sun_full    = interp["sun"]        # photons s^-1 m^-2 nm^-1
f_ast_full    = interp["asteroid"]   # unitless reflectance
f_atm_full    = interp["atm"]        # transmission (0–1)
f_qe_full     = interp["qe"]         # e-/photon
f_mirror_full = interp["mirror"]     # reflectivity / throughput (0–1)
f_sky_full    = interp["sky"]        # photons s^-1 m^-2 nm^-1 arcsec^-2


# Use the loader’s grid
mask_range = (lam_full >= LAM_MIN_NM) & (lam_full <= LAM_MAX_NM)
WAVELENGTH_NM = lam_full[mask_range]

S_sun_nm  = f_sun_full(WAVELENGTH_NM)
R_ast_nm  = f_ast_full(WAVELENGTH_NM)
T_atm_nm  = f_atm_full(WAVELENGTH_NM)
T_opt_nm  = f_mirror_full(WAVELENGTH_NM)
QE_nm     = f_qe_full(WAVELENGTH_NM)

# ----------------------------
# V-band constants
# ----------------------------
C_V_PHOT_M2_S_NM = 995.5e5  # photons m^-2 s^-1 nm^-1
V_MIN_NM = 505.0
V_MAX_NM = 595.0
DELTA_LAMBDA_V = V_MAX_NM - V_MIN_NM

# ----------------------------
# Instrument
# ----------------------------
instr = {
    "d_m": 1.0,            # telescope diameter [m]
    "G_e_per_ph": 1.0,     # CCD gain [e-/photon]
    "s_pix_m": 5e-6,       # pixel size [m]
    "read_noise_e": 5.0,   # read noise [e- rms] per pixel
    "dark_e_per_pix_s": 0,  # dark current [e-/pix/s]
}
t_exp_s = 60.0  # exposure time [s]

# ----------------------------
# Eq. (2) and Eq. (3)
# ----------------------------
def C_V_scale_from_Vmag(V_mag):
    """Solve Eq. (3) for C_V:
    C_V * ∫_{V band} S_sun(λ) R_ast(λ) dλ = C * 10^{-V/2.5} * Δλ_V
    """
    mask = (WAVELENGTH_NM >= V_MIN_NM) & (WAVELENGTH_NM <= V_MAX_NM)
    lhs_int = np.trapz(S_sun_nm[mask] * R_ast_nm[mask], WAVELENGTH_NM[mask])
    rhs = C_V_PHOT_M2_S_NM * (10.0 ** (-V_mag / 2.5)) * DELTA_LAMBDA_V
    return 0.0 if lhs_int <= 0 else (rhs / lhs_int)

def Sdot_ast_ph_per_m2_s_nm(V_mag):
    """Eq. (2): Ṡ_ast(V, λ) = C_V * Ṡ_sun(λ) * R_ast(λ) on the common grid."""
    C_V = C_V_scale_from_Vmag(V_mag)
    return C_V * S_sun_nm * R_ast_nm



# ----------------------------
# Eq. (4) Signal and Eq. (5) Background
# ----------------------------
def tophat_T(lam1_nm, lam2_nm):
    tf = np.zeros_like(WAVELENGTH_NM, dtype=float)
    tf[(WAVELENGTH_NM >= lam1_nm) & (WAVELENGTH_NM <= lam2_nm)] = 1.0
    return tf

def telescope_area_m2(d_m):
    return math.pi * (d_m ** 2) / 4.0

def A_squared_pixels(lambda2_nm, s_pix_m, d_m):
    """Eq. (6): A^2 = (π / (4 s_pix^2)) * arcsin(2.44 λ2 / d)^2"""
    lam2_m = lambda2_nm * 1e-9
    x = 2.44 * lam2_m / max(d_m, 1e-9)
    x = max(-1.0, min(1.0, x))
    return (math.pi / (4.0 * (s_pix_m ** 2))) * (math.asin(x) ** 2)

def Signal_e(lam1_nm, lam2_nm, V_mag, t_s, inst):
    """Eq. (4) detected signal electrons in band [λ1, λ2]."""
    T_band   = tophat_T(lam1_nm, lam2_nm)
    Sdot_ast = Sdot_ast_ph_per_m2_s_nm(V_mag)
    integrand = Sdot_ast * T_atm_nm * T_opt_nm * T_band * QE_nm
    A = telescope_area_m2(inst["d_m"])  # m^2
    return t_s * inst["G_e_per_ph"] * A * np.trapz(integrand, WAVELENGTH_NM)

def Background_e(lam1_nm, lam2_nm, V_mag, t_s, inst):
    T_band = tophat_T(lam1_nm, lam2_nm)
    Sdot_sky = f_sky_full(WAVELENGTH_NM)
    

    A2 = A_squared_pixels(lam2_nm, inst["s_pix_m"], inst["d_m"])   # Eq. (6)
    read_term = A2 * (inst["read_noise_e"] ** 2)                    # A^2 R^2

    integrand = Sdot_sky * T_atm_nm * T_opt_nm * T_band * QE_nm     # sky-only per Eq. (5)
    A_tel = telescope_area_m2(inst["d_m"])
    integ_term = A2 * t_s * inst["G_e_per_ph"] * A_tel * np.trapz(integrand, WAVELENGTH_NM)

    dark_term = 0.0  # enforce Ḋ(λ)=0 (paper’s neglect of dark current)
    return read_term + integ_term + dark_term

def snr_band(lam1_nm, lam2_nm, V_mag, t_s, inst):
    S = Signal_e(lam1_nm, lam2_nm, V_mag, t_s, inst)
    B = Background_e(lam1_nm, lam2_nm, V_mag, t_s, inst)
    dS = math.sqrt(max(S + B, 0.0))  # Eq. (7)
    return 0.0 if dS <= 0 else (S / dS)

# ----------------------------
# Eq. (8): color + uncertainty
# ----------------------------
def color_mag_eq8(l1_min, l1_max, l2_min, l2_max, V_mag, t_s, inst):
    S1 = Signal_e(l1_min, l1_max, V_mag, t_s, inst)
    B1 = Background_e(l1_min, l1_max, V_mag, t_s, inst)
    S2 = Signal_e(l2_min, l2_max, V_mag, t_s, inst)
    B2 = Background_e(l2_min, l2_max, V_mag, t_s, inst)

    dS1 = math.sqrt(max(S1 + B1, 0.0))
    dS2 = math.sqrt(max(S2 + B2, 0.0))

    eps = 1e-30

    c0 = 2.5 * math.log10(max(S1, eps) / max(S2, eps))
    # Eq. (8) uncertainty
    c_plus  = 2.5 * math.log10(max(S1 + dS1, eps) / max(S2 - dS2, eps))
    c_minus = 2.5 * math.log10(max(S1 - dS1, eps) / max(S2 + dS2, eps))

    dc = max(abs(c_plus - c0), abs(c0 - c_minus))
    return c0, dc, c_plus, c_minus

# ----------------------------
# Eq. 8 grid scan that returns the color, uncertainty, and best filter "merit"
# ----------------------------
def color_table_eq8(L_vals, V_mag, t_s, inst):
    """
    Build a table of (λ1, λ2, λ3, λ4, c, dc, merit) with merit = |c|/dc.
    """
    out = []
    L_vals = np.asarray(L_vals, float)
    n = len(L_vals)
    for i in range(n):
        l1 = L_vals[i]
        for j in range(i + 1, n):
            l2 = L_vals[j]
            for k in range(n):
                l3 = L_vals[k]
                for m in range(k + 1, n):
                    l4 = L_vals[m]
                    c, dc, _, _ = color_mag_eq8(l1, l2, l3, l4, V_mag, t_s, inst)
                    merit = abs(c) / max(dc, 1e-30)
                    out.append((l1, l2, l3, l4, c, dc, merit))
    return out

def print_sample_color_rows(rows, k=10):
    print(f"\n[Eq.8 color + uncertainty] Showing {min(k, len(rows))} of {len(rows)} rows:")
    for r in rows[:k]:
        l1, l2, l3, l4, c, dc, merit = r
        print(f"  [{l1:5.1f},{l2:5.1f}] vs [{l3:5.1f},{l4:5.1f}]"
              f"  ->  c={c: .4f} mag,  dc={dc: .4f} mag,  |c|/dc={merit: .3f}")

def best_bands_eq8(rows):
    """Pick the row with the maximum |c|/dc merit."""
    if not rows:
        return None
    return max(rows, key=lambda r: r[-1])

def best_bands_for_type(taxon_label, ids, smass_path, V_mag_rep, L_vals, t_s, inst):
    #taking in wavelengths to test
    # magnitude of types, exposure, insturment, asteroid IDs
    # combines all the spectra into a curve
    Ravg = _robust_type_average_on_common_grid(ids, smass_path)
    # uses the Ravg from now on
    _set_reflectance_on_common_grid(WAVELENGTH_NM, Ravg)
    # loops through every band and finds c, dc and its merit
    rows = color_table_eq8(L_vals, V_mag_rep, t_s, inst)
    #this puts it into the merit and finds the best one
    return best_bands_eq8(rows)

# ------------------------------------------------------------------
# PLOTTING HELPERS
# ------------------------------------------------------------------
def plot_spectrum_sun():
    plt.figure(figsize=(8,5))
    plt.plot(WAVELENGTH_NM, S_sun_nm)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Solar photons / s / m² / nm")
    plt.title("Solar Spectrum  S_sun(λ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spectrum_atm():
    plt.figure(figsize=(8,5))
    plt.plot(WAVELENGTH_NM, T_atm_nm)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission (0–1)")
    plt.title("Atmospheric Transmission  T_atm(λ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spectrum_qe():
    plt.figure(figsize=(8,5))
    plt.plot(WAVELENGTH_NM, QE_nm)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Electrons per photon")
    plt.title("CCD Quantum Efficiency  QE(λ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spectrum_mirror():
    plt.figure(figsize=(8,5))
    plt.plot(WAVELENGTH_NM, T_opt_nm)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectivity / Throughput (0–1)")
    plt.title("Optics Throughput  T_opt(λ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spectrum_sky():
    Sdot_sky = f_sky_full(WAVELENGTH_NM)
    plt.figure(figsize=(8, 5))
    plt.plot(WAVELENGTH_NM, Sdot_sky)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Sky photons / s / m² / nm / arcsec²")
    plt.title("Sky Background Spectrum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
# --- swap asteroid reflectance on the common grid (overwrite R_ast_nm) ---
def _set_reflectance_on_common_grid(wl_src_nm, R_src):
    """Interpolate a (wl, R) pair onto WAVELENGTH_NM and overwrite R_ast_nm."""
    global R_ast_nm
    R_ast_nm = np.interp(WAVELENGTH_NM, np.asarray(wl_src_nm, float), np.asarray(R_src, float))

def _load_smass_spectrum_from_file(path_txt: str, id_str: str):
    """Return (wavelength_nm, reflectance) for a SMASS object ID from the table."""
    raw = np.loadtxt(
        path_txt,
        dtype={"names": ("id", "w_um", "R"), "formats": ("U16", "f8", "f8")},
    )
    mask = (raw["id"] == id_str)
    wl_nm = raw["w_um"][mask] * 1_000.0  # µm → nm
    R = raw["R"][mask]
    return wl_nm, R

# per-nm integrands (electrons per nm), using current V and t
def _signal_integrand_e_per_nm(V_mag, t_s, inst):
    A = telescope_area_m2(inst["d_m"])
    return t_s * inst["G_e_per_ph"] * A * (Sdot_ast_ph_per_m2_s_nm(V_mag) * T_atm_nm * T_opt_nm * QE_nm)

def _background_integrand_e_per_nm_per_as2(V_mag_unused, t_s, inst):
    Sdot_sky = f_sky_full(WAVELENGTH_NM)
    A = telescope_area_m2(inst["d_m"])
    return t_s * inst["G_e_per_ph"] * A * (Sdot_sky * T_atm_nm * T_opt_nm * QE_nm)

def _cumulative_from_integrand(y_e_per_nm):
    dlam = np.gradient(WAVELENGTH_NM)
    return np.cumsum(y_e_per_nm * dlam)

def plot_signal_integrand(V_mag=10.0, t_s=None, inst=instr):
    if t_s is None:
        t_s = t_exp_s
    y_sig = _signal_integrand_e_per_nm(V_mag, t_s, inst)
    plt.figure(figsize=(8,5))
    plt.plot(WAVELENGTH_NM, y_sig)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Electrons / nm")
    plt.title("Signal Integrand  (electrons per nm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_background_integrand(V_mag_ignored=10.0, t_s=None, inst=instr):
    if t_s is None:
        t_s = t_exp_s
    y_bkg = _background_integrand_e_per_nm_per_as2(V_mag_ignored, t_s, inst)
    plt.figure(figsize=(8,5))
    plt.plot(WAVELENGTH_NM, y_bkg)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Electrons / nm / arcsec²")
    plt.title("Sky Background Integrand  (electrons per nm per arcsec²)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_signal_and_background(V_mag=10.0, t_s=None, inst=instr):
    if t_s is None:
        t_s = t_exp_s
    sig = _signal_integrand_e_per_nm(V_mag, t_s, inst)
    bkg = _background_integrand_e_per_nm_per_as2(V_mag, t_s, inst)
    plt.figure(figsize=(8,5))
    plt.plot(WAVELENGTH_NM, sig, label="Signal (e⁻ / nm)")
    plt.plot(WAVELENGTH_NM, bkg, label="Background (e⁻ / nm / arcsec²)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Electrons")
    plt.title("Signal vs Background per Wavelength")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cumulative_curves(V_mag=10.0, t_s=None, inst=instr):
    if t_s is None:
        t_s = t_exp_s
    sig = _signal_integrand_e_per_nm(V_mag, t_s, inst)
    bkg = _background_integrand_e_per_nm_per_as2(V_mag, t_s, inst)
    cum_sig = _cumulative_from_integrand(sig)
    cum_bkg_as2 = _cumulative_from_integrand(bkg)
    plt.figure(figsize=(8,5))
    plt.plot(WAVELENGTH_NM, cum_sig, label="Signal (electrons)")
    plt.plot(WAVELENGTH_NM, cum_bkg_as2, label="Background (electrons / arcsec²)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Cumulative Electrons")
    plt.title("Cumulative Signal and Background")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_input_curves(normalize=False):
    """Plot the main input curves: Sun, atmosphere, optics, QE, and sky."""
    Sdot_sky = f_sky_full(WAVELENGTH_NM)

    def _norm(y):
        y = np.asarray(y, float)
        return y / np.nanmax(y) if normalize and np.nanmax(y) > 0 else y

    plt.figure(figsize=(8, 5))
    plt.plot(WAVELENGTH_NM, _norm(S_sun_nm), label="Sun (phot s⁻¹ m⁻² nm⁻¹)")
    plt.plot(WAVELENGTH_NM, _norm(T_atm_nm), label="Atmosphere (transmission)")
    plt.plot(WAVELENGTH_NM, _norm(T_opt_nm), label="Optics (throughput)")
    plt.plot(WAVELENGTH_NM, _norm(QE_nm), label="CCD QE (e⁻/photon)")
    plt.plot(WAVELENGTH_NM, _norm(Sdot_sky), label="Sky (phot s⁻¹ m⁻² nm⁻¹ arcsec⁻²)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized" if normalize else "Native units")
    plt.title("Model input curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_snr_for_asteroid(id_str, smass_path, instr, t_exp_s, V_mag_for_id,
                          min_nm=400, max_nm=900, step_nm=10):
    # 1) install that asteroid’s reflectance
    wl_nm, R = _load_smass_spectrum_from_file(smass_path, id_str)
    _set_reflectance_on_common_grid(wl_nm, R)

    # 2) build SNR map
    L1_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L2_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L1, L2 = np.meshgrid(L1_vals, L2_vals)
    mask = L2 > L1

    SNR_vals = np.full_like(L1, np.nan, dtype=float)
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            if mask[i, j]:
                SNR_vals[i, j] = snr_band(float(L1[i, j]), float(L2[i, j]),
                                          V_mag_for_id, t_exp_s, instr)

    plt.figure(figsize=(7, 6))
    pc = plt.pcolormesh(L1, L2, SNR_vals, shading="auto")
    plt.colorbar(pc, label="SNR")
    plt.xlabel("Band min λ (nm)")
    plt.ylabel("Band max λ (nm)")
    plt.title(f"SNR for asteroid {id_str}")
    plt.tight_layout()
    plt.show()

def _robust_type_average_on_common_grid(ids, smass_path, norm_nm=550.0, sigma=3.0):
    """Median+MAD clip (3σ) → mean, normalized so R(norm_nm)=1, on WAVELENGTH_NM."""
    stacks = []
    for aid in ids:
        try:
            wl, R = _load_smass_spectrum_from_file(smass_path, aid)
            Rg = np.interp(WAVELENGTH_NM, wl, R)
            n = np.interp(norm_nm, WAVELENGTH_NM, Rg)
            if np.isfinite(n) and n > 0:
                Rg = Rg / n
            stacks.append(Rg)
        except Exception:
            pass
    if not stacks:
        raise RuntimeError("No usable spectra for that type.")
    X = np.vstack(stacks)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    keep = np.abs(X - med) <= (sigma * 1.4826 * mad)
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(np.where(keep, X, np.nan), axis=0)
    avg = np.where(np.isfinite(avg), avg, med)
    return avg

def plot_snr_for_type_avg(taxon_label, ids, smass_path, instr, t_exp_s, V_mag_rep,
                          min_nm=400, max_nm=900, step_nm=10):
    Ravg = _robust_type_average_on_common_grid(ids, smass_path)
    _set_reflectance_on_common_grid(WAVELENGTH_NM, Ravg)

    L1_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L2_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L1, L2 = np.meshgrid(L1_vals, L2_vals)
    mask = L2 > L1

    SNR_vals = np.full_like(L1, np.nan, dtype=float)
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            if mask[i, j]:
                SNR_vals[i, j] = snr_band(float(L1[i, j]), float(L2[i, j]),
                                          V_mag_rep, t_exp_s, instr)

    plt.figure(figsize=(7, 6))
    pc = plt.pcolormesh(L1, L2, SNR_vals, shading="auto")
    plt.colorbar(pc, label="SNR")
    plt.xlabel("Band min λ (nm)")
    plt.ylabel("Band max λ (nm)")
    plt.title(f"SNR for {taxon_label}-avg")
    plt.tight_layout()
    plt.show()


# --- Optional: robust stack plot (kept same spirit as your second file) ---
def _interp_and_norm_to_grid(wl_src_nm, R_src, grid_nm, norm_nm=550.0):
    Rg = np.interp(grid_nm, np.asarray(wl_src_nm, float), np.asarray(R_src, float))
    if norm_nm is not None:
        n = np.interp(norm_nm, grid_nm, Rg)
        if np.isfinite(n) and n > 0:
            Rg = Rg / n
    return Rg

def load_smass_spectrum_from_file(path_txt: str, id_str: str):
    raw = np.loadtxt(
        path_txt,
        dtype={"names": ("id", "w_um", "R"), "formats": ("U16", "f8", "f8")},
    )
    mask = (raw["id"] == id_str)
    wl_nm = raw["w_um"][mask] * 1_000.0
    R = raw["R"][mask]
    return wl_nm, R

def robust_average_spectra(id_list, smass_path, grid_nm, norm_nm=550.0, sigma=3.0):
    curves, kept_ids = [], []
    for aid in id_list:
        try:
            wl_nm, R = load_smass_spectrum_from_file(smass_path, aid)
            Rg = _interp_and_norm_to_grid(wl_nm, R, grid_nm, norm_nm=norm_nm)
            if np.all(~np.isfinite(Rg)):
                continue
            curves.append(Rg); kept_ids.append(aid)
        except Exception:
            continue
    if not curves:
        raise RuntimeError("No usable spectra.")
    X = np.vstack(curves)
    median = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - median), axis=0)
    thresh = sigma * 1.4826 * mad
    keep = np.abs(X - median) <= thresh
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(np.where(keep, X, np.nan), axis=0)
    avg = np.where(np.isfinite(avg), avg, median)
    return {"avg": avg, "median": median, "mad": mad, "mask": keep, "stack": X, "ids": kept_ids}

def get_ids_for_type_from_taxonomy(taxonomy_map, prefix):
    P = str(prefix).upper()
    return [aid for aid, cls in taxonomy_map.items() if str(cls).upper().startswith(P)]

def plot_type_stack_with_outliers(
    taxon: str,
    smass_path: str,
    ids=None,
    grid_nm=WAVELENGTH_NM,
    norm_nm: float = 550.0,
    sigma: float = 3.0,
    keep_threshold: float = 0.80,
    show_median: bool = True,
    title_suffix: str = "",
):
    if ids is None:
        raise RuntimeError("Provide ids= list for the given taxon.")
    res = robust_average_spectra(ids, smass_path, grid_nm, norm_nm=norm_nm, sigma=sigma)
    X = res["stack"]
    keep_mask = res["mask"]
    with np.errstate(invalid="ignore"):
        frac_kept = np.nanmean(keep_mask, axis=1)
    frac_kept = np.where(np.isfinite(frac_kept), frac_kept, 0.0)
    kept_idx = np.where(frac_kept >= keep_threshold)[0]
    out_idx  = np.where(frac_kept <  keep_threshold)[0]

    plt.figure(figsize=(8, 5))
    for k in out_idx:
        plt.plot(grid_nm, X[k], lw=1.0, alpha=0.35, color="tab:red", label="Outlier" if k == out_idx[0] else None)
    for k in kept_idx:
        plt.plot(grid_nm, X[k], lw=1.0, alpha=0.20, label=None)
    if show_median and "median" in res:
        plt.plot(grid_nm, res["median"], lw=1.5, linestyle="--", label="Per-λ median")
    plt.plot(grid_nm, res["avg"], lw=2.5, label=f"{taxon}-type robust average")
    n_tot = X.shape[0]; n_kept = kept_idx.size; n_out = out_idx.size
    ttl = f"{taxon}-type spectra: all inputs (kept vs outliers) + robust average"
    if title_suffix:
        ttl += f" — {title_suffix}"
    plt.title(ttl)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(f"Reflectance (norm @ {norm_nm:.0f} nm)")
    plt.grid(True, alpha=0.3)
    plt.legend(title=f"N_total={n_tot}, kept≈{n_kept}, outliers≈{n_out}", loc="best")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Load and print SNRs
# Swap True/False for SNR plots
# ----------------------------
if __name__ == "__main__":
    SHOW_ASTEROID_SNR = False
    SHOW_TYPE_AVG_SNR = False

    V_example = 10.0
    # new file paths
    SMASS_PATH = "/Users/karakanetis/Documents/spyder_transfer/smass2_all_spfit.txt"
    TAX_PATH   = "/Users/karakanetis/Documents/spyder_transfer/taxonomy.pds.table.txt"

    # 2) taxonomy -> ids per type (C/S/V)
    taxonomy_map = ld.load_taxonomy_table(TAX_PATH)
    def ids_for_type(prefix: str):
        P = prefix.upper()
        return [aid for aid, cls in taxonomy_map.items() if str(cls).upper().startswith(P)]
    type_sets = {"C": ids_for_type("C"), "S": ids_for_type("S"), "V": ids_for_type("V")}
    if SHOW_ASTEROID_SNR:
    # 3) per-asteroid examples (000024 C, 000006 S, 000004 V)
        ID_TO_Vmag = {"000004": 11.0, "000006": 8.0, "000024": 10.0}
        for aid in ("000024", "000006", "000004"):
            plot_snr_for_asteroid(aid, SMASS_PATH, instr, t_exp_s, ID_TO_Vmag[aid])
    if SHOW_TYPE_AVG_SNR:
        # 4) per-type averages (representative V mags)
        mV_by_type = {"C": 10.0, "S": 8.5, "V": 6.5}
        for taxon in ("V", "S", "C"):
            ids = type_sets[taxon]
            if ids:
                plot_snr_for_type_avg(f"{taxon}", ids, SMASS_PATH, instr, t_exp_s, mV_by_type[taxon])
    # ================================================================
    # Looks at eqn 8 and prints out the sorted top 10 and the merit
    L_vals = np.arange(440, 921, 10)  # same as your scan grid
    rows = color_table_eq8(L_vals, V_example, t_exp_s, instr)

    # Print a few examples and the best set
    rows_sorted = sorted(rows, key=lambda r: r[-1], reverse=True)
    print_sample_color_rows(rows_sorted, k=10)
    best = best_bands_eq8(rows)
    if best:
        l1, l2, l3, l4, c, dc, merit = best
        print(f"\nBest by |c|/dc:\n  band1=[{l1:.1f},{l2:.1f}], band2=[{l3:.1f},{l4:.1f}]"
              f"  ->  c={c:.4f} mag, dc={dc:.4f} mag, |c|/dc={merit:.3f}") # bad at this format, asked Ai to fill in
    
    # test print for eqn 8 values
    L_vals = np.arange(440, 921, 10)
    mV_by_type = {"C": 10.0, "S": 8.5, "V": 6.5}

    for taxon in ("V", "S", "C"):
        ids = type_sets.get(taxon, [])
        if not ids:
            print(f"[{taxon}] No IDs found; skipping.")
            continue
        best = best_bands_for_type(taxon, ids, SMASS_PATH, mV_by_type[taxon], L_vals, t_exp_s, instr)
        if best:
            l1, l2, l3, l4, c, dc, merit = best
            print(f"[{taxon}] Best bands by |c|/dc: "
                  f"band1=[{l1:.1f},{l2:.1f}], band2=[{l3:.1f},{l4:.1f}]  "
                  f"c={c:.4f} mag, dc={dc:.4f} mag, |c|/dc={merit:.1f}") #ai formatted



    # ----------------------------------------------------------------
    # Extra test plots
    # NOTICE: this prints after the best bands- so it will take a bit longer to print
    # ----------------------------------------------------------------
    # plot_spectrum_sun()
    # plot_spectrum_atm()
    # plot_spectrum_qe()
    # plot_spectrum_mirror()
    # plot_spectrum_sky()
    # plot_signal_integrand(V_mag=V_example, t_s=t_exp_s, inst=instr)
    # plot_background_integrand(V_mag_ignored=V_example, t_s=t_exp_s, inst=instr)
    # plot_signal_and_background(V_mag=V_example, t_s=t_exp_s, inst=instr)
    # plot_cumulative_curves(V_mag=V_example, t_s=t_exp_s, inst=instr)
    # plot_input_curves(normalize=True)
    # if type_sets["S"]:
    #    _ = plot_type_stack_with_outliers("S", SMASS_PATH, ids=type_sets["S"], sigma=3.0, keep_threshold=0.8)
