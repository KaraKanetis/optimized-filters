import math
import os
import sys
from collections import Counter # this helps count the occourance of types so I can double check it all works
from typing import Dict, List, Tuple # better habits so code is easier to understand/read

import matplotlib.pyplot as plt
import numpy as np
import SMASS as smass  # your helper (unchanged)

# -----------------------------------------------------------------------------
# Loads the data files here
# -----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))  # this fixes my issues with the load.data file, Ai created
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from data_files import load_data as ld  # noqa: E402

print("Using load_data from:", ld.__file__)  # sanity print

# -----------------------------------------------------------------------------
# Load solar / atmosphere / QE / mirror curves
# All λ are on a common 1-nm grid. Units:
#   f_sun(λ_nm)      -> photons s^-1 m^-2 nm^-1
#   f_ast(λ_nm)      -> unitless reflectance (0–1; can be normalized at 550 nm)
#   f_atm(λ_nm)      -> unitless transmission (0–1)
#   f_qe(λ_nm)       -> e- per photon  (Quantum Efficiency)
#   f_mirror(λ_nm)   -> unitless reflectivity (0–1)
# -----------------------------------------------------------------------------
wavelength_nm, curves = ld.load_all_curves(base_dir="data_files")
interp = ld.make_curve_interpolators(wavelength_nm, curves)
f_sun = interp["sun"]
f_ast = interp["asteroid"]
f_atm = interp["atm"]
f_qe = interp["qe"]
f_mirror= interp["mirror"]

# -----------------------------------------------------------------------------
# Instrument description
# -----------------------------------------------------------------------------
def make_instrument(
    d_m: float,                  # mirror diameter [meters]
    s_pix_m: float,              # pixel size [meters]
    read_noise_e: float,         # read noise [electrons rms] per pixel
    dark_e_per_pix_s: float = 0.0,  # dark current [electrons / pixel / second]
) -> Dict[str, float]:
    """
    Return a tiny dict describing the camera/telescope. 
    """
    return {
        "d_m": d_m,
        "s_pix_m": s_pix_m,
        "read_noise_e": read_noise_e,
        "dark_e_per_pix_s": dark_e_per_pix_s,
    }

instr = make_instrument(
    d_m=1.0,
    s_pix_m=5e-6,
    read_noise_e=5.0,
    dark_e_per_pix_s=0.2,
)

t_exp_s: float = 600.0  # exposure time [seconds]

# -----------------------------------------------------------------------------
# Physics constants
# -----------------------------------------------------------------------------
_H = 6.62607015e-34  # Planck [J*s]
_C = 2.99792458e8    # speed of light [m/s]

# -----------------------------------------------------------------------------
# Helper: AB magnitude → photon flux per nm at λ0
# Ai added this because my SNR wasn't looking accurate
# -----------------------------------------------------------------------------
def scale_factor_from_AB_mag(m_AB: float, lambda0_nm: float = 550.0) -> float:
    """
    Return a dimensionless scale factor K that sets the **absolute brightness** of the
    asteroid spectrum without changing its shape.

    We choose K so that, at a reference wavelength λ0 (in nm), the monochromatic photon flux
    predicted by the model matches the photon flux implied by an object of AB magnitude m_AB:

        K * f_sun(λ0) * R_ast(λ0)  =  Φ_AB(m_AB, λ0)

    where:
      • f_sun(λ) is the solar spectrum in photons s⁻¹ m⁻² nm⁻¹ (from file)
      • R_ast(λ) is the unitless reflectance (from SMASS / averaged spectrum)
      • Φ_AB(m, λ0) is the AB-magnitude photon flux at λ0 in photons s⁻¹ m⁻² nm⁻¹

    Args:
      m_AB (float): Object’s AB magnitude near λ0.
      lambda0_nm (float): Reference wavelength in nanometers (default: 550.0).

    Returns:
      float: K (dimensionless). Multiply the reflected spectrum by K to get correct absolute counts.

    Notes:
      • This sets only the overall normalization; the spectral *shape* stays the same.
      • If you skip this (i.e., K = 1), your SNR maps will have arbitrary absolute values,
        although relative filter comparisons may still look similar.
    """
    Fnu_AB0_W_m2_Hz = 3631.0 * 1e-26                   # 0-mag AB reference
    lam_m = float(lambda0_nm) * 1e-9                    # nm → m
    # photons s^-1 m^-2 nm^-1 implied by m_AB at λ0
    photons_per_nm_target = (10.0 ** (-0.4 * m_AB)) * (Fnu_AB0_W_m2_Hz / (_H * lam_m)) * 1e-9
    base_photons_per_nm = float(f_sun(lambda0_nm) * f_ast(lambda0_nm))
    return photons_per_nm_target / max(base_photons_per_nm, 1e-30)

# Example apparent magnitudes (only used when you plot specific objects)
ID_TO_Vmag: Dict[str, float] = {
    "000004": 6.0,
    "000006": 8.0,
    "000024": 10.0,
}

# -----------------------------------------------------------------------------
# Sky model: constant AB surface brightness (mag / arcsec^2) across λ
# We treat it as ~1/λ in photons-per-nm space via the AB definition.
# Output units: photons s^-1 m^-2 nm^-1 arcsec^-2
# -----------------------------------------------------------------------------
SKY_SB_MAG_AB: float = 21.7            # typical dark sky brightness in V [mag / arcsec^2]
PLATE_SCALE_AS_PER_PIX: float = 0.50   # change one I know what this actual number is 
OMEGA_PIX_AS2: float = PLATE_SCALE_AS_PER_PIX ** 2  # pixel solid angle [arcsec^2 / pix]

def sky_surface_brightness_photons_per_nm(
    m_AB: float, lambda_nm: np.ndarray
) -> np.ndarray:
    """AB mag/arcsec^2 → photons s^-1 m^-2 nm^-1 arcsec^-2 at each wavelength."""
    Fnu_AB0_W_m2_Hz = 3631.0 * 1e-26
    lam_m = np.asarray(lambda_nm, float) * 1e-9
    return (10.0 ** (-0.4 * m_AB)) * (Fnu_AB0_W_m2_Hz / (_H * lam_m)) * 1e-9

def sky_photon_spectrum(lambda_nm: np.ndarray) -> np.ndarray:
    return sky_surface_brightness_photons_per_nm(SKY_SB_MAG_AB, lambda_nm)

# -----------------------------------------------------------------------------
# Per-wavelength (per-nm) integrands -> electrons per nm (or per nm per arcsec^2)
# -----------------------------------------------------------------------------
def asteroid_flux_raw(lambda_nm: np.ndarray | float, K_scale: float = 1.0) -> np.ndarray:
    """Photons s^-1 m^-2 nm^-1 coming *from the asteroid* before atmosphere/optics/QE."""
    return K_scale * f_sun(lambda_nm) * f_ast(lambda_nm)

def _collect_common_factors(instr: Dict[str, float], t_exp_s: float) -> Tuple[float, float, float]:
    """Telescope area [m^2] and exposure time [s]."""
    area_m2 = math.pi * (instr["d_m"] ** 2) / 4.0
    return area_m2, float(t_exp_s)

def signal_integrand_e_per_nm(
    lambda_nm: np.ndarray, instr: Dict[str, float], t_exp_s: float, K_scale: float = 1.0
) -> np.ndarray:
    """
    Electrons per nm produced by the asteroid signal at each wavelength.
    Ṡ_ast × atmosphere × optics × QE × area × time
    """
    S_ast_ph_nm = asteroid_flux_raw(lambda_nm, K_scale)
    Tatm = f_atm(lambda_nm)
    Topt = f_mirror(lambda_nm)
    QE_e_per_ph = f_qe(lambda_nm)
    area_m2, time_s = _collect_common_factors(instr, t_exp_s)
    return time_s * area_m2 * (S_ast_ph_nm * Tatm * Topt * QE_e_per_ph)

def background_integrand_e_per_nm(
    lambda_nm: np.ndarray, instr: Dict[str, float], t_exp_s: float
) -> np.ndarray:
    """
    Electrons per nm *per arcsec^2* from the sky background.
    Pixel solid angle and #pixels are applied later at the band level.
    """
    S_sky_ph_nm_as2 = sky_photon_spectrum(lambda_nm)  # photons s^-1 m^-2 nm^-1 arcsec^-2
    Tatm = f_atm(lambda_nm)
    Topt = f_mirror(lambda_nm)
    QE_e_per_ph = f_qe(lambda_nm)
    area_m2, time_s = _collect_common_factors(instr, t_exp_s)
    return time_s * area_m2 * (S_sky_ph_nm_as2 * Tatm * Topt * QE_e_per_ph)

# -----------------------------------------------------------------------------
# Cumulative integrals in electrons (so we can get fast band integrals by differencing)
# -----------------------------------------------------------------------------
def cumulative_from_integrand(lambda_nm: np.ndarray, y_e_per_nm: np.ndarray) -> np.ndarray:
    """Simple trapezoid-like accumulate: sum(y × Δλ)."""
    dlam_nm = np.gradient(lambda_nm)
    return np.cumsum(y_e_per_nm * dlam_nm)

# Built on demand after the scene brightness (K) is set:
gCumSig_e: np.ndarray | None = None        # electrons (cumulative vs λ)
gCumBkg_e_per_as2: np.ndarray | None = None  # electrons per arcsec^2 (cumulative)

def rebuild_cums(K_scale: float) -> None:
    """Recompute the cumulative electron curves for the current reflectance and magnitude.
    Ai helped with modifying this:
    What it does:
      1) Uses the current asteroid reflectance + telescope/CCD settings to compute
        how many *electrons per nanometer* we’d record from the asteroid (the signal).
      2) Uses the sky surface brightness to compute how many *electrons per nanometer
        per square arcsecond* we’d record from the sky (the background).
      3) Integrates each of those along wavelength to make *cumulative sums*:
         - gCumSig_e[λ]        = total signal electrons from the blue end up to λ
         - gCumBkg_e_per_as2[λ]= total background electrons per arcsec² up to λ
    """
    global gCumSig_e, gCumBkg_e_per_as2
    sig_e_per_nm = signal_integrand_e_per_nm(wavelength_nm, instr, t_exp_s, K_scale)
    bkg_e_per_nm_as2 = background_integrand_e_per_nm(wavelength_nm, instr, t_exp_s)
    gCumSig_e = cumulative_from_integrand(wavelength_nm, sig_e_per_nm)
    gCumBkg_e_per_as2 = cumulative_from_integrand(wavelength_nm, bkg_e_per_nm_as2)

# -----------------------------------------------------------------------------
# Band SNR (electrons domain)
# -----------------------------------------------------------------------------
def _band_edges_to_indices(bmin_nm: float, bmax_nm: float) -> Tuple[int, int]:
    """Turn [λmin, λmax] into index range on wavelength_nm (safely clipped)."""
    i1 = int(np.searchsorted(wavelength_nm, bmin_nm, side="left"))
    i2 = int(np.searchsorted(wavelength_nm, bmax_nm, side="right") - 1)
    i1 = max(0, min(i1, len(wavelength_nm) - 1))
    i2 = max(0, min(i2, len(wavelength_nm) - 1))
    if i2 < i1:
        i1, i2 = i2, i1
    return i1, i2

def snr_for_band(
    bmin_nm: float, bmax_nm: float, instr: Dict[str, float], t_exp_s: float, n_pix_aperture: int
) -> float:
    """
    Compute SNR for a top-hat band [bmin_nm, bmax_nm].
      S_e      = ∫ signal_e_per_nm dλ          (electrons)
      B_sky_e  = (∫ bkg_e_per_nm_per_as2 dλ) × Ω_pix(as^2) × n_pix
      B_dark_e = dark_e_per_pix_s × t × n_pix
      σ_total  = sqrt(S_e + B_sky_e + B_dark_e + n_pix × read_noise_e^2)
      SNR      = S_e / σ_total
    """
    #little catch added to stop issue from earlier forgetting ot call rebuild
    assert gCumSig_e is not None and gCumBkg_e_per_as2 is not None, "Call rebuild_cums(K) first."
    i1, i2 = _band_edges_to_indices(bmin_nm, bmax_nm)

    S_band_e = float(gCumSig_e[i2] - gCumSig_e[i1])
    B_phot_e_per_as2 = float(gCumBkg_e_per_as2[i2] - gCumBkg_e_per_as2[i1])
    B_sky_e = B_phot_e_per_as2 * OMEGA_PIX_AS2 * float(n_pix_aperture)
    B_dark_e = instr["dark_e_per_pix_s"] * float(n_pix_aperture) * float(t_exp_s)

    read_noise_e = float(instr["read_noise_e"])
    sigma_e = math.sqrt(max(S_band_e + B_sky_e + B_dark_e + n_pix_aperture * (read_noise_e ** 2), 0.0))
    return 0.0 if sigma_e <= 0 else S_band_e / sigma_e

# -----------------------------------------------------------------------------
# Swap in a specific SMASS reflectance (overrides f_ast with an interpolator)
# -----------------------------------------------------------------------------
def set_reflectance_from_arrays(wl_src_nm: np.ndarray, R_src: np.ndarray) -> None:
    """Install a reflectance curve R(λ) (on any grid) as the current asteroid reflectance."""
    global f_ast
    wl_src_nm = np.asarray(wl_src_nm, float)
    R_src = np.asarray(R_src, float)
    R_on_grid = np.interp(wavelength_nm, wl_src_nm, R_src) 

    def _interp(x_nm: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(x_nm, float), wavelength_nm, R_on_grid)

    f_ast = _interp

def load_smass_spectrum_from_file(path_txt: str, id_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read one object from the SMASS table:
      columns: id, wavelength_um, reflectance
    Returns (wavelength_nm, reflectance).
    """
    raw = np.loadtxt(
        path_txt,
        dtype={"names": ("id", "w_um", "R"), "formats": ("U16", "f8", "f8")},
    )
    mask = (raw["id"] == id_str)
    wl_nm = raw["w_um"][mask] * 1_000.0  # μm → nm
    R = raw["R"][mask]
    return wl_nm, R

# -----------------------------------------------------------------------------
# Robust type-averaged spectra (per-λ median + 3σ-MAD clipping → mean)
# using MAD because slightly better thand stdev
# -----------------------------------------------------------------------------
def get_ids_for_type(taxon: str) -> List[str]:
    """Try SMASS helpers if available; otherwise return []."""
    try:
        if hasattr(smass, "ids_for_type"):
            return list(smass.ids_for_type(taxon))
        if hasattr(smass, "get_ids_for_type"):
            return list(smass.get_ids_for_type(taxon))
    except Exception:
        pass
    return []

def _interp_and_norm_to_grid(
    wl_src_nm: np.ndarray, R_src: np.ndarray, grid_nm: np.ndarray, norm_nm: float = 550.0
) -> np.ndarray:
    """Interpolate onto 'grid_nm' and normalize so R(norm_nm) = 1 (if possible)."""
    Rg = np.interp(grid_nm, np.asarray(wl_src_nm, float), np.asarray(R_src, float))
    if norm_nm is not None:
        norm = np.interp(norm_nm, grid_nm, Rg)
        if np.isfinite(norm) and norm > 0:
            Rg = Rg / norm
    return Rg

def robust_average_spectra(
    id_list: List[str], smass_path: str, grid_nm: np.ndarray, norm_nm: float = 550.0, sigma: float = 3.0
) -> Dict[str, np.ndarray]:
    """
    Build a robust average reflectance for the given IDs:
      1) Interpolate & normalize each spectrum to the grid (R(550 nm)=1).
      2) Per-wavelength median and MAD.
      3) Keep values within sigma × 1.4826 × MAD of the median; average them.
    Returns dict with keys: avg, median, mad, mask, stack, ids
    """
    curves: List[np.ndarray] = []
    kept_ids: List[str] = []
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

    if not curves:
        raise RuntimeError("No usable spectra for averaging.")

    X = np.vstack(curves) # (N_spectra, N_wave)
    median = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - median), axis=0)
    thresh = sigma * 1.4826 * mad # MAD→σ for normal dist.
    keep = np.abs(X - median) <= thresh

    with np.errstate(invalid="ignore"):
        avg = np.nanmean(np.where(keep, X, np.nan), axis=0)
    avg = np.where(np.isfinite(avg), avg, median)

    return {"avg": avg, "median": median, "mad": mad, "mask": keep, "stack": X, "ids": kept_ids}

def make_type_average(
    taxon: str,
    smass_path: str,
    grid_nm: np.ndarray = wavelength_nm,
    norm_nm: float = 550.0,
    sigma: float = 3.0,
    explicit_ids: List[str] | None = None,
    save_path: str | None = None,
) -> Dict[str, np.ndarray]:
    """Compute a robust average for a taxon (e.g., 'C', 'S', 'V')."""
    ids = list(explicit_ids) if explicit_ids is not None else get_ids_for_type(taxon)
    if not ids:
        raise RuntimeError(
            f"No IDs provided/found for type '{taxon}'. "
            f"Pass explicit_ids=['000004', ...] if SMASS helper can't list them."
        )
    res = robust_average_spectra(ids, smass_path, grid_nm, norm_nm=norm_nm, sigma=sigma)
    if save_path:
        np.savetxt(
            save_path,
            np.column_stack([grid_nm, res["avg"]]),
            delimiter=",",
            header=f"wavelength_nm, reflectance_avg_{taxon} (normalized at {norm_nm} nm)",
            comments="",
        )
    return res

def set_reflectance_to_type_average(
    taxon: str,
    smass_path: str,
    norm_nm: float = 550.0,
    sigma: float = 3.0,
    explicit_ids: List[str] | None = None,
    save_path: str | None = None,
    also_plot: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute the average and install it as the current reflectance (optionally plot the stack)."""
    res = make_type_average(
        taxon, smass_path, grid_nm=wavelength_nm, norm_nm=norm_nm, sigma=sigma,
        explicit_ids=explicit_ids, save_path=save_path
    )
    set_reflectance_from_arrays(wavelength_nm, res["avg"])

    if also_plot:
        plt.figure(figsize=(7, 4))
        for row in res["stack"]:
            plt.plot(wavelength_nm, row, alpha=0.12, lw=1)
        plt.plot(wavelength_nm, res["avg"], lw=2, label=f"{taxon}-avg")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance (norm @ 550 nm)")
        plt.title(f"{taxon}-type: individual spectra (faint) and robust average")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return res

# -----------------------------------------------------------------------------
# SNR plotting helpers
# -----------------------------------------------------------------------------
def plot_snr_for_asteroid(
    id_str: str,
    smass_path: str,
    instr: Dict[str, float],
    t_exp_s: float,
    n_pix_aperture: int,
    min_nm: int = 400,
    max_nm: int = 900,
    step_nm: int = 10,
) -> None:
    """Compute & plot the SNR for one SMASS object ID."""
    wl_nm, R = load_smass_spectrum_from_file(smass_path, id_str)
    set_reflectance_from_arrays(wl_nm, R)

    mV = ID_TO_Vmag.get(id_str, 10.0)
    K = scale_factor_from_AB_mag(mV, lambda0_nm=550.0)
    rebuild_cums(K)

    L1_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L2_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L1, L2 = np.meshgrid(L1_vals, L2_vals)
    mask = L2 > L1

    SNR_vals = np.full_like(L1, np.nan, dtype=float)
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            if mask[i, j]:
                SNR_vals[i, j] = snr_for_band(float(L1[i, j]), float(L2[i, j]), instr, t_exp_s, n_pix_aperture)

    plt.figure(figsize=(7, 6))
    pc = plt.pcolormesh(L1, L2, SNR_vals, shading="auto")
    plt.colorbar(pc, label="SNR")
    plt.xlabel("Band min λ (nm)")
    plt.ylabel("Band max λ (nm)")
    plt.title(f"SNR for asteroid {id_str}")
    plt.tight_layout()
    plt.show()

def plot_snr_current(
    label: str,
    instr: Dict[str, float],
    t_exp_s: float,
    n_pix_aperture: int,
    min_nm: int = 400,
    max_nm: int = 900,
    step_nm: int = 10,
) -> None:
    """Plot SNR using the *currently installed* reflectance (f_ast)."""
    L1_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L2_vals = np.arange(min_nm, max_nm + 1, step_nm)
    L1, L2 = np.meshgrid(L1_vals, L2_vals)
    mask = L2 > L1

    SNR_vals = np.full_like(L1, np.nan, dtype=float)
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            if mask[i, j]:
                SNR_vals[i, j] = snr_for_band(float(L1[i, j]), float(L2[i, j]), instr, t_exp_s, n_pix_aperture)

    plt.figure(figsize=(7, 6))
    pc = plt.pcolormesh(L1, L2, SNR_vals, shading="auto")
    plt.colorbar(pc, label="SNR")
    plt.xlabel("Band min λ (nm)")
    plt.ylabel("Band max λ (nm)")
    plt.title(f"SNR for {label}")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(7, 4))
    plt.plot(wavelength_nm, sky_photon_spectrum(wavelength_nm))
    plt.xlabel("Wavelength nm")
    plt.ylabel("Sky")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------
tax_path = "/Users/karakanetis/Documents/GitHub/optimized-filters/main_code/taxonomy.pds.table.txt"
taxonomy_map = ld.load_taxonomy_table(tax_path)
print("taxonomy file:", tax_path, "| entries:", len(taxonomy_map))
print("type counts (first 10):", Counter(taxonomy_map.values()).most_common(10))

if __name__ == "__main__":
    smass_path = "/Users/karakanetis/Documents/GitHub/optimized-filters/main_code/smass2_all_spfit.txt"
    n_pix_aperture = 300  # ≈ number of pixels in photometric aperture

    # Build ID lists per taxon (include subclasses like 'S0', 'Sv')
    def ids_for_type(prefix: str) -> List[str]:
        P = prefix.upper()
        return [aid for aid, cls in taxonomy_map.items() if str(cls).upper().startswith(P)]

    type_sets = {"C": ids_for_type("C"), "S": ids_for_type("S"), "V": ids_for_type("V")}

    # Plot three objects (one per class)
    print("\nSpecific examples: C:000024, S:000006, V:000004")
    for aid, label in (("000024", "C-type example"), ("000006", "S-type example"), ("000004", "V-type example")):
        plot_snr_for_asteroid(aid, smass_path, instr, t_exp_s, n_pix_aperture)

    # Plot for the type-AVERAGE spectra (once per class)
    print("\nType-average spectra: C/S/V")
    mV_by_type = {"C": 10.0, "S": 8.5, "V": 6.5}  # representative apparent magnitudes
    for taxon, ids in type_sets.items():
        if not ids:
            print(f"[{taxon}] skipped: no IDs found in taxonomy file.")
            continue

        set_reflectance_to_type_average(
            taxon, smass_path, norm_nm=550.0, sigma=3.0, explicit_ids=ids, save_path=None, also_plot=False
        )
        K = scale_factor_from_AB_mag(mV_by_type.get(taxon, 10.0), lambda0_nm=550.0)
        rebuild_cums(K)
        plot_snr_current(f"{taxon}-avg", instr, t_exp_s, n_pix_aperture, 400, 900, 10)
