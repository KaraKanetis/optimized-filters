#returns # with toy numbers
#deleting all notes here, only purpose here is to test with ballpark numbers, closing code that doesnt need changing
import numpy as np
def make_instrument(d_m, gain_e_per_ph, s_pix_m, read_noise_e):
    return{
        "d": d_m,
        "gain": gain_e_per_ph,
        "s_pix": s_pix_m,
        "read_noise": read_noise_e,
    }
def make_bands(V_min_nm, V_max_nm, V_width_nm):
    return {
        "V_min": V_min_nm,
        "V_max": V_max_nm,
        "V_width": V_width_nm,
    }

# to save time asking ai to come up with toy numbers - I will explain where each number came from/ reasoning behind
def spectrum_sun(lam_nm):
    lam = np.asarray(lam_nm, float)
    #ai reccomended this limit again as another safeguard against having a zero and crashing 
    return 1e6 / np.clip(lam, 1.0, None) #1e6 is arbitrary number just so its not zero
def reflectance_asteroid(lam_nm):
    lam = np.asarray(lam_nm, float)
    return 0.95 + 0.0002 * (lam - 550.0) #close to 100% reflectance, just so the numbers are easy to look at, not realistic at all since it should be between 10-50%
    #550 is roughly the center of the V-band
    # Will be replaced with propper data later, need to review and understand why it wants to use this as a test more later
def transmission_atm(lam_nm):
    return np.ones_like(lam_nm, dtype=float)
def transmission_optics(lam_nm):
    return 0.9 * np.ones_like(lam_nm, dtype=float) # telescope optics transmit ~80–95% of light so okay guess
def ccd_qe(lam_nm):
    lam = np.asarray(lam_nm, float)
    return np.exp(-((lam - 650.0)/200.0)**2) # figure this one out more later but, 650= about where ccd is most sensitive
    #Gaussian bump peaking at 650 nm, width ~200 nm. This just mimics the rough bell shape
def transmission_filter_placeholder(lam_nm, lam1, lam2):
    lam = np.asarray(lam_nm, float)
    return ((lam >= lam1) & (lam <= lam2)).astype(float)
def sky_photon_spectrum(lam_nm):
    return 1e2 * np.ones_like(lam_nm, dtype=float) #just so background isnt zero


def detected_signal(lam1, lam2, t_exp_s, instr, lam_nm): 
    Ssun = spectrum_sun(lam_nm)
    Rast = reflectance_asteroid (lam_nm)
    Tatm = transmission_atm(lam_nm)
    Topt = transmission_optics(lam_nm)
    QE = ccd_qe(lam_nm)
    Tf = transmission_filter_placeholder (lam_nm, lam1, lam2)

    inside_integral= Ssun*Rast*Tatm*Topt*Tf*QE
    area_m2= np.pi*(instr["d"]**2)/4.0
    return t_exp_s * instr["gain"]*area_m2*np.trapz(inside_integral, lam_nm)

def detected_background(lam1, lam2, t_exp_s, instr, lam_nm):
    Ssky = sky_photon_spectrum(lam_nm)
    Tatm = transmission_atm(lam_nm)
    Topt = transmission_optics(lam_nm)
    QE = ccd_qe(lam_nm)
    Tf = transmission_filter_placeholder (lam_nm, lam1, lam2)

    inside_integral= Ssky*Tatm*Topt*Tf*QE
    area_m2= np.pi*(instr["d"]**2)/4.0
    return t_exp_s * instr["gain"] * area_m2 *np.trapz(inside_integral, lam_nm)

def color_mag (S1, dS1,S2, dS2):
    eps = 1e-30
    c = 2.5 * np.log10(max(S1, eps)/max(S2, eps))
    var= 0.0
    if S1>0:
        var+= (dS1/S1)**2
    if S2 >0:
        var += (dS2/S2)**2
    dc=(2.5/np.log(10.0))*np.sqrt(var)
    return c, dc

lam = np.arange(440.0, 921.0, 1.0)
instr = make_instrument(1.0, 1.0, 5e-6, 5.0)
S1 = detected_signal(460, 470, 60, instr, lam)
B1 = detected_background(460, 470, 60, instr, lam)
print("S1 =", S1, "B1 =", B1)


#I expect to see a positive number... not realistic count but hopefully no errors or negative numbers
# printed out what I expected, positive and finite but unrealistic