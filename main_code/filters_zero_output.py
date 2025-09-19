#returns 0's
import numpy as np
# CoP. reccomends turning these into classes instead with __init__ to speed the code up
def make_instrument(d_m, gain_e_per_ph, s_pix_m, read_noise_e):
    # creates instrument dictionary that combines telescope & camera
    return{
        "d": d_m,
        "gain": gain_e_per_ph,
        "s_pix": s_pix_m,
        "read_noise": read_noise_e,
    }
def make_bands(V_min_nm, V_max_nm, V_width_nm):
    #creates the bands dictionary for V bands
    return {
        "V_min": V_min_nm,
        "V_max": V_max_nm,
        "V_width": V_width_nm,
    }

#place holders for now, details on variables in "NOTES.txt"
# returns are temporary scafolding so I can run the code

def spectrum_sun(lam_nm):
    # using np.zeros_like because they provide counts, using ones would give fake counts
    # this sets the counts to zero
    return np.zeros_like(lam_nm, dtype=float)
def reflectance_asteroid(lam_nm):
    #using ones here because setting to 0 would "kill the signal"
    # so this keeps the code running
    return np.ones_like(lam_nm, dtype=float)
def transmission_atm(lam_nm):
    return np.ones_like(lam_nm, dtype=float)
def transmission_optics(lam_nm):
    return np.ones_like(lam_nm, dtype=float)
def ccd_qe(lam_nm):
    return np.ones_like(lam_nm, dtype=float)
def transmission_filter_placeholder(lam_nm, lam1, lam2):
    # placeholder until it can be converted to transmission_filter_real
    # for the sake of the skeleton, using tophat filter where if the wavelength
    # is between lam1 and lam2 the transmission = 1 or 100% light gets through,
    # if its less than lam 1 or greater than lam2 the transmission = 0.
    # so only wavelengths inside this temp filter cout
    # later should loop through lam1 and lam2 to scan different values to find the best
    # pair to filter asteroid classes
    lam = np.asarray(lam_nm, float)
    return ((lam >= lam1) & (lam <= lam2)).astype(float)
def sky_photon_spectrum(lam_nm):
    return np.zeros_like(lam_nm, dtype=float)


#leaving out camera induced background and dark current for now for simplicity

def detected_signal(lam1, lam2, t_exp_s, instr, lam_nm): 
    # havent made S_ast (V, l) yet so placeholder is Ssun * Rast
    Ssun = spectrum_sun(lam_nm) #photons/s/m^2/nm
    Rast = reflectance_asteroid (lam_nm)
    Tatm = transmission_atm(lam_nm)
    Topt = transmission_optics(lam_nm)
    QE = ccd_qe(lam_nm) #e-/photon
    Tf = transmission_filter_placeholder (lam_nm, lam1, lam2) #0 or 1

    #eqn 4
    inside_integral= Ssun*Rast*Tatm*Topt*Tf*QE
    area_m2= np.pi*(instr["d"]**2)/4.0 #instr will be created/added @ bottom fore testing later
    return t_exp_s * instr["gain"]*area_m2*np.trapz(inside_integral, lam_nm)

def detected_background(lam1, lam2, t_exp_s, instr, lam_nm):
    #background electrpms ignoring camera and dark terms for now
    Ssky = sky_photon_spectrum(lam_nm)
    Tatm = transmission_atm(lam_nm)
    Topt = transmission_optics(lam_nm)
    QE = ccd_qe(lam_nm) #e-/photon
    Tf = transmission_filter_placeholder (lam_nm, lam1, lam2) #0 or 1

    inside_integral= Ssky*Tatm*Topt*Tf*QE #eqn 5
    area_m2= np.pi*(instr["d"]**2)/4.0
    return t_exp_s * instr["gain"] * area_m2 *np.trapz(inside_integral, lam_nm)

def color_mag (S1, dS1,S2, dS2):
    #S1 and S2 are signal electrons in each band and d values are uncertanties
    #using eps= very small number prevents program from crashing if signal is 0
    # this is a safety net that Ai reccomended
    eps = 1e-30
    c = 2.5 * np.log10(max(S1, eps)/max(S2, eps)) #ai also added the max as a safeguard instead of S1/S2, says they shouldnt be lower than eps
    var= 0.0
    if S1>0: # if either of these is 0, divigding by it would have issues, so only adding if its safe to do so
        var+= (dS1/S1)**2
    if S2 >0:
        var += (dS2/S2)**2
    dc=(2.5/np.log(10.0))*np.sqrt(var)# eqn 8
    return c, dc

# Does my structure work!! Hopefully...
lam=np.arange(440.0, 921.0, 1.0) #start stop and step, using variables from paper, pg 9
instr = make_instrument(1.0,1.0, 5e-6,5.0) # test saying telescope diameter is 1 meter, gain is 1 elecrtron per photon, pixel is 5 according to internet typical CCD meter size, 5.0 for read noise because Ai said so
S1 = detected_signal(460,470,60, instr, lam) #460, 470 example from paper pg 9, 60 seconds exposure
B1 = detected_background(460, 470, 60, instr, lam)
# this should find the signal electrons in a 60 second exposure through a 10 nm wide filter at around 465 nm
print("S1 =", S1, "B1 =", B1)

# Printed S1 = 0.0 which was expected