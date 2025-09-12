#make code mathc papers eqns better for organization
#added eqn2 and eqn 7
#units funct with sci.
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

def spectrum_sun(wavelength_nm):
    lam = np.asarray(wavelength_nm, float)
    return 1e6 / np.clip(lam, 1.0, None)
def reflectance_asteroid(wavelength_nm):
    lam = np.asarray(wavelength_nm, float)
    return 0.95 + 0.0002 * (lam - 550.0)
def transmission_atm(wavelength_nm):
    return np.ones_like(wavelength_nm, dtype=float)
def transmission_optics(wavelength_nm):
    return 0.9 * np.ones_like(wavelength_nm, dtype=float) 
def ccd_qe(wavelength_nm):
    lam = np.asarray(wavelength_nm, float)
    return np.exp(-((lam - 650.0)/200.0)**2) 
def transmission_filter_placeholder(wavelength_nm, wavelength1_nm, wavelength2_nm):
    lam = np.asarray(wavelength_nm, float)
    return ((lam >= wavelength1_nm) & (lam <= wavelength2_nm)).astype(float)
def sky_photon_spectrum(wavelength_nm):
    return 1e2 * np.ones_like(wavelength_nm, dtype=float) 

def asteroid_flux_raw(wavelength_nm): # eqn 2- ish
    # Placeholder for S_ast(λ) ≈ S_sun(λ) * R_ast(λ). Replace with full V-band later
    Ssun = spectrum_sun(wavelength_nm)
    Rast = reflectance_asteroid(wavelength_nm)
    return Ssun * Rast 

def detected_signal(wavelength1_nm, wavelength2_nm, t_exp_s, instr, wavelength_nm): 
    # Ssun = spectrum_sun(wavelength_nm)
    # Rast = reflectance_asteroid (wavelength_nm)
    # deleting to move them into their own section asteroid_flux_raw
    S_ast = asteroid_flux_raw(wavelength_nm)
    Tatm = transmission_atm(wavelength_nm)
    Topt = transmission_optics(wavelength_nm)
    QE = ccd_qe(wavelength_nm)
    Tf = transmission_filter_placeholder (wavelength_nm, wavelength1_nm, wavelength2_nm)

    inside_integral= S_ast*Tatm*Topt*Tf*QE
    area_m2= np.pi*(instr["d"]**2)/4.0
    return t_exp_s * instr["gain"]*area_m2*np.trapz(inside_integral, wavelength_nm)

def delta_S(S, B): #eqn 7
    return float(np.sqrt(max(S + B, 0.0))) #again max, ai reccomendation from before to avoid 0s

def detected_background(wavelength1_nm, wavelength2_nm, t_exp_s, instr, wavelength_nm):
    Ssky = sky_photon_spectrum(wavelength_nm)
    Tatm = transmission_atm(wavelength_nm)
    Topt = transmission_optics(wavelength_nm)
    QE = ccd_qe(wavelength_nm)
    Tf = transmission_filter_placeholder (wavelength_nm, wavelength1_nm, wavelength2_nm)

    inside_integral= Ssky*Tatm*Topt*Tf*QE
    area_m2= np.pi*(instr["d"]**2)/4.0
    return t_exp_s * instr["gain"] * area_m2 *np.trapz(inside_integral, wavelength_nm)

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

#More tests
lam = np.arange(440.0, 921.0, 1.0)
#test instrument
instr = make_instrument(
    d_m=1.0, # telescope diameter
    gain_e_per_ph=1.0,# CCD gain
    s_pix_m=5e-6, # pixel size
    read_noise_e=5.0 # read noise
)

# Two filters and exposure
band1 = (460, 470)#blue-ish band
band2 = (730, 740) #red/NIR-ish
t_exp = 60.0 #1 min, 60 sec

# Find signals detected and background in each band
S1 = detected_signal(*band1, t_exp, instr, lam) # signal band1
B1 = detected_background(*band1, t_exp, instr, lam)# background band1
S2 = detected_signal(*band2, t_exp, instr, lam)# signal band2
B2 = detected_background(*band2, t_exp, instr, lam)# background band2

# uncertainty eqn 7
dS1 = delta_S(S1, B1) 
dS2 = delta_S(S2, B2)

#color and uncertainty eqn 8
c, dc = color_mag(S1, dS1, S2, dS2)

# results
print("Band 1 (", band1[0], "to", band1[1], "nm )")
print("Signal S1 (electrons):", S1) # how many electrons we collected
print("Background B1 (electrons):", B1) #how many electrons come from the sky
print("Uncertainty ΔS1 (electrons):", dS1)

print("\nBand 2 (", band2[0], "to", band2[1], "nm )")
print("Signal S2 (electrons):", S2)
print("Background B2 (electrons):", B2)
print("Uncertainty ΔS2 (electrons):", dS2)

# comparing these two bands we get the color, which shows the asteroid surface peoperties
# dif asteroids have dif colors thats why we want to print this
print("\nColor")
print("Color (m2 - m1) [mag]:", round(c, 3))
print("Uncertainty in color [mag]:", round(dc, 3))

# prints how I expect it. yay.