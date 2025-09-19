#plotting distribution
import numpy as np
import matplotlib.pyplot as plt

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
    wavelength = np.asarray(wavelength_nm, float)
    return 1e6 / np.clip(wavelength, 1.0, None)
def reflectance_asteroid(wavelength_nm):
    wavelength = np.asarray(wavelength_nm, float)
    return 0.95 + 0.0002 * (wavelength - 550.0)
def transmission_atm(wavelength_nm):
    return np.ones_like(wavelength_nm, dtype=float)
def transmission_optics(wavelength_nm):
    return 0.9 * np.ones_like(wavelength_nm, dtype=float) 
def ccd_qe(wavelength_nm):
    wavelength = np.asarray(wavelength_nm, float)
    return np.exp(-((wavelength - 650.0)/200.0)**2) 
def transmission_filter_placeholder(wavelength_nm, wavelength1_nm, wavelength2_nm):
    wavelength = np.asarray(wavelength_nm, float)
    return ((wavelength >= wavelength1_nm) & (wavelength <= wavelength2_nm)).astype(float)
def sky_photon_spectrum(wavelength_nm):
    return 1e2 * np.ones_like(wavelength_nm, dtype=float) 


# integrated signal
# instead of integrating in detected_signal do that here
#edit wavelangth 1, 2 to just signle wavelength
def cumulative_signal(signal): # return signal from min wavelength to given wavelength
    return np.cumsum(signal)

def detected_signal(wavelength_nm, t_exp_s, instr, wavelength1_nm, wavelength2_nm):  # review the wavelength 1 and 2 why did I add this notes
    Ssun = spectrum_sun(wavelength_nm)
    Rast = reflectance_asteroid (wavelength_nm)
    Tatm = transmission_atm(wavelength_nm)
    Topt = transmission_optics(wavelength_nm)
    QE = ccd_qe(wavelength_nm)
    Tf = transmission_filter_placeholder (wavelength_nm, wavelength1_nm, wavelength2_nm)

    area_m2= np.pi*(instr["d"]**2)/4.0
    return t_exp_s * instr["gain"]*area_m2* Ssun*Rast*Tatm*Topt*Tf*QE

def delta_S(S, B):
    return float(np.sqrt(max(S + B, 0.0)))

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

wavelength = np.arange(440.0, 921.0, 1.0)

instr = make_instrument(
    d_m=1.0,
    gain_e_per_ph=1.0,
    s_pix_m=5e-6,
    read_noise_e=5.0 
)

band1 = (450, 650) #increased to make 
band2 = (730, 740)
t_exp = 60.0 #change t_exp to t_exp_s

# S1 = detected_signal(wavelength, t_exp, instr, *band1)
# B1 = detected_background(*band1, t_exp, instr, wavelength)
# S2 = detected_signal(wavelength, t_exp, instr, *band2)
# B2 = detected_background(*band2, t_exp, instr, wavelength)

# dS1 = delta_S(S1, B1) 
# dS2 = delta_S(S2, B2)

# c, dc = color_mag(S1, dS1, S2, dS2)

# print("Band 1 (", band1[0], "to", band1[1], "nm )")
# print("Signal S1 (electrons):", S1)
# print("Background B1 (electrons):", B1)
# print("Uncertainty ΔS1 (electrons):", dS1)
# print("\nBand 2 (", band2[0], "to", band2[1], "nm )")
# print("Signal S2 (electrons):", S2)
# print("Background B2 (electrons):", B2)
# print("Uncertainty ΔS2 (electrons):", dS2)
# print("\nColor")
# print("Color (m2 - m1) [mag]:", c)
# print("Uncertainty in color [mag]:", dc)

#PLOTS

# x axis
wavelength = np.arange(400,901, 1)
# transmission variables
Tatm= transmission_atm(wavelength)
Topt = transmission_optics(wavelength)
Tf =transmission_filter_placeholder(wavelength, 460, 470) #example from paper for fun
QE =ccd_qe(wavelength)

# Plot, a few grouped together
plt.figure(figsize=(8,5))
plt.plot(wavelength,Tatm,label="atm transmission")
plt.plot(wavelength,Topt,label="optics transmission")
plt.plot(wavelength,Tf,label="Filter of 460–470 nm")
plt.plot(wavelength,QE,label="CCD QE")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission")
plt.title("Fake numbers plots")
plt.legend()
plt.grid(True)
plt.show()

signal = detected_signal(wavelength, t_exp, instr, band1[0], band1[1])

plt.figure(figsize=(8,5))
plt.plot(wavelength, signal, label= "detected signal") # review
plt.xlabel("Wavelength (nm)")
plt.ylabel("Signal")
plt.title("New Detected Signal Plot")
plt.legend()
plt.grid(True)
plt.show()

cumulative_signal = cumulative_signal(signal)
plt.figure(figsize=(8,5))
plt.plot(wavelength, cumulative_signal, label= "cumulative signal") 
plt.xlabel("Wavelength (nm)")
plt.ylabel("Cumulative signal")
plt.title("New Cumulative Signal Plot")
plt.legend()
plt.grid(True)
plt.show()

#more/extra

#solar spectrum
Ssun = spectrum_sun(wavelength)
plt.figure(figsize=(8,5))
plt.plot(wavelength, Ssun)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Solar spectrum - arbitrary units")
plt.title("Solar spectrum S_sun(λ)")
plt.grid(True)
plt.show()

#asteroid reflectance
Rast = reflectance_asteroid(wavelength)
plt.figure(figsize=(8,5))
plt.plot(wavelength, Rast)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (unitless)")
plt.title("Asteroid reflectance Rast(λ)")
plt.grid(True)
plt.show()


