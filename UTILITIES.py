#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file rjedicke_utilities.py
@brief Utilities for printing and file operations.
@created on Tue Jan 19 10:15:20 2021
@author rjedicke
"""

# NOTE:  THIS FILE WAS COMMENTED AND DOXYGENATED BY GEMINI

import os
import math
import subprocess
import inspect
import numpy as np
from scipy.constants import pi
from numpy.lib import recfunctions
from sigfig import round as sigfig_round
from sty import fg  # Text coloring for terminal output

# Constants
TWOPI = 2 * pi
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
gM_earth_kg = 5.972e24
gM_moon_kg = 7.348e22
gM_EM_kg = gM_earth_kg + gM_moon_kg

# ----------------------------------------------------------------------------------------------------------------------
def print_names(data_object):
    """
    @brief Print all column names from a NumPy structured array.
    @param data_object The NumPy structured array.
    """
    for name in data_object.dtype.names:
        print(name)

# ----------------------------------------------------------------------------------------------------------------------
def total_orbital_energy_per_kg(v_mps, r_m, M_kg=gM_EM_kg):
    """
    @brief Compute total orbital energy per unit mass (J/kg).
    @param v_mps velocity in m/s.
    @param r_m distance from center of mass in meters.
    @param M_kg mass of the central body in kg (default is Earth+Moon).
    @return Total orbital energy per unit mass.
    """
    T = 0.5 * v_mps**2               # Kinetic energy per unit mass
    U = -G * M_kg / r_m              # Gravitational potential energy per unit mass
    return T + U                     # Total energy per unit mass

# ----------------------------------------------------------------------------------------------------------------------
def all_identical(nparray, significant_figures=8):
    """
    @brief Check if all values in a NumPy array are equal to the same value after rounding.
    @param nparray The input array.
    @param significant_figures The number of decimals to round for comparison.
    @return True if all values are approximately identical, False otherwise.
    """
    rounded = np.round(nparray, decimals=significant_figures)
    return np.all(rounded == rounded[0])

# ----------------------------------------------------------------------------------------------------------------------
def rebin_1d(X, n):
    """
    @brief Rebin a 1D array by averaging over bins of size n.
    @param X The input 1D array.
    @param n The bin size.
    @return The rebinned array with averages.
    @exception ValueError if the array length is not a multiple of n.
    """
    if len(X) % n != 0:
        raise ValueError("Array length must be a multiple of n to rebin")

    return X.reshape(-1, n).mean(axis=1)

# ----------------------------------------------------------------------------------------------------------------------
def clip(value, min_value, max_value):
    """
    @brief Clamp a value between min_value and max_value.
    @param value The value to be clamped.
    @param min_value The minimum allowed value.
    @param max_value The maximum allowed value.
    @return The clamped value.
    """
    return max(min_value, min(value, max_value))

# ----------------------------------------------------------------------------------------------------------------------
def file_empty(file_path):
    """
    @brief Check if a file is empty.
    @param file_path The path to the file.
    @return True if the file is empty, False otherwise.
    """
    return os.stat(file_path).st_size == 0

# ----------------------------------------------------------------------------------------------------------------------
def file_exists(file_path):
    """
    @brief Check if a file exists.
    @param file_path The path to the file.
    @return True if the file exists, False otherwise.
    """
    return os.path.exists(file_path)

# ----------------------------------------------------------------------------------------------------------------------
def load_table(filename, delimiter=',', bNames=True, encoding=None):
    """
    @brief Load tabular data from a file using NumPy's genfromtxt.
    @param filename The path to the file.
    @param delimiter The field separator (default is ',').
    @param bNames If true, the first row is treated as column names.
    @param encoding The file encoding (e.g., 'utf-8').
    @return A structured NumPy array with the data.
    """
    return np.genfromtxt(
        filename,
        delimiter=delimiter,
        names=bNames,
        dtype=None,
        encoding=encoding
    )

# ------------------------------------------------------------------------------------
def add_table_row(table, new_row):
    """
    @brief Append a new row to an existing NumPy structured array (table).
    @param table The original structured array.
    @param new_row The new row of data to be added.
    @return A new structured array with the added row.
    """
    print('new_row = ', new_row)
    new_table = recfunctions.append_fields(table, names=table.dtype.names, data=[new_row], usemask=False)
    return new_table

# ------------------------------------------------------------------------------------
def value_pm_uncertainty(x, dx):
    """
    @brief Format a value with its uncertainty into a string for LaTeX.
    @param x The value.
    @param dx The uncertainty of the value.
    @return A formatted string representing the value with its uncertainty, e.g., "1.234 \pm 0.056".
    """
    formatted_string = sigfig_round(x, dx)
    szValue, szUncertainty = formatted_string.split(' ± ')
    return szValue + ' \pm ' + szUncertainty

# ------------------------------------------------------------------------------------
def value_and_uncertainty_latex(x, dx):
    """
    @brief Format a value with its uncertainty for a LaTeX newcommand.
    @param x The value.
    @param dx The uncertainty of the value.
    @note This function is marked as 'NOT WORKING' in the original code.
    The implementation is a placeholder and may produce incorrect results.
    """
    sig_figs = len(str(dx).lstrip('0').replace('.', '').rstrip('0'))
   #print(sig_figs)
    value = sigfig_round(x,  sig_figs)
    unc   = sigfig_round(dx, sig_figs)
    return(f"({value} ± {unc}) × 10^{int(np.log10(x))}")

# ------------------------------------------------------------------------------------
def chi2_per_dof(zMeasured, zExpected, dz, df=1):
    """
    @brief Calculate the reduced chi-squared statistic (chi2 per degree of freedom).
    @param zMeasured The measured data values.
    @param zExpected The expected data values.
    @param dz The uncertainty of the measured data.
    @param df The number of degrees of freedom.
    @return The reduced chi-squared statistic.
    """
    return np.sum(((zMeasured - zExpected) / dz)**2) / df

# ------------------------------------------------------------------------------------
def integrate_linear_polynomial(m, b, x1, x2):
    """
    @brief Calculate the definite integral of a first-degree polynomial y = mx + b.
    @param m slope of the linear polynomial.
    @param b y-intercept of the linear polynomial.
    @param x1 lower limit of integration.
    @param x2 upper limit of integration.
    @return The definite integral of the linear polynomial between x1 and x2.
    """
    # Antiderivative of mx + b is (m/2)x^2 + bx
    antiderivative_at_x2 = (m/2) * x2**2 + b * x2
    antiderivative_at_x1 = (m/2) * x1**2 + b * x1

    # Definite integral is the difference of antiderivative values
    definite_integral = antiderivative_at_x2 - antiderivative_at_x1

    return definite_integral

# ------------------------------------------------------------------------------------
def getStringRepresentationOfThisVariable(var):
    """
    @brief Get the string representation of a variable's name.
    @param var The variable.
    @return The name of the variable as a string.
    @note This function is a workaround and may not be reliable in all contexts.
    It relies on inspecting the caller's local variables.
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return str([k for k, v in callers_local_vars if v is var][0])

# ------------------------------------------------------------------------------------
def printWarning(szWarningMessage):
    """
    @brief Print a warning message in a distinct, colored format.
    @param szWarningMessage The warning message to print.
    """
    print('\n')
    print_in_color('***************************************', 9)
    print_in_color(szWarningMessage, 9)
    print_in_color('***************************************', 9)
    print('\n')

# ------------------------------------------------------------------------------------
def print_in_color(szText, color8bit=13):
    """
    @brief Prints text in a specified 8-bit color using the `sty` library.
    @param szText The text to print.
    @param color8bit The 8-bit color code (default is 13).
    """
    print(fg(color8bit) + szText + fg.rs)

# ------------------------------------------------------------------------------------
def print_value(szValue, value):
    """
    @brief Print a key-value pair with a simple format.
    @param szValue The name or description of the value.
    @param value The value to print.
    """
    print(szValue, value)

# ------------------------------------------------------------------------------------
def print_value_with_veto(bPrint, szText, value):
    """
    @brief Conditionally print a key-value pair based on a boolean flag.
    @param bPrint If true, the value is printed.
    @param szText The name or description of the value.
    @param value The value to print.
    """
    if bPrint:
        print(szText, value)

# ------------------------------------------------------------------------------------
def print_text_with_veto(bPrint, szText):
    """
    @brief Conditionally print a string based on a boolean flag.
    @param bPrint If true, the text is printed.
    @param szText The text to print.
    """
    if bPrint:
        print(szText)

# ------------------------------------------------------------------------------------
def print_in_color_test():
    """
    @brief Test function to demonstrate different colors from the `sty` library.
    @note The original loop is commented out as it does not work.
    """
    print(fg(0) + 'COLOR 0' + fg.rs)
    print(fg(1) + 'COLOR 1' + fg.rs)
    print(fg(2) + 'COLOR 2' + fg.rs)
    print(fg(3) + 'COLOR 3' + fg.rs)
    print(fg(4) + 'COLOR 4' + fg.rs)
    print(fg(5) + 'COLOR 5' + fg.rs)
    print(fg(6) + 'COLOR 6' + fg.rs)
    print(fg(7) + 'COLOR 7' + fg.rs)
    print(fg(8) + 'COLOR 8' + fg.rs)
    print(fg(9) + 'COLOR 9' + fg.rs)

# ------------------------------------------------------------------------------------
def randomExponential(n=1, a=1, x=(0, 1)):
    """
    @brief Generate a NumPy array of random values exponentially distributed in a given range.
    @param n The number of random values to generate.
    @param a The exponential parameter.
    @param x A tuple (min, max) specifying the range.
    @return An array of n random values.
    """
    f = np.random.uniform(0.0, 1.0, n)
    return np.log(np.exp(a * x[1]) * f + np.exp(a * x[0]) * (1 - f)) / a

# ------------------------------------------------------------------------------------
def randomPower(n=1, a=10, b=0.5, x=(0, 1)):
    """
    @brief Generate a NumPy array of n random values distributed according to the power-law a^(bx).
    @param n The number of random values to generate.
    @param a The base of the power-law.
    @param b The exponent of the power-law.
    @param x A tuple (min, max) specifying the range.
    @return An array of n random values.
    """
    r = np.random.uniform(0.0, 1.0, n)
    c0 = a**(b * x[0])
    c1 = a**(b * x[1])
    return np.emath.logn(a, c1 * r + c0 * (1 - r)) / b

# ------------------------------------------------------------------------------------
def deltav_Tsiolkovsky(ISP_s=1, m_propellant_kg=2000, m_dry_kg=1000):
    """
    @brief Calculate the change in velocity (delta-v) using the Tsiolkovsky rocket equation.
    @param ISP_s Specific impulse in seconds.
    @param m_propellant_kg Mass of the propellant in kg.
    @param m_dry_kg Dry mass of the rocket in kg.
    @return The change in velocity in m/s.
    """
    g0_mps2 = 9.80665
    deltav_mps = ISP_s * g0_mps2 * np.log(1 + m_propellant_kg / m_dry_kg)
    return deltav_mps

# ------------------------------------------------------------------------------------
def lawOfCosines(l1, l2, angle12_deg):
    """
    @brief Solve the law of cosines for the length of a side of a triangle.
    @param l1 The length of the first side.
    @param l2 The length of the second side.
    @param angle12_deg The angle in degrees between sides l1 and l2.
    @return The length of the side opposite to the given angle.
    """
    return np.sqrt(l1**2 + l2**2 - 2 * l1 * l2 * np.cos(np.radians(angle12_deg)))

# ------------------------------------------------------------------------------------
def normalize(value, vmin, vmax):
    """
    @brief Normalize any number to an arbitrary range [min, max) by assuming the range wraps around.
    @param value The value to normalize.
    @param vmin The minimum of the range.
    @param vmax The maximum of the range.
    @return The normalized value within the specified range.
    """
    width = vmax - vmin
    offsetValue = value - vmin
    return (offsetValue - (math.floor(offsetValue / width) * width)) + vmin

# ------------------------------------------------------------------------------------
def truncN(val, N):
    """
    @brief Round and truncate a value to N decimal places.
    @param val The value to truncate.
    @param N The number of decimal places.
    @return The truncated value.
    """
    return np.rint(val * 10**N) / 10**N

# ------------------------------------------------------------------------------------
def errorAdd(x, dx, y, dy):
    """
    @brief Return the sum of two values and the error of their sum.
    @param x The first value.
    @param dx The error on the first value.
    @param y The second value.
    @param dy The error on the second value.
    @return A tuple containing the sum and the combined error.
    """
    add = x + y
    if isinstance(x, np.ndarray):
        err = np.sqrt(dx**2 + dy**2)
    else:
        err = math.sqrt(dx**2 + dy**2)
    return add, err

# ------------------------------------------------------------------------------------
def errorSub(x, dx, y, dy):
    """
    @brief Return the difference of two values and the error of their difference.
    @param x The first value.
    @param dx The error on the first value.
    @param y The second value.
    @param dy The error on the second value.
    @return A tuple containing the difference and the combined error.
    """
    sub = x - y
    err = np.sqrt(dx**2 + dy**2)
    return sub, err

# ------------------------------------------------------------------------------------
def errorMult(x, dx, y, dy):
    """
    @brief Return the product of two values and the error of their product.
    @param x The first value.
    @param dx The error on the first value.
    @param y The second value.
    @param dy The error on the second value.
    @return A tuple containing the product and the combined error.
    """
    product = x * y
    err = np.sqrt((dx / x)**2 + (dy / y)**2) * product
    return product, err

# ------------------------------------------------------------------------------------
def errorDiv(x, dx, y, dy):
    """
    @brief Return the ratio of two values and the error of their ratio.
    @param x The numerator.
    @param dx The error on the numerator.
    @param y The denominator.
    @param dy The error on the denominator.
    @return A tuple containing the ratio and the combined error.
    """
    ratio = x / y
    err = np.sqrt((dx / x)**2 + (dy / y)**2) * ratio
    return ratio, err

# ------------------------------------------------------------------------------------
def efficiency(k, n, CI=0.68):
    """
    @brief Calculate the efficiency and confidence interval using Marc Paterno's method.
    This function calls an external C++ executable `calceff2`.
    @param k A vector of successes.
    @param n A vector of trials.
    @param CI The confidence interval (default is 0.68).
    @return A tuple containing the efficiency, min efficiency, and max efficiency.
    """
    np.savetxt('junk.dat', np.column_stack([k, n]), fmt='%d %d')
    try:
        subprocess.check_output(
            '/Users/rjedicke/Dropbox/src/C++/calceff2/calceff2 junk.dat ' + str(CI) + ' > junk.results',
            shell=True
        )
    except subprocess.CalledProcessError:
        pass  # Ignore the error and continue
    return np.loadtxt('junk.results', unpack=True)

# ------------------------------------------------------------------------------------
def fraction(k, n, CI=0.68):
    """
    @brief Calculate the fraction and its positive and negative uncertainties.
    @param k A vector of successes.
    @param n A vector of trials.
    @param CI The confidence interval (default is 0.68).
    @return A tuple containing the fraction, positive uncertainty, and negative uncertainty.
    """
    f, fmin, fmax = efficiency(k, n, CI=CI)
    return f, fmax - f, f - fmin

# ------------------------------------------------------------------------------------
def inRange(value, vrange):
    """
    @brief Check if a value is within a specified range [min, max].
    @param value The value to check.
    @param vrange A tuple (min, max) defining the range.
    @return True if the value is within the range, False otherwise.
    """
    return (value >= vrange[0] and value <= vrange[1])

# ------------------------------------------------------------------------------------
def inRangeMinusPi2PlusPi(angle_rad):
    """
    @brief Normalize an angle in radians to the range [-pi, +pi).
    @param angle_rad The angle in radians.
    @return The normalized angle.
    """
    return angle_rad - (np.floor((angle_rad + pi) / TWOPI)) * TWOPI

# ------------------------------------------------------------------------------------
def InRange0to360(angle_deg):
    """
    @brief Normalize an angle in degrees to the range [0, 360).
    @param angle_deg The angle in degrees.
    @return The normalized angle.
    """
    return angle_deg % 360

# ------------------------------------------------------------------------------------
def weighted_mean( x, sigma ):
    """
    Compute the weighted mean of values with uncertainties.

    Parameters
    ----------
    x : array_like
        Values.
    sigma : array_like
        1-sigma uncertainties on the values (must be > 0).

    Returns
    -------
    mean : float
        Weighted mean of the values.
    mean_err : float
        Uncertainty on the weighted mean.
    """
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("All uncertainties must be > 0.")

    weights = 1.0 / sigma**2
    mean = np.average(x, weights=weights)
    mean_err = np.sqrt(1.0 / weights.sum())
    
    return mean, mean_err

#------------------------------------------------------------------------------------------------------------------
# The meat of this function was created by ChatGPT

def piecewise_linear_function( x_data, y_data ):
    
    """
    Create a piecewise-linear function interpolating given data points.

    Parameters
    ----------
    x_data : array_like
        Sorted array of x-values (independent variable).
    y_data : array_like
        Corresponding y-values.

    Returns
    -------
    f : callable
        Function f(x) that linearly interpolates between points.
        Valid only for x between x_data[0] and x_data[-1].
    """

    def f(x):
        # x = np.asarray(x, dtype=float)
        # use numpy's efficient linear interpolation
        return np.interp( x, x_data, y_data )

    return f

#------------------------------------------------------------------------------------------------------------------
def wrap_angle_0_to_180( angle_deg ):
    """
    Convert an angle in degrees from [0, 360) into (-180, 180].

    Parameters
    ----------
    angle_deg : float
        Input angle in degrees (0 <= angle < 360).

    Returns
    -------
    float
        Equivalent angle in the range (-180, 180].
    """
    wrapped = ((angle_deg + 180) % 360) - 180
    return wrapped



# ------------------------------------------------------------------------------------
# def normalize_angle_180(angle):
#     """
#     @brief Normalize an angle in degrees to the range [-180, 180).
#     @param angle The angle in degrees.
#     @return The normalized angle.
#     """
#     return angle - (np.floor((angle + 180) / 360) * 360)

# # ------------------------------------------------------------------------------------
# def normalize_angle_pi(angle):
#     """
#     @brief Normalize an angle in radians to the range [-pi, pi).
#     @param angle The angle in radians.
#     @return The normalized angle.
#     """
#     return angle - (np.floor((angle + pi) / TWOPI) * TWOPI)

# # ------------------------------------------------------------------------------------
# def normalize_angle_2pi(angle):
#     """
#     @brief Normalize an angle in radians to the range [0, 2pi).
#     @param angle The angle in radians.
#     @return The normalized angle.
#     """
#     return normalize_angle_pi(angle) + pi

# # ------------------------------------------------------------------------------------
# def normalize_angle_180_closed(angle):
#     """
#     @brief Normalize an angle in degrees to the range (-180, 180].
#     @param angle The angle in degrees.
#     @return The normalized angle.
#     """
#     return angle - np.ceil(angle / 360.0 - 0.5) * 360.0