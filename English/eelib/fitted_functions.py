# fitted_functions.py

# Author: Elizabeth Gould
# Date Last Edit: 15.03.2026

# This contains functions used by reference for non-linear function regression (scipy.optimize.curve_fit). 
# They are included outside the class structure and in a unified form so that they can be easily switched 
# out. 
# 
# The code which uses these functions to perform the regression is below in the same document.

# ---- Functions

# 1: Trigonometry
# sin_fit(x, amp, k, theta)
# sin_fit_2(x, amp, k, x0)

# 2: Jacobian Elliptical (unused)
# These are the Jacobian elliptic functions as implemented in scipy.special.
# Parameters are:
#   x -- independent variable
#   k -- wave number (multiplies x values), the period is larger than expected for a trigonometric number
#   x0 -- coordinate of the x-axis for the elliptic function
#   m -- elliptic parameter
#   A -- amplitude (multiplies the result of the function)
#   y0 -- coordinate of the y-axis for the elliptic function
# cn_fit(x, k, x0, m, A, y0) -- cosine elliptic function, prefers the axis when compared to cosine
# sn_fit(x, k, x0, m, A, y0) -- sine elliptic function, avoids the axis when compared to sine
# dn_fit(x, k, x0, m, A, y0) -- delta amplitude elliptic function, positive, with more values closer to the axis
# cn2_fit(x, k, x0, m, A, y0) -- square of cn
# sn2_fit(x, k, x0, m, A, y0) -- square of sn
#
# Weierstrass elliptic functions are not implemented in SciPy, but other implementations exist.
# The Weierstrass p function can look like the secant squared curve, so these elliptic functions may be important.

# 3: Rotated Secant / Cosecant Squared (unused and minimally tested)
#    The function handle skew_csc_sq(x, period, amp, tilt, x0) is passed to the regression algorithm.
#    The other functions are just helper functions.
# rotation_tan_sq(x1, theta) 
#   -- Given x1 and the rotation angle theta (radians), find the values x2 and y2 produced by rotating
#      that point on the tan^2 curve around the origin by theta.
# find_min_x_skew_tan(tilt) 
#   -- This finds the minimum x value produced when a standard tan^2 curve is rotated around the origin
#      by the angle tilt (in radians).
# find_skew_tan(x2, a, b, tilt)
#   -- Given our point x2 (the x coordinate of the rotated tan^2 function), find the value of 
#      our function y2 (the value of the rotated tan^2 function).
#      We are rotating the tan**2 curve by tilt (radians), and we know x1 (the x coordinate of the 
#      unrotated tan^2 curve) is between a and b.
# skew_csc_sq(x, period, amp, tilt, x0)
#   -- Find the value of our rotated sec^2 function, with the given period, amplitude (amp), 
#      rotation (tilt) and x coordinate of its minima x0, at point x.

# 4: Sawtooth and Tilted Sine -- not implemented

# 5: Cycloid - not implemented

# 6: Piecewise Functions - not implemented

# ---- Function Fitting
#
# fit_sin(sol) -- Takes a solution object returned by the SciPy ODE solver, as called by our loop class,
#                 takes the real part of the points triggered by maxima of the absolute value of our solution,
#                 fits this data to a sine curve, and returns the solution outputted by the SciPy nonlinear 
#                 regressor. The returned values are an array of parameters: amplitude, wave function, angle offset.

# -- Helper functions for fit_sin
#    Estimates the parameters before attempting regression.
# find_amp(sol) -- Estimates the amplitude based on the maximum absolute value of y in the dataset.
# find_root_dif(y_points) -- Finds the spacing, in number of points, between sign flips.
# find_root_points(y_points) -- Finds the location of the sign flips in the data.
# find_root_start(sol) -- Estimates the starting point of the sine function based on when and how the function switches sign.
# find_period(sol) -- Estimates the period of the sine function based on the distance between sign flips.
# find_fit_params(sol) -- Estimates the parameters fit by fit_sin, so that the regressor starts close to the correct value.

# ----- LIBRARIES -------
import numpy as np
from scipy.special import ellipj
from scipy.optimize import curve_fit
from scipy.optimize import brentq

from eelib.consts import pi

# ------- FUNCTIONS FOR FUNCTION FITTING --------
# This contains functions used by reference for non-linear function regression (scipy.optimize.curve_fit). 

# 1: Basic Trigonometry
#    Fitting function wrappers for trigonometric functions.
# Parameters are amp (amplitude), k (wave number), 
# and theta (angle at x = 0) or x0 (x coordinate for the start of the sine function).

def sin_fit(x, amp, k, theta):
    return amp * np.sin(k * x + theta)

def sin_fit_2(x, amp, k, x0):
    return amp * np.sin(k * (x - x0))

# 2: Jacobian Elliptical (unused)
#    Fitting function wrappers for elliptical functions.
#    Note that x0 for cn must move the function to a maximum, while sn starts at zero, like sine.
# Parameters are k (multiples x, like a wave number), 
#                x0 (x coordinate for the start of the function),
#                m (Jacobian elliptical parameter),
#                A (amplitude of oscillations), 
#                and y0 (coordinate of y axis for oscillations).

def cn_fit(x, k, x0, m, A, y0):
    cn,sn,dn,ph = ellipj(k * (x - x0), m)
    return A * cn - y0

def sn_fit(x, k, x0, m, A, y0):
    cn,sn,dn,ph = ellipj(k * (x - x0), m)
    return A * sn - y0

def dn_fit(x, k, x0, m, A, y0):
    cn,sn,dn,ph = ellipj(k * (x - x0), m)
    return A * dn - y0

def cn2_fit(x, k, x0, m, A, y0):
    cn,sn,dn,ph = ellipj(k * (x - x0), m)
    return A * (cn)**2 - y0

def sn2_fit(x, k, x0, m, A, y0):
    cn,sn,dn,ph = ellipj(k * (x - x0), m)
    return A * (sn)**2 - y0

# 3: Rotated Secant / Cosecant Squared (unused and minimally tested)
#    Fitting function for a rotated secant squared curve.
#    Note that the period is known, so it will need to be set with a lambda function.
#    > fun = lambda x, amp, tilt, x0: skew_csc_sq(x, period, amp, tilt, x0)
# Parameters are period (of oscillations / repetition), 
#                amplitude (minimum of secant squared which multiplies all secant squared values),
#                tilt (angle of rotation for our curve),
#                and x0 (x-axis, minimum of our secant squared curve).

# Given x1 and the rotation angle theta (radians), find the values x2 and y2 produced by rotating
# that point on the tan^2 curve around the origin by theta.
def rotation_tan_sq(x1, theta):
    x2 = np.cos(theta)* x1 + np.sin(theta)*np.power(np.tan(x1),2)
    y2 = - np.sin(theta)* x1 + np.cos(theta)*np.power(np.tan(x1),2)
    return x2, y2

# This finds the minimum x value produced when a standard tan^2 curve is rotated around the origin
# by the angle tilt (in radians).
def find_min_x_skew_tan(tilt):

    # Roots of the derivative are extremal points, here the minima. 
    # This is the derivative d/dx1 of:
    #        x2 = np.cos(theta)* x1 + np.sin(theta)*np.power(np.tan(x1),2)
    fun = lambda x: np.cos(tilt) + 2 * np.sin(tilt)* np.tan(x)*np.power(np.cos(x),-2)

    # Avoid the singularity at the period boundaries.
    eps = 0.000001 

    # And return the results of finding this root.
    ret = brentq(fun, -pi / 2 + eps, 0)
    return ret

# Period of csc^2 is pi. 

# Given our point x2 (the x coordinate of the rotated tan^2 function), find the value of 
# our function y2 (the value of the rotated tan^2 function).
# We are rotating the tan**2 curve by tilt (radians), and we know x1 (the x coordinate of the 
# unrotated tan^2 curve) is between a and b.
def find_skew_tan(x2, a, b, tilt):
    # x2 > x1
    #a = min_x1 (or 0)
    #b = x2 (or 0)

    # Define a function for which to find the roots.
    # x2 = np.cos(theta)* x1 + np.sin(theta)*np.power(np.tan(x1),2)
    fun = lambda x: x2 - np.cos(tilt)* x - np.sin(tilt)*np.power(np.tan(x),2)

    # Find the value of x1 which gives our value of x2 when the tan^2 function is rotated by tilt.
    # Our upper bound is either 0 or our desired x2.
    # Our lower bound is either 0 or the lowest possible value of x1 for our rotated curve.
    x1 = brentq(fun, a, b)

    # x2b here should now return our inputted x2, so we only need y2
    x2b, y2 = rotation_tan_sq(x1, tilt)

    return y2

# Fitting function for a rotated secant squared curve.
# Parameters are period (of oscillations / repetition), 
#                amplitude (minimum of secant squared which multiplies all secant squared values),
#                tilt (angle of rotation for our curve),
#                and x0 (x-axis, minimum of our secant squared curve).
def skew_csc_sq(x, period, amp, tilt, x0):
    x_new = x - x0 # Shift x by x0 (the x0 coordinate of a minimum).
    x_rescale = x_new / period # Divide by the period.
    period_count = x_rescale // 1.0 # The number of periods from the center is unused.
    x_per_rem = x_rescale % 1.0 # The fractional part gives the functional value.
    x_rem = x_per_rem * pi # Coordinate from the minimum to the left of x
    x_neg_rem = (x_per_rem-1) * pi # Coordinate from the minimum to the right of x

    # The minimum x point of the right period selects which curve to analyze, in order to find the
    # value. If the curve on the right has our value, use that one, otherwise use the left curve.
    # The root of the derivative dx2/dx1 returns the x coordinate (x1) of our minimum.
    # We then need to rotate this function to find the x and y coordinates of our rotated function.
    min_x1 = find_min_x_skew_tan(tilt)
    min_x2, min_y2 = rotation_tan_sq(min_x1, tilt)

    # Now select which curve to use. 
    if x_neg_rem == min_x2:
        y = min_y2 # If we are at our boundary, we already know the y value.
    elif x_neg_rem > min_x2:
        # If our value exists on the right curve, find the value on that curve.
        # We know that x1 is negative and greater than our minimum.
        y = find_skew_tan(x_neg_rem, min_x1, 0, tilt)
    else:
        # If the value doesn't exist on the right curve, find the value on the left curve.
        # We know that x1 is positive and less than x2.
        y = find_skew_tan(x_rem, 0, x_rem, tilt)

    # Rescale our function by our amplitude, and add the amplitude to switch from tan^2 to sec^2,
    # since sec^2 = tan^2 + 1.
    y = y * amp + amp
    return y

# 4: Tilted Sine / Sawtooth

# 5: Cycloid

# 6: Piecewise Functions


# ----- FUNCTION FITTING CODE -----

# Now to fit the sine function. All of our parameters must be set close to the expected value 
# for this to work well, which is why we estimate our parameters first.
def fit_sin(sol):
    # Estimate our parameters amp, M, theta,
    # because we need to start close to the expected value to get a correct fit.
    amp, TT, rt_st = find_fit_params(sol)

    # Extract our x and psi values from our solution.
    xdata = sol['t']
    ydata = np.real(sol['y'][0])

    # Error case here. So that I just mark the case as unsolvable if there is not enough data
    # for my simple algorithm. For the slow t grid, this data is not required, since we have other
    # data.
    if rt_st is None:
        return None
    
    MM = 2 * pi / TT # Switch from period to wave number.
    theta = - MM * xdata[rt_st] # Rescale our x_0 to be an angle shift.

    # Prevent our function fitter from deviating too far from our expected values.
    amp_tol = 0.1
    M_tol = 0.1
    th_tol = 0.1 * pi

    # Minimum, maximum, and starting values for our function fitter, in the correct form.
    min_arr = np.array([(1.0 - amp_tol) * amp, MM - M_tol * MM , theta - th_tol])
    max_arr = np.array([(1.0 + 2 * amp_tol) * amp, MM + M_tol * MM, theta + th_tol])
    start_arr = np.array([amp, MM, theta])

    # And call SciPy.
    popt, pcov = curve_fit(sin_fit, xdata, ydata, p0=start_arr, bounds=(min_arr, max_arr))
    return popt


# ---------- PERIOD AND AMPLITUDE CALCULATIONS -------------

# These are helper functions for the fit_sine function. They assume nice behavior. So far, 
# they have worked well, but I can think of cases when they won't.

# Amplitude Estimate
# Given a solution, estimate the amplitude as the furthest distance in the solution from 0.
# The actual amplitude should be slightly more than this, but can be found with function fitting.
# This just gives the initial guess.
def find_amp(sol):
    arr = np.real(sol['y_events'][1][:, 0])
    return np.max(np.abs(arr))

# This returns the spacings between roots in number of points.
# To find the period, one needs to multiply by the t spacing between points.
def find_root_dif(y_points):
    root_list = find_root_points(y_points)
    y_diff = np.array(root_list[1:])- np.array(root_list[:-1])
    return np.average(y_diff)

# This is used to estimate the values for which the curve crosses the zero axis.
# It is used for both x_0 estimation and period estimation.
# y_points = y1_list_0 = np.real(sol['y'][0])
# Given the y points, find the indexes of the roots.
def find_root_points(y_points):
    y_mult = np.array(y_points[:-1])*np.array(y_points[1:])
    zero_index = np.nonzero(y_mult == 0)
    root_index = np.nonzero(y_mult < 0)
    full_index = list(root_index[0]) + list(zero_index[0])
    full_index.sort()
    return full_index

# x_0 Estimate
# Here I want to find the first two x values where the function changes sign
# and return the one where the sign goes from negative to positive,
# since the sine function starts at 0 and increases from there.
def find_root_start(sol):
    y1_list = np.real(sol['y'][0])
    root_ls = find_root_points(y1_list)
    if len(root_ls) < 2:
        rt_start = None
    elif y1_list[root_ls[0]+1] > 0: 
        rt_start = root_ls[0]
    else: 
        rt_start = root_ls[1]
    return rt_start

# Period Estimate
# The function goes through 0 twice in a period. For the sine function it starts t zero, 
# goes through zero once, and then ends at zero. I can use this to estimate the period as
# twice the spacing between roots.
def find_period(sol):
    t_list = sol['t']
    y1_list = np.real(sol['y'][0])
    root_list = find_root_points(y1_list)
    t_root_list = []
    for i in root_list: 
        t_root_list.append(t_list[i])
    if len(t_root_list) == 0:
        return None
    t_diff = np.array(t_root_list[1:])- np.array(t_root_list[:-1])
    t_dif_ave = np.average(t_diff)
    return t_dif_ave * 2

# Now for a wrapper function to call and return the result of all three estimates.
def find_fit_params(sol):
    amp = find_amp(sol)
    rt_start = find_root_start(sol)
    MM = find_period(sol)
    return amp, MM, rt_start       

