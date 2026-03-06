# This contains functions used by reference for non-linear function regression (scipy.optimize.curve_fit). 
# They are included outside the class structure and in a unified form
# so that they can be easily switched out.

#--LIBRARIES--------
import numpy as np
from scipy.special import ellipj
from scipy.optimize import curve_fit
from scipy.optimize import brentq

from eelib.consts import pi

# ------- FUNCTIONS FOR FUNCTION FITTING --------
# This contains functions used by reference for non-linear function regression (scipy.optimize.curve_fit). 

# Fitting functions for trigonometric functions

# 1: Basic Trigonometry

def sin_fit(x, amp, k, theta):
    return amp * np.sin(k * x + theta)

def sin_fit_2(x, amp, k, x0):
    return amp * np.sin(k * (x - x0))

# 2: Jacobian Elliptical

# Fitting functions for elliptical functions
# note that x0 for cn must move the function to a maximum
# sn starts at zero, like sine
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


# 3: Skewed Secant / Cosecant Squared

def rotation_tan_sq(x1, theta):
    x2 = np.cos(theta)* x1 + np.sin(theta)*np.power(np.tan(x1),2)
    y2 = - np.sin(theta)* x1 + np.cos(theta)*np.power(np.tan(x1),2)
    return x2, y2

def find_min_x_skew_tan(tilt):

    fun = lambda x: np.cos(tilt) + 2 * np.sin(tilt)* np.tan(x)*np.power(np.cos(x),-2)

    eps = 0.000001

    ret = brentq(fun, -pi / 2 + eps, 0)
    return ret

# Period of csc^2 is pi. 

# Take x2, want to find x1. 
# Our desired x2 is the upper bound for x1
# Our lower bound is either 0 or min_x1
def find_skew_tan(x2, a, b, tilt):
    # x2 > x1
    #a = min_x1
    #b = x2

    # x2 = np.cos(theta)* x1 + np.sin(theta)*np.power(np.tan(x1),2)
    fun = lambda x: x2 - np.cos(tilt)* x - np.sin(tilt)*np.power(np.tan(x),2)

    #print(a, fun(a), b, fun(b), x2)

    x1 = brentq(fun, a, b)

    x2b, y2 = rotation_tan_sq(x1, tilt)

    return y2

def skew_csc_sq(x, period, amp, tilt, x0):
    x_new = x - x0
    x_rescale = x_new / period 
    period_count = x_rescale // 1.0
    x_per_rem = x_rescale % 1.0
    x_rem = x_per_rem * pi 
    x_neg_rem = (x_per_rem-1) * pi

    min_x1 = find_min_x_skew_tan(tilt)
    min_x2, min_y2 = rotation_tan_sq(min_x1, tilt)
    #print(min_x1, min_x2, x_neg_rem)
    if x_neg_rem == min_x2:
        y = min_y2
    elif x_neg_rem > min_x2:
        #print('a')
        y = find_skew_tan(x_neg_rem, min_x1, 0, tilt)
    else:
        #print('b')
        y = find_skew_tan(x_rem, 0, x_rem, tilt)

    y = y * amp + amp
    return y

# Note that the period is known, so I will need to fix it with a lambda function

#def skew_csc_sq_st(x, amp, tilt, x0):
#    return skew_csc_sq(x, pi, amp, tilt, x0)

# 4: Skewed Sine / Sawtooth

# 5: Cycloid

# 6: Piecewise Functions


# ----- FUNCTION FITTING CODE -----

# ---------- PERIOD AND AMPLITUDE CALCULATIONS -------------

# -- Amplitude Estimate --
# Given a solution, estimate the amplitude as the furthest distance in the solution from 0.
# The actual amplitude should be slightly more than this, but can be found with function fitting.
# This just gives the initial guess.
def find_amp(sol):
    arr = np.real(sol['y_events'][1][:, 0])
    return np.max(np.abs(arr))

# This is used to estimate the values for which the curve crosses the zero axis.
# It is used for both x_0 estimation and period estimation.
# y_points = y1_list_0 = np.real(sol['y'][0])
# Given the y points, find the indicies of the roots.
def find_root_points(y_points):
    y_mult = np.array(y_points[:-1])*np.array(y_points[1:])
    zero_index = np.nonzero(y_mult == 0)
    root_index = np.nonzero(y_mult < 0)
    full_index = list(root_index[0]) + list(zero_index[0])
    full_index.sort()
    return full_index

# -- x_0 Estimate --
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

# -- Period Estimate --
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

# Now to fit the sine function. All of our parameters must be set close to the expected value 
# for this to work well, which is why we estimate our parameters first.
def fit_sin(sol):
    # Estimate our parameters amp, M, theta),
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
    
    MM = 2 * pi / TT # Switch from period to wavenumber.
    theta = - MM * xdata[rt_st] # Rescale our x_0 to be an angle shift.

    # Prevent our function fitter from deviating too far from our expected values.
    amp_tol = 0.1
    M_tol = 0.1
    th_tol = 0.1 * pi

    # Minimum, maximum, and starting values for our function fitter, in the correct form.
    min_arr = np.array([(1.0 - amp_tol) * amp, MM - M_tol * MM , theta - th_tol])
    max_arr = np.array([(1.0 + 2 * amp_tol) * amp, MM + M_tol * MM, theta + th_tol])
    start_arr = np.array([amp, MM, theta])

    # And call scipy.
    popt, pcov = curve_fit(sin_fit, xdata, ydata, p0=start_arr, bounds=(min_arr, max_arr))
    return popt

# These functions assume nice behavior. So far, it works well, but I can think of cases when it won't.
        
