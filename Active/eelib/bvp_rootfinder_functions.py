# bvp_rootfinder_functions.py

# Author: Elizabeth Gould
# Date Last Edit: 09.03.2026

# This contains my function wrapper for solving the BVP with a root finder, the linear models 
# to be used with that function wrapper, and a sample script to perform the root finding.

# k_calc_0(dpsi0, mu, dk, B, R, A = 1., k0=kFAu) -- returns k = k0+dk/R_max/2.0, our linear model
# M_calc_0(dpsi0, mu, dk, B, R, A = 1., k0=kFAu) -- returns M = B*R*phi0inv, our linear model

# function_wrapper_bvp(x, mu, dk, B, R, A = 1., k0=kFAu, k_calc_f = pred_fast_k, M_calc_f = pred_slow_k_v3)
#   -- Takes loop parameters mu, dk, B, R, A, and k0 and calculates the difference between the given
#      Psi initial derivative dpsi0 = x[0] + 1.j*x[1] and the required derivative for matching the BVP.
#      Finding the roots of this function solves the BVP.
#
#   -- To pass this function to the root finder, use the code:
#      >  yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f, M_calc_f)
#      and pass yy as the function handle for which to find the roots.
#
#   -- Loop parameters B and R are in real units, not the percent of their maximum value. 
#
#   -- This function requires a model for k and M as a function of our parameters.
#   -- k_calc_f(dpsi0, mu, dk, B, R, A, k0) and M_calc_f(dpsi0, mu, dk, B, R, A, k0)
#   -- Here, dpsi0 is in the form of a single complex number, and B and R are in real units, as stored in
#      the loop class, NOT as a percent of their maximum value, as passed to the loop class.

# find_root_both(dk, R, B, mu, method = 'broyden2', tol = 1e-20, ratio = 1.0,
#                  check_solution = False, model_k=pred_fast_k_true, model_M = pred_slow_k_v3)
#   -- This is a function which will find one solution of our nonlinear BVP as well as the solution to
#      the linear BVP. If check_solution is false, it will only return a nonlinear BVP solution as an array
#      of two real numbers with the real and imaginary components of the initial derivative of psi. 
#      If check_solution is true, it will return 4 values:
#          - A complex solution for the initial derivative of psi for the linear case.
#          - The value of our function for which to fit the roots for this (linear) derivative.
#          - A complex solution for the initial derivative of psi for the nonlinear case.
#          - The value of our function for which to fit the roots for this (nonlinear) derivative.
#   -- Our inputs are:
#       - dk, R, B, mu, in real units, as retrieved from our loop.
#       - method - A root finder algorithm from the list of available SciPy root finders.
#       - tol - Our tolerance for this root finder algorithm.
#       - ratio - Our initial guess is in proportion to our linear solution as dphi0 / ratio.
#       - model_k and model_M are our models of k and M for our function for which to find the roots.

# I have Jacobian code which is not used right now as it has not been checked and debugged.
# I had problems with matching before, so I needed to switch the model, and have not updated my Jacobian.
# All the Jacobian code has been hidden from the computer as comments.

import numpy as np
from scipy import optimize
from eelib.consts import pi, kFAu, phi0inv, R_max
from eelib.k_M_models_ivp import pred_fast_k, pred_slow_k_v3, pred_fast_k_true
#from eelib.k_M_models_ivp import deriv_fast_k, deriv_slow_k_v3

# The first functions give the models for k and M for the linear case.
# They take unused parameters since their parameters need to match those of other models.
def k_calc_0(dpsi0, mu, dk, B, R, A = 1., k0=kFAu):
    return k0+dk/R_max/2.0

def M_calc_0(dpsi0, mu, dk, B, R, A = 1., k0=kFAu):
    return B*R*phi0inv

#def dk_calc_0(x, mu, dk, B, R, A = 1., k0=kFAu):
#    return [0.0, 0.0]

#def dM_calc_0(x, mu, dk, B, R, A = 1., k0=kFAu):
#    return [0.0, 0.0]

# Here is the function for which we are finding the roots.
# Note that this is the same as the linear case, but with shifted wave numbers.
# This is an approximation of the solution, not the actual expected form, for
# which the fast oscillations are not completely sinusoidal.
# However, plots show that this is an acceptable approximation.
def function_wrapper_bvp(x, mu, dk, B, R, A = 1., k0=kFAu, k_calc_f = pred_fast_k, M_calc_f = pred_slow_k_v3):
    # decipher arguments
    dpsi = x[0] + 1j * x[1]

    k_calc = k_calc_f(dpsi, mu, dk, B, R, A, k0)  #k0+dk/R_max/2.0 #pi / pred_fast_t(dpsi, mu, dk, B, R)
    M_calc = M_calc_f(dpsi, mu, dk, B, R, A, k0)  #B*R*phi0inv #pred_slow_t(dpsi, mu, dk, B, R)

    # note that x[0] is the real part and x[1] is the imaginary part

    F_re = (np.cos( M_calc *2 * pi * R) - np.cos(2*pi*k_calc*R)) / np.sin(2*pi*k_calc*R)
    F_im =  np.sin( M_calc *2 * pi * R) / np.sin(2*pi*k_calc*R)

    # this gives the correct values for psi_0 without ee-interaction
    root_real = x[0] - k_calc * F_re
    root_im = x[1] - M_calc + k_calc * F_im

    return [root_real, root_im]

'''
# The Jacobian here is not used as it has not been debugged
def jacobian_wrapper_bvp(x, mu, dk, B, R, A = 1., 
                         k0=kFAu, k_calc_f = pred_fast_k, M_calc_f = pred_slow_k_v3,
                         dk_calc_f = deriv_fast_k, dM_calc_f = deriv_slow_k_v3):
    #[[df0 / dx0,
    #df0/dx1],
    #[df1/dx0,
    #df1/dx1]])

    dpsi = x[0] + 1j * x[1]

    k_calc = k_calc_f(dpsi, mu, dk, B, R, A, k0)  #k0+dk/R_max/2.0 #pi / pred_fast_t(dpsi, mu, dk, B, R)
    M_calc = M_calc_f(dpsi, mu, dk, B, R, A, k0)  #B*R*phi0inv #pred_slow_t(dpsi, mu, dk, B, R)

    dkdx = dk_calc_f(x, mu, dk, B, R, A, k0)  #0#deriv_fast_k(x, mu, dk, B, R)
    dmdx = dM_calc_f(x, mu, dk, B, R, A, k0)  #0#deriv_slow_k(x, mu, dk, B, R)

    F_re = (np.cos( M_calc *2 * pi * R) - np.cos(2*pi*k_calc*R)) / np.sin(2*pi*k_calc*R)

    j00 = 1. + dkdx[0] * np.sin(2*pi*R*M_calc)/np.sin(2*pi*R*k_calc) 
    j00 = j00 + k_calc * dmdx[0] * np.cos(2*pi*R*M_calc)/np.sin(2*pi*R*k_calc) 
    j00 = j00 - k_calc * dkdx[0] * np.sin(2*pi*R*M_calc)*np.cos(2*pi*R*k_calc)/(np.sin(2*pi*R*k_calc)**2)

    j01 = 0.
    j01 = dkdx[1] * np.sin(2*pi*R*M_calc)/np.sin(2*pi*R*k_calc) 
    j01 = j01 + k_calc * dmdx[1] * np.cos(2*pi*R*M_calc)/np.sin(2*pi*R*k_calc) 
    j01 = j01 - k_calc * dkdx[1] * np.sin(2*pi*R*M_calc)*np.cos(2*pi*R*k_calc)/(np.sin(2*pi*R*k_calc)**2)

    j10 = 0.
    j10 = -dmdx[0] + dkdx[0] * F_re
    j10 = j10 + k_calc * (dkdx[0] * np.sin(2*pi*R*k_calc) - dmdx[0] * np.sin(2*pi*R*M_calc)) / np.sin(2*pi*R*k_calc) 
    j10 = j10 + k_calc * dkdx[0] * (np.cos(2*pi*R*M_calc)- np.cos(2*pi*R*k_calc)) *np.cos(2*pi*R*k_calc)/(np.sin(2*pi*R*k_calc)**2)

    j11 = 1. -dmdx[1] + dkdx[1] * F_re
    j11 = j11 + k_calc * (dkdx[1] * np.sin(2*pi*R*k_calc) - dmdx[1] * np.sin(2*pi*R*M_calc)) / np.sin(2*pi*R*k_calc) 
    j11 = j11 + k_calc * dkdx[1] * (np.cos(2*pi*R*M_calc)- np.cos(2*pi*R*k_calc)) *np.cos(2*pi*R*k_calc)/(np.sin(2*pi*R*k_calc)**2)

    return np.array([[j00,j01],[j10,j11]])
'''

#   -- This is a function which will find one solution of our nonlinear BVP as well as the solution to
#      the linear BVP. If check_solution is false, it will only return a nonlinear BVP solution as an array
#      of two real numbers with the real and imaginary components of the initial derivative of psi. 
#      If check_solution is true, it will return 4 values:
#          - A complex solution for the initial derivative of psi for the linear case.
#          - The value of our function for which to fit the roots for this (linear) derivative.
#          - A complex solution for the initial derivative of psi for the nonlinear case.
#          - The value of our function for which to fit the roots for this (nonlinear) derivative.
#   -- Our inputs are:
#       - dk, R, B, mu, in real units, as retrieved from our loop.
#       - method - A root finder algorithm from the list of available SciPy root finders.
#       - tol - Our tolerance for this root finder algorithm.
#       - ratio - Our initial guess is in proportion to our linear solution as dphi0 / ratio.
#       - model_k and model_M are our models of k and M for our function for which to find the roots.

def find_root_both(dk, R, B, mu, method = 'broyden2', tol = 1e-20, ratio = 1.0,
                   check_solution = False, model_k=pred_fast_k_true, model_M = pred_slow_k_v3):

    # Find the root of the linear equation. There is only one, so algorithm and initial guess are irrelevant.
    yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=k_calc_0, M_calc_f=M_calc_0)
    #jj = lambda x: jac(x, mu, dk, B, R, k_calc_f=k_calc_0, M_calc_f=M_calc_0, dk_calc_f=dk_calc_0, dM_calc_f=dM_calc_0)

    sol = optimize.root(yy, [-1e12, 1e12], method='hybr')
    xs = sol.x

    out_0 = yy(xs)

    # Now find the root for the nonlinear solution.
    yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=model_k, M_calc_f=model_M)
    #jj = lambda x: jac(x, mu, dk, B, R)

    sol = optimize.root(yy, xs/ratio, method=method, tol=tol)
    dphi0c = sol.x

    deriv_psi = dphi0c[0] + 1j * dphi0c[1]
    deriv_psi_o = xs[0]+1j*xs[1]

    # And return our desired solutions.
    if check_solution:
        return deriv_psi_o, out_0, deriv_psi, yy(dphi0c)
    else:
        return dphi0c
    
