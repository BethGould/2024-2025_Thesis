# k_M_models_ivp.py
#
# Author: Elizabeth Gould
# Date Last Edit: 03.03.2026

# This contains our models for the shift in frequencies of both types of oscillations. 
# They are calculated as functions of mu, dk, B, R, and dpsi0. The dpsi0 dependence means that 
# they solve the IVP, not the BVP.

# The most important functions contained here are:
# pred_fast_k(dpsi0, mu, dk, B, R, A = 1., k0=kFAu) 
#       -- Our model for k for ee interaction, as calculated by the ODE solver.
# pred_fast_k_true(dpsi0, mu, dk, B, R, A = 1., k0=kFAu)
#       -- Our model for k for ee interaction, after subtracting the error term.
# pred_slow_k(dpsi0, mu, dk, B, R, A = 1., k0=kFAu)
#       -- Our model for M for ee interaction.

# Uncommented are also: 
# pred_fast_t(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001) 
#       This is our raw model for the fast oscillations. It is the same as pred_fast_k, but 
#       returns t/2 instead of k. I tend to prefer this to pred_fast_k, since it was first.
# pred_slow_k_v3(dpsi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_004)
#       This is the same as pred_slow_k. Our current model was the third I have in this
#       document, hence the name. The others have been removed, but I still use this 
#       name due to this reason.
# pred_slow_k_v2(dpsi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_004)
#       This is a less precise model for slow oscillations, which remains for demostration of 
#       our model extraction script.


# ----- LIBRARIES --------

import numpy as np
from eelib import pi, kFAu, phi0inv, pppterm, B_max, R_max

# ------ CONSTANTS --------
# These dictionaries are hard-coded parameters, which we already found numerically.
# They are not used outside of the functions here.

# Model 1 fast t
const_dic_001 ={"mA2": 3.675494115574291e-08, 
                "mDIM": -6.179621225636901e-11, 
                "mDI2": 3.674200444430406e-08, 
                "mDR2": 3.674365054354659e-08, 
                "m2": 8.522223583430349e-06, 
                "m2DI2": 2.0556114758548036e-05, 
                "m2DR2": 2.9358597545134476e-05, 
                "m2DR4": 1.598207117609667e-05, 
                "m2DI4": 1.5910577914323505e-05, 
                "m2DI2R2": 3.182358695140475e-05, 
                "intercept": -5.697364648383741e-16, 
                "DI2": -2.7618261397447087e-16, 
                "DR2": 8.0960114314636e-17, 
                "DI4": 9.19298181818588e-17}

# Model 2 slow t
const_dic_002 = {"mdI": 104537.83229936952
                 }

# Model 3 slow M v2
const_dic_003 ={"mdI": 1116125114862.058
                 }

# Model 4 slow M v3
const_dic_004 = {"mdI": 1114503328463.9722,
                  "mdR2I": 1487350163.9047058}

# ----- DELTA k MODELS -----

# Model 1
# Calculates t/2 for the fast oscillations. t/2 is what I get from my averaging 
# of the spacing between maxima of the |Psi|**2 function.
def pred_fast_t(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001):
    t_pred = 0

    #k_full = k0+dk/R/1e-6/2.0
    k_full = k0+dk/R_max/2.0
    t0 = pi / k_full
    dphik = dphi0/k_full

    # constant terms -- t_0 and intercept (error)
    t_pred = t0 + consts['intercept']

    # terms independent of mu (error)                                
    t_pred += (np.real(dphik)**2 * consts['DR2'] 
                + np.imag(dphik)**2 * consts['DI2'] 
                + np.imag(dphik)**4 * consts['DI4'])

    # mu linear terms
    t_pred += mu*(consts['mDR2'] * np.real(dphik)**2
                + consts['mDI2'] * np.imag(dphik)**2
                + consts['mA2']
                + consts['mDIM'] * np.imag(dphik) * B * R)

    # mu quadradic terms
    t_pred += mu**2*(consts['m2']
                    + consts['m2DI2'] * np.imag(dphik)**2
                    + consts['m2DR2'] * np.real(dphik)**2 
                    + consts['m2DR4'] * np.real(dphik)**4 
                    + consts['m2DI4'] * np.imag(dphik)**4
                    + consts['m2DI2R2'] * np.real(dphik)**2 * np.imag(dphik)**2)
    
    return t_pred
    
# Here I return k instead of t/2, which tends to be more useful.
def pred_fast_k(dpsi0, mu, dk, B, R, A = 1., k0=kFAu):
    return pi / pred_fast_t(dpsi0, mu, dk, B, R, A, k0)

# Rescaling of A is equivalent to a rescaling of mu and dphi0.
# However, the rescaling of dphi0 is not straightforward without knowing Psi(x).

# Our final solution is sensitive to R, but not A, while our model for k is sensitive to A, but not R.
# This is due to the difference between IVPs and BVPs. It also means I can leave out both A and R variations.
# But I should test R variations for stability (to know if it affects error, which it did, but no longer does). 

# Our model contains:
# 0th order:
# -- The expected k without ee interaction.
# 1st order: 
# -- The main delta k term(s) for non-linear ee interaction.
#    Appears to be effectively one term, which is useful.
# 2nd order: 
# -- Second order corrections for the non-linear ee interaction modification.
# -- First order error corrections for numerical solutions without ee interaction.
#    As far as I can tell, the error is identical to the case without ee interaction.
#    Most likely the error from the corrections will be visible only in fourth order terms,
#    while I will need at most third order terms. This is a hypothesis, and not proven.

# This is the derivative of k, splitting real and imaginary Psi derivative as seperate variables.
# It has been removed, as the Jacobian is not used.
'''
def deriv_fast_k(x, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001):
    t_pred_r = 0
    t_pred_i = 0

    dphi0 = x[0] + 1j * x[1]

    #k_full = k0+dk/R/1e-6/2.0
    k_full = k0+dk/R_max/2.0
    dphik = dphi0/k_full

    t_new = pred_fast_t(dphi0, mu, dk, B, R, A , k0, consts)

    # constant terms have zero derivitive

    # terms independent of mu (error)                                
    t_pred_r += (2 * np.real(dphik) * consts['DR2'] / k_full)
    t_pred_i += (2 * np.imag(dphik) * consts['DI2'] / k_full
                + 4 * np.imag(dphik)**3 * consts['DI4'] / k_full)

    # mu linear terms
    t_pred_r += mu*(consts['mDR2'] * np.real(dphik)* 2 / k_full)
    t_pred_i += mu*(consts['mDI2'] * np.imag(dphik)* 2 / k_full
                + consts['mDIM'] * B * R / k_full)

    # mu quadradic terms
    t_pred_r += mu**2*(consts['m2DR2'] * np.real(dphik)*2/k_full 
                    + consts['m2DR4'] * np.real(dphik)**3 * 4 / k_full
                    + consts['m2DI2R2'] * np.real(dphik)*2 * np.imag(dphik)**2/k_full)
    t_pred_i += mu**2*(consts['m2DI2'] * np.imag(dphik)*2/k_full
                    + consts['m2DI4'] * np.imag(dphik)**3*4/k_full
                    + consts['m2DI2R2'] * np.real(dphik)**2 * np.imag(dphik)*2/k_full)
    
    dk_pred_r = - pi * t_pred_r / (t_new**2)
    dk_pred_i = -pi * t_pred_i / (t_new**2)
    
    return [dk_pred_r, dk_pred_i]
'''

# These are our old, unsuccessful models of M, and have thus been removed.
'''
def pred_slow_t(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_002):
    M = B*R*phi0inv
    #k_full = k0+dk/R/R_max/2.0
    k_full = k0+dk/R_max/2.0
    BB = B * R / R_max

    #print(M)
    #print(B, R, M, BB)

    # constant terms
    #M_pred = M 

    # mu linear terms
    #M_pred += mu * consts['mdI'] * np.imag(dphi0) * R * phi0inv / k_full / 4

    #print(mu * consts['mdI'] * np.imag(dphi0) / k_full)

    # Note that the scaling may be wrong due to a factor of 4
    M_pred = 2 * (pi / (M)) / (mu/ BB * B_max * np.imag(dphi0) * consts['mdI'] / k_full + 1)
    
    return M_pred


# derivative for Jacobian; note that re psi ' = x_0 and im psi ' = x_1
def deriv_slow_k(x, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_002):
    M = B*R*phi0inv
    #k_full = k0+dk/R/2.0
    k_full = k0+dk/R_max/2.0
    BB = B * R / R_max

    # Note that the scaling may be wrong due to a factor of 4
    M_pred_i = - 2 * (pi / (2*M)) / (mu/ BB * B_max * x[1] * consts['mdI'] / k_full + 1)**2 * mu/ BB * B_max * consts['mdI'] / k_full
    
    return [0., M_pred_i]

'''
# This model here remains for demonstrating my model tests.
def pred_slow_k_v2(dpsi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_003):
    M = B*R*phi0inv
    k_full = k0+dk/R_max/2.0
    dpsi0_s = dpsi0 / k_full

    # check for real-imag switching
    #M_pred = (B*R*R_max*phi0inv*B_max) + consts['mdI'] * mu * np.real(dphi0) / k_full
    M_pred = M + consts['mdI'] * mu * np.imag(dpsi0_s)
    #   2 * (pi / (M)) / (mu/ BB * B_max * np.imag(dphi0) * consts['mdI'] / k_full + 1)
    
    return M_pred

# The model for M here is just calling our most successful model below.
def pred_slow_k(dpsi0, mu, dk, B, R, A = 1., k0=kFAu):
    return pred_slow_k_v3(dpsi0, mu, dk, B, R, A, k0)

# Our successful model for M
def pred_slow_k_v3(dpsi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_004):
    M = B*R*phi0inv
    k_full = k0+dk/R_max/2.0
    dpsi0_s = dpsi0 / k_full
    
    # check for real - imag switching
    M_pred = M + consts['mdI'] * mu * np.imag(dpsi0_s) + consts['mdR2I'] * mu * np.imag(dpsi0_s) * np.real(dpsi0_s)**2
    
    return M_pred

# The Jacobian is not used, so it has been removed.
'''
def deriv_slow_k_v3(x, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_004):
    #M = B*R*phi0inv
    k_full = k0+dk/R_max/2.0
    dpsi0_sr = x[0] / k_full
    dpsi0_si = x[1] / k_full
    
    #M_pred = M + consts['mdI'] * mu * np.imag(dpsi0_s) + consts['mdR2I'] * mu * np.imag(dpsi0_s) * np.real(dpsi0_s)**2
    M_pred_r = 2 * consts['mdR2I'] * mu * dpsi0_si * dpsi0_sr / k_full
    M_pred_i = consts['mdI'] * mu / k_full + consts['mdR2I'] * mu * dpsi0_sr**2 / k_full
    
    return [M_pred_r, M_pred_i]
'''

# Our model for k (our fast oscillation wavenumber), with the error term removed. 
def pred_fast_k_true(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001):
    t_pred = 0

    if A != 1.:
        mu = mu * np.abs(A)**2

    #k_full = k0+dk/R/1e-6/2.0
    k_full = k0+dk/R_max/2.0
    t0 = pi / k_full
    dphik = dphi0/k_full

    # constant terms -- t_0 and intercept (error)
    t_pred = t0 #+ consts['intercept']

    # terms independent of mu (error)                                
    #t_pred += (np.real(dphik)**2 * consts['DR2'] 
    #            + np.imag(dphik)**2 * consts['DI2'] 
    #            + np.imag(dphik)**4 * consts['DI4'])

    # mu linear terms
    t_pred += mu*(consts['mDR2'] * np.real(dphik)**2
                + consts['mDI2'] * np.imag(dphik)**2
                + consts['mA2']
                + consts['mDIM'] * np.imag(dphik) * B * R)

    # mu quadradic terms
    t_pred += mu**2*(consts['m2']
                    + consts['m2DI2'] * np.imag(dphik)**2
                    + consts['m2DR2'] * np.real(dphik)**2 
                    + consts['m2DR4'] * np.real(dphik)**4 
                    + consts['m2DI4'] * np.imag(dphik)**4
                    + consts['m2DI2R2'] * np.real(dphik)**2 * np.imag(dphik)**2)
    
    return pi / t_pred
    
