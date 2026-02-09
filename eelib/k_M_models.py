
import numpy as np
from eelib import pi, kFAu, phi0inv, pppterm, B_max

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
# delta k models

# Model 1
def pred_fast_t(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001):
    t_pred = 0

    k_full = k0+dk/R/1e-6/2.0
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
    

# rescaling of A is equivalent to a rescaling of mu and dphi0
# however, the rescaling of dphi0 is not straightforward without knowing Psi(x)

# Our final solution is sensitive to R, but not A, while our model for k is sensitive to A, but not R
# This is due to the difference between IVPs and BVPs. It also means I can leave out both A and R variations
# but I should test R variations for stability (to know if it affects error)

# our model contains 
# 0th order:
# -- the expected k without ee interaction
# 1st order: 
# -- the main delta k term(s) for non-linear ee interaction
#    appears to be effectively one term, which is useful
# 2nd order: 
# -- second order corrections for the non-linear ee interaction modification
# -- first order error corrections for numerical solutions without ee interaction
#    As far as I can tell, the error is identical to the case without ee interaction
#    most likely the error from the corrections will be visible only in fourth order terms
#    while I will need at most third order terms. This is a hypothesis, and not proven.
# Orders are most likely at least close to powers of M / k, which is around 10^-2. (need to check this)



# delta M models

def pred_slow_t(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_002):
    M = B*R*phi0inv
    k_full = k0+dk/R/1e-6/2.0

    #print(M)

    # constant terms
    #M_pred = M 

    # mu linear terms
    #M_pred += mu * consts['mdI'] * np.imag(dphi0) * R * phi0inv / k_full / 4

    #print(mu * consts['mdI'] * np.imag(dphi0) / k_full)

    # Note that the scaling may be wrong due to a factor of 4
    M_pred = 4 * (pi / (2*B*R*phi0inv)) / (mu/ B * B_max * np.imag(dphi0) * consts['mdI'] / k_full + 1)
    
    return M_pred

# fot[imu, ib, idr, idi]
# (pi / (2*B_g[ib]*eelib.R_max*eelib.phi0inv*eelib.B_max))/(mu_g[imu]+1)

#  2 pi / T = k
# (mu_g[imu] * S * di / k / b +1) * B_g[ib]*eelib.R_max*eelib.phi0inv
#* B_g[ib] / np.imag(dgrid[idr, idi]) /mu_g[imu]/slope_fin* (gridl_1.l_calc.k)


#(np.power(fot[imu, ib, idr, idi]/(pi / (2*B_g[ib]*eelib.R_max*eelib.phi0inv*eelib.B_max)), -1)-1)
# psi = e^iq e^iTx f(x)
# psi0 = 1, f0 = 1, q unknown
# psip0 = iT psi0 + psi0(fp0/f0)
# psipp0 = - T^2 psi0 + 2 (iT psi0) (fp0 / f0) + psi0 (fpp0/f0)

'''
def dM_model_001(f0p, f0pp, mu, B, R, k_0):
    knl = pppterm * mu
    T_pred = np.imag(f0p)+ B*R*phi0inv
    T_rad = np.imag(f0p)**2 - np.real(f0pp) + k_0**2 - knl
    # sign of the sqrt term chosen to remove the np.imag(f0p) term
    T_pred -= np.sqrt(T_rad)
    return T_pred

def dM_model_002(f0p, f0pp, mu, B, R, k_0):
    knl = pppterm * mu
    T_pred = np.imag(f0p) + B*R*phi0inv
    T_rad = 1 + (k_0**2 - np.real(f0pp))/(np.imag(f0p)**2) - knl / (np.imag(f0p)**2)
    # sign of the sqrt term chosen to remove the np.imag(f0p) term
    T_pred -= np.imag(f0p) * np.sqrt(T_rad)
    return T_pred

def dM_model_003(f0p, k_new, mu, B, R, k_0):
    knl = pppterm * mu
    T_pred = np.imag(f0p) + B*R*phi0inv
    T_rad = 1 + (k_0**2 - k_new**2)/(np.imag(f0p)**2) - knl / (np.imag(f0p)**2)
    # sign of the sqrt term chosen to remove the np.imag(f0p) term
    T_pred -= np.imag(f0p) * np.sqrt(T_rad)
    return T_pred

# Im f' = Im psi ' - M 
def dM_model_004(f0p, k_new, mu, B, R, k_0):
    knl = pppterm * mu
    T_pred = B*R*phi0inv
    T_rad = (k_0**2 - k_new**2)/(np.imag(f0p)**2) - knl / (np.imag(f0p)**2)
    # sign of the sqrt term chosen to remove the np.imag(f0p) term
    T_pred -= np.imag(f0p) * (0.5 * T_rad - 0.24 * T_rad **2)
    return T_pred

def dM_model_004(f0p, k_new, mu, B, R, k_0):
    knl = pppterm * mu
    T_pred = B*R*phi0inv
    T_rad = (k_0**2 - k_new**2)/(np.imag(f0p)**2) - knl / (np.imag(f0p)**2)
    # sign of the sqrt term chosen to remove the np.imag(f0p) term
    T_pred -= np.imag(f0p) * (0.5 * T_rad - 0.24 * T_rad **2)
    return T_pred
    '''