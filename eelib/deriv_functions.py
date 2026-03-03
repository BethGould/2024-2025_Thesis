# This contains functions used by reference by the ODE solver. 
# They are included outside the class structure and in a unified form
# so that they can be easily switched out.

#--LIBRARIES--------
import numpy as np
from scipy.special import ellipj

from eelib.consts import pppterm, phi0inv

#--DERIVITIVE FUNCTIONS FOR ODE SOLVERS--
# x
# y  = [psi, dpsi/dx]
# k  -- electron wavenumber
# B  -- magnetic field strength
# R  -- radius -- note that here only B * R is relevent, not B and R independently
# mu -- ee coupling strength
# mA -- maximum amplitude -- used by events, not derivative solvers, but must be included

# I could combine B and R

# Within a loop, with ee coupling
def psi_deriv(x, y, k, B, R, mu, mA):
    knl = mu * pppterm
    k0  = complex(np.square(k))
    k1  = 2j * complex(B * R * phi0inv)
    k2  = complex(np.square(B * R * phi0inv))
    abssqpsi = (np.square(np.real(y[0]))+np.square(np.imag(y[0])))

    dpdxx = complex(knl * abssqpsi) * y[0] - k0 * y[0] + k1 * y[1] + k2 * y[0]
    dpdx  = y[1]
    return [dpdx, dpdxx]

# Within a loop, without ee coupling
def psi_deriv_old(x, y, k, B, R, mu, mA):
    k0 = complex(np.square(k))
    k1 = 2j * complex(B * R * phi0inv)
    k2 = complex(np.square(B * R * phi0inv))
    
    dpdxx = k1 * y[1] + k2 * y[0] - k0 * y[0]
    dpdx = y[1]
    return [dpdx, dpdxx]

# For the input to the loop, without magnetic field or coupling
def psi_deriv_0(x, y, k, B, R, mu, mA):
    k0 = complex(np.square(k))
    
    dpdxx = - k0 * y[0]
    dpdx = y[1]
    return [dpdx, dpdxx]

# Contains both versions together, for compactness.
# May or may not be used ...
def psi_deriv_full(x, y, k, B, R, mu, mA):
    knl = mu * pppterm
    k0  = complex(np.square(k))
    k1  = 2j * complex(B * R * phi0inv)
    k2  = complex(np.square(B * R * phi0inv))
    abssqpsi = (np.square(np.real(y[2]))+np.square(np.imag(y[2])))

    dpdxx2 = complex(knl * abssqpsi) * y[2] - k0 * y[2] + k1 * y[3] + k2 * y[2]
    dpdx2  = y[3]
    dpdxx = k1 * y[1] + k2 * y[0] - k0 * y[0]
    dpdx = y[1]
    return [dpdx, dpdxx, dpdx2, dpdxx2]  


# ------- FUNCTIONS FOR FUNCTION FITTING --------
# Fitting functions for trigonometric functions

def sin_fit(x, amp, k, theta):
    return amp * np.sin(k * x + theta)

def sin_fit_2(x, amp, k, x0):
    return amp * np.sin(k * (x - x0))

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