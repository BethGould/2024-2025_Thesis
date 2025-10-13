#--LIBRARIES--------
import numpy as np
#from ee_lib import h_pl, hbar, m_e, e_e, c_l, pi, pppterm, phi0inv
from eelib.consts import pppterm, phi0inv

#--DERIVITIVE FUNCTIONS FOR ODE SOLVERS--
# y = [psi, dpsi/dx]

def psi_deriv(x, y, k, B, R, mu, mA):
    knl = mu * pppterm
    k0  = complex(np.square(k))
    k1  = 2j * complex(B * R * phi0inv)
    k2  = complex(np.square(B * R * phi0inv))
    abssqpsi = (np.square(np.real(y[0]))+np.square(np.imag(y[0])))

    dpdxx = complex(knl * abssqpsi) * y[0] - k0 * y[0] + k1 * y[1] + k2 * y[0]
    dpdx  = y[1]
    return [dpdx, dpdxx]

def psi_deriv_old(x, y, k, B, R, mu, mA):
    k0 = complex(np.square(k))
    k1 = 2j * complex(B * R * phi0inv)
    k2 = complex(np.square(B * R * phi0inv))
    
    dpdxx = k1 * y[1] + k2 * y[0] - k0 * y[0]
    dpdx = y[1]
    return [dpdx, dpdxx]

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
