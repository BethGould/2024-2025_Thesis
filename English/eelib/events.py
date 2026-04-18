# events.py

# Author: Elizabeth Gould
# Date Last Edit: 03.03.2026

# Like the derivative functions, these are used by the loop class
# when it calls the ODE solver. They are referred to by handle.

#--LIBRARIES--------
import numpy as np

#--EVENTS----
# x
# y  = [psi, dpsi/dx]
# k  -- electron wavenumber
# B  -- magnetic field strength
# R  -- radius
# mu -- ee coupling strength
# mA -- maximum amplitude

# event1 -- used to cut off the solver when the amplitude gets too large (caused by some errors)
# event2 -- triggers when real(psi) = 0
# deriv_amp -- triggers when abs(psi)' = 0
# deriv_phase -- triggers when arg(psi)' = 0
# deriv_real -- triggers when real(psi)' = 0

#This event returns 0 when the absolute value of the wave function crosses a defined value
#all the input parameters are the same as all the derivative calculations, but 
#used is mA (the maximum amplitude) and not the other parameters
def event1(x, y, k, B, R, mu, mA):
    return np.abs(y[0]) - mA
#event 2 returns the real value of the wave function: it triggers for 0 values
#typical oscillations are just phase oscillations, so I need to find roots on the real axis
def event2(x, y, k, B, R, mu, mA):
    return np.real(y[0])
#an alternate event would be psi' = 0, which should work even if 0 is not the center
#here is the trigger for |psi|' = 0
def deriv_amp(x,y,k,B,R,mu,mA):
    d_A = np.real(y[1]/y[0]) * np.abs(y[0])
    return d_A
def deriv_phase(x,y,k,B,R,mu,mA):
    d_ph = np.imag(y[1]/y[0]) 
    return d_ph
def deriv_real(x,y,k,B,R,mu,mA):
    dA = deriv_amp(x,y,k,B,R,mu,mA)
    dp = deriv_phase(x,y,k,B,R,mu,mA)
    A = np.abs(y[0])
    p = np.angle(y[0])
    return dA * np.cos(p) - A * dp * np.sin(p)
