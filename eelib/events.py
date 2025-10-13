#--LIBRARIES--------
import numpy as np

#--EVENTS----
#This event returns 0 when the absolute value of the wavefunction crosses a defined value
#all the input parameters are the same as all the derivative calculations, but 
#used is mA (the maximum amplitude) and not the other parameters
def event1(x, y, k, B, R, mu, mA):
    return np.abs(y[0]) - mA
#event 2 returns the real value of the wavefunction: it triggers for 0 values
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

#What if I solve for abs(psi) rather than psi? Or abs and phi separately?
