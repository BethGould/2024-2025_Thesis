#--LIBRARIES--------
import scipy

#--CONSTANTS--------
h_pl = scipy.constants.h #J⋅Hz−1
hbar = scipy.constants.hbar #J⋅Hz−1
m_e = scipy.constants.m_e #kg
e_e = scipy.constants.e #C
c_l = scipy.constants.c #m/s
pi = scipy.constants.pi

kFAu = 1.20 * 10**10 #m-1

R_max = 10**(-6)
R_min = 0.57 * R_max
B_min = 15 * 10**(-4)
B_max = 130 * 10**(-4)
I_e   = 10**(-7) #A

phi0inv = e_e/hbar/2 #note speed of light not correct for units, pi cancels
pppterm = m_e/hbar/e_e #sets units for unknown constant strength of e-e interaction

#The defaults are rtol = 1e-3, atol = 1e-6 (from SciPy)
rtol = 2.0e-5
atol = 1.0e-7

#These don't have as much reasoning from the experiment
MU_min = 1.0e-12 #should give oscillation speed less than from B 
MU_max = 7.0e-4  #~max positive mu for proper functioning with k_F
DK_min = rtol    
DK_max = 190.0  # approaches the level of error in k_Fermi

DK_min = DK_min / 2.0 / R_max
DK_max = DK_max / 2.0 / R_max 
#dk goes from 0 to 1/(2*R) + a number of half cycles n/(2*R)
