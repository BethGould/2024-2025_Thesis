# consts.py

# Author: Elizabeth Gould
# Date Last Edit: 03.03.2026

# Contains copies of fundamental constants from scipy, other used constants in the program,
# and defined mins, maxes, and tolerances.

#--LIBRARIES--------
import scipy

#--CONSTANTS--------

# Fundamental
h_pl = scipy.constants.h    # J⋅Hz−1
hbar = scipy.constants.hbar # J⋅Hz−1
m_e = scipy.constants.m_e   # kg
e_e = scipy.constants.e     # C
c_l = scipy.constants.c     # m/s
pi = scipy.constants.pi

# Fermi wavenumbers
kFAu = 1.20 * 10**10        # m-1
                            # Other materials can also be added as desired.
                            # Number used has low precision.

# Ranges of values from experiments.
R_max = 10**(-6)            # m
R_min = 0.57 * R_max        # %
B_min = 15 * 10**(-4)       # T
B_max = 130 * 10**(-4)      # T
I_e   = 10**(-7)            # A

# Units for various terms in differential equations
phi0inv = e_e/hbar/2   # note speed of light not correct for units, pi cancels
pppterm = m_e/hbar/e_e # sets units for unknown constant strength of e-e interaction

# Tolerance for integration.
# The defaults are rtol = 1e-3, atol = 1e-6 (from SciPy)
rtol = 2.0e-5
atol = 1.0e-7

# Other ranges of examination. (mu and psi'_0)
# These don't have as much reasoning from the experiment.
# They are not actually used in our program, as they are often set by hand.

MU_min = 1.0e-12 # should give oscillation speed less than from B 
MU_max = 1.0e-5  # even at this level, I have issues with my analysis
DK_min = rtol    
DK_max = 190.0  # approaches the level of error in k_Fermi

DK_min = DK_min / 2.0 / R_max
DK_max = DK_max / 2.0 / R_max 
#dk goes from 0 to 1/(2*R) + a number of half cycles n/(2*R)
