# class loop 
# 
# Author: Elizabeth Gould
# Date Last Modified: 17.03.2026
#
# This class is the core of the IVP and BVP solver for the nonlinear Schrödinger equation for this 
# model. The class can solve either the IVP or the BVP on a single ring for a given set of parameters. 
#
# It can solve the equations for both the linear and nonlinear cases. There is no need to set the 
# nonlinearity parameter to zero in order to solve the linear case. Just call the methods which
# provide the solution for the linear case. This functionality was ensured for easy comparison between
# both solutions. Even the solution to the BVP for the linear case is saved, and not deleted upon 
# changes to the initial derivative, despite it having a different initial derivative.
#
# IVP and BVP solutions are separate functionalities of the class. The class can solve one or the other
# at once, but not both together. The IVP solver takes a derivative as an input and does not require
# solving the BVP to run, while the BVP solutions will not provide a matching solution when run using 
# the IVP solver. BVP full solutions are found using the modeled solutions of the IVP, which have been 
# previously found by analysis of the IVP solver solutions. You will need to call the methods which 
# give the modeled solution at a given x value to get the BVP solution. The BVP solvers also reset the 
# derivative, which will clear all of the IVP solver data. BVP solution data, however, is only cleared 
# upon resetting the other parameters, but is saved upon changes to only the derivative.
#
# Input parameters are:
# - ring radius (R) as a percent of the maximum expected ring radius, 
# - magnetic field strength (B) as a percent of the maximum expected field strength,
# - electron wave number (k and dk), where k defaults to the Fermi wave number for gold and dk is a
#        percent of the period of a sine function in a ring of the maximum length,
# - nonlinear coupling strength (mu) as a dimensionless coefficient,
# - initial wave function (A), which defaults to 1. + 0.j,
# - and the initial derivative of the wave function, which is to be found by solving the BVP.
#
# The solutions to the BVP are modeled assuming the loop material is gold, and will not be valid for 
# other materials. The initial wave function is assumed to be 1, and changes to this will correspond 
# to a change in mu. (mu_eff = mu * |A|^2 with mu and A being the inputs, and mu_eff being what our 
# solutions correspond to.) dk values of 0 will cause problems for the BVP solution for the linear 
# case, but are assumed to not be physical, but rather just a relic of our imprecision.

'''
The code and documentation of this file are split into the following sections:
1. Libraries -- This is where I import external libraries or code from elsewhere in this library.
2. Default values -- This is where I set the default values used by the code. Default values can 
   be changed either directly in the code, or by changing the variable in your loop class instance 
   which stores this default value. You can also just call the relevant methods with your 
   preferred value and ignore the default. 
   There are a few exceptions, where the default values are stored in the consts.py file and these 
   values can't be modified within the class structure, but can be ignored by providing your own
   value.
3. Table of contents -- Lists the sections of the loop class and the methods contained within.
4. Variables -- Lists the variables contained within the loop class and their purpose.
5. Methods to call -- Provides a short description of all the relevant methods of the class.
6. class loop -- Our code.
'''

# ---- LIBRARIES ------------
# Numerical analysis libraries
import numpy as np
import scipy

# constants
from eelib.consts import pi, kFAu, R_max, B_max, phi0inv, rtol, atol
# function handles
from eelib.deriv_functions import psi_deriv, psi_deriv_old
from eelib.events import deriv_amp, deriv_real
# models
from eelib.k_M_models_ivp import pred_fast_k, pred_slow_k_v3, pred_fast_k_true
from eelib.bvp_rootfinder_functions import function_wrapper_bvp

# ---- DEFAULT VALUES ------
# Tolerances for error
default_tol_root_finder    = 1e-20
default_tol_bvp_matching   = 0.01
# SciPy algorithm choices
default_rootfinder         = 'broyden2'
default_integration_method = 'RK45'
# Chosen from our libraries, based on modeling of the IVP solutions
default_k_model_bvp        = pred_fast_k_true
default_k_model_ivp        = pred_fast_k   
default_M_model            = pred_slow_k_v3

# Default positive integers (counts)
# How many fast oscillation cycles should be averaged to estimate the period
default_number_averaged_initial_oscillations        = 20
# How many points should the integrator return
default_number_of_points_from_integration           = 1000 
# How frequently should we recover our amplitude (delete error)
default_number_fast_oscillations_between_recoveries = 4


# ---- TABLE OF CONTENTS --------

# 1 ----- Defining constants
#    __init__(R, B, dk, mu, k = kFAu, amp=1.)
#    ivp_solved(ivp_type="er")
#    clear_solution()
#    update_params(R=-1., B=-1., dk=-1., mu=-1., k=-1., amp=-1.)
#    calcInit()
#    setDeriv(p_prime)

# 2 ----- Analytic calculations of the case without e-e interaction
#    rem(number)
#    adjdiv1r()
#    bjdiv1r()
#    normToAmp()
#    aj_calc()
#    bj_calc()
#    ajN_calc()
#    bjN_calc()
#    psij(x)
#    psij_pred(x)
#    psij_pred_true(x)
#    psij0(x)
#    psi_prime_0()
#    psi_prime_0_max()
#    amp_max()
#    amp_max_0()

# 3 ----- ODE solver wrappers
#    get_solve_code(solve_arr)
#    solve_ivp(n=None, percent_range = 1.0, method = None, rtol = rtol, atol = atol, solve = 1)
#    ivp_solver_steps(t0, tf, y0, yp0, n=None, ee_int=True, m=None, method = None, rtol = rtol, atol = atol, fullSol = False)
#    call_ivp_solver(t0, tf, y0, yp0, n=0, fullSol = True, t0_start = False, ee_int= True, method = None, rtol = rtol, 
#                           atol=atol, first_step = None, max_step = np.inf, t_eval = None)

# 4 ----- Fast oscillation calculations
#    find_fast_oscillations(n=None, method = None, rtol = rtol, atol = atol)
#    find_start_exact()
#    find_start_fast_oscillations(n, sol)
#    find_period_fast_oscillations(n, sol)
#    find_amplitude_fast_oscillations(sol)

# 5 ----- Slow oscillation and coupled calculations
#    find_period_shift_exact()
#    find_slow_oscillations_start()
#    find_real_env_start()
#    find_t_points(n_points, t_max, t_start, T)

# 6 ----- BVP matching methods
#    find_root_rand(method = None, tol = None, ratio = 1.0)
#    find_root_many(tol_root_finder = None, tol_bvp_matching = None)
#    check_solution_for_boundary_matching(deriv_psi = None, tol = None)
#    find_root_min(tol_root_finder = None, tol_bvp_matching = None)

# 7 ----- Finding current
#    current_old()
#    curent_bvp()
#    curent_new()
#    curent_alt()
#    current_calc(psi=None, psi_pr=None)


# ---- VARIABLES -----

# Do not change variables by hand, with the exception of the indicated defaults.
# loop parameters are changed with update_params and setDeriv methods, while others 
# are set by running the code, not given by the user.

# -- Inputs
# The parameters of our loop
#   R (m)
#   k (hz), k0 + dk
#   B (T)
#   mu 
#   amp
# Saved inputs for when we require k0 and dk separately
#   k0 -- saves our input
#   dk -- saves our input
# Calculated initially, but can be set with setDeriv
#   psi0_deriv_0 -- psi' for time 0

# -- Derived quantities
# All these are calculated when our parameters are set.
#   lngt     = 2 * pi * self.R
#   period_k = 2 * self.R * self.k
#   T0       = 2. * pi / self.k     
#   M        = self.B*self.R*phi0inv
#   denrem   = self.rem(self.R*self.M)
#   aj
#   bj
#   aj0      -- for matched exact solution
#   bj0      -- for matched exact solution
# Periods from our models. Also calculated when parameters are set.
#   T_fast_mod      - half period of fast oscillation, with error
#   T_fast_mod_true - half period of fast oscillation without error
#   T_slow_mod      - period of slow oscillation
#
# These are calculated with find_fast_oscillations.
#   T_fast   -- estimated half period of fast oscillations
#   A_max    -- maximum absolute value of the wave function of the first n fast oscillations
#   T_fast0  -- known, error-free period of fast oscillations for the linear case
#   T_fast_ex -- known, error-free period of fast oscillations for the linear case
#   T_fast_0_calc -- estimated half period of fast oscillations for the linear case
# Starting values for our nonlinear, linear, and linear exact solutions for our integrator.
# They should be the first maxima and minima of our absolute value of our wave function.
#   stl
#   sth
#   stl0
#   sth0
#   stl_ex
#   stu_ex

# -- State indicators
#   k_ee_estimated -- Have we estimated our fast oscillation wave number for the nonlinear case from solving for the first n oscillations?
#   k_0_estimated  -- Have we estimated our fast oscillation wave number for the linear case from solving for the first n oscillations?
#   bvp_solved     -- Is our initial psi derivative that which has been determined by solving the BVP?
#   deriv_set      -- Has our derivative been changed from the solution to the linear BVP?

# -- Solutions to full IVP  
#   solu    -- recovered nonlinear solution
#   solu_d  -- decreasing nonlinear solution
#   solu0   -- decreasing linear solution
#   solu0_r -- recovered linear solution
#   solu_m  -- recovered nonlinear solution with points selected based on our modeled k
#
#   ivp     -- dictionary of the following solutions, with the following labels: ['er', 'ed', 'em', '0r', '0d']
#              'er' = recovered nonlinear
#              'ed' = decreasing nonlinear
#              'em' = recovered nonlinear, modeled
#              '0r' = recovered linear solution
#              '0d' = decreasing linear solution
#   percent_R_solved -- dictionary of the percent of the radius R each solution contains a solution for

# -- Solution to the BVP
# bvp_deriv  -- The derivative for which are parameters solve the BVP. This is set when the BVP is solved,
#               so that if we later change our derivative, but not our parameters, the solution is saved.

# -- Defaults
# These may be altered by hand if needed. 
#   k_model_bvp -- function handle of model for fast oscillation wave number without error
#   k_model_ivp -- function handle of model for fast oscillation wave number with error
#   M_model     -- function handle of model for slow oscillation wave number
#   tol_root_finder
#   tol_bvp_matching
#   default_rootfinder         -- string of name of SciPy root finder algorithm
#   default_integration_method -- string of name of SciPy root finder algorithm
#   default_no_ave_init_osc
#   default_no_int_points
#   default_recovery_rate
#
#   solve_mu_0 -- This is a boolean which determines if the estimation for the fast 
#                 oscillation wave number should include and estimation for the linear case.
#                 It defaults to True. It is required for accurate envelope following of the 
#                 linear case (mu = 0) for the IVP solutions due to the error in period.
#                 If you want to solve the nonlinear IVP without solving the linear IVP, 
#                 setting this to false will save some time.


# ---- METHODS TO CALL --------

# 1 ----- Defining Constants
#    __init__(R, B, dk, mu, k = kFAu, amp=1.)
#    update_params(R=-1., B=-1., dk=-1., mu=-1., k=-1., amp=-1.) 
#        -- Will change all passed parameters of the loop, while keeping the old values of those not passed.
#    setDeriv(p_prime)                 -- Set the initial derivative of psi to the given value.
#    ivp_solved(ivp_type="er")         -- Indicates if the queried solution exists.

# 2 ----- Analytic calculations of the case without e-e interaction
#    psij(x)           -- Returns exact solution of psi without e-e interaction, for the set derivative.
#    psij_pred(x)      -- Returns modeled solution of psi with e-e interaction, for the set derivative.
#    psij_pred_true(x) -- Returns modeled solution without error of psi with e-e interaction, for the set derivative.
#    psij0(x)          -- Returns exact solution of psi without e-e interaction, for the BVP solution.
#    psi_prime_0_max() -- Returns the maximum abs of psi' without e-e interaction for the BVP solution. 
#                         For use in constructing derivative grids.
#    amp_max()         -- Returns the maximum absolute value of psi for the modeled solution without error of psi with
#                         e-e interaction, for the set derivative.
#    amp_max_0()       -- Returns the maximum absolute value of psi for the exact solution of psi without
#                         e-e interaction, which solves the BVP.

# 3 ----- ODE solver wrappers
#    get_solve_code(solve_arr)  -- Translates a list of codes ('er', 'ed', '0r', '0d', 'em') into an integer for 
#                                  the solve parameter of solve_ivp. 
#    solve_ivp(n=None, percent_range = 1.0, method = None, rtol = rtol, atol = atol, solve = 30)
#           -- Used for running the ivp for the set loops. 

# 4 ----- Fast oscillation calculations
#    find_fast_oscillations(n=None, method = None, rtol = rtol, atol = atol)
#        -- Estimates the half period of the fast oscillation from the first n oscillations of the absolute value of psi.
#           It can be run separately of the IVP solver, but the IVP solver will call this method if needed on its own. 
#           There is a separate input of solve_mu_0, which indicated whether or not to estimate k for the linear case. 
#           It is set to True by default, and must be set to False by hand to avoid this extra calculation. The IVP solver
#           will set this to True if it is needed, but will ignore this if the linear solutions are not calculated.

# 5 ----- Slow oscillation and coupled calculations
#    find_period_shift_exact() -- This was my attempt to follow and draw the envelope of the fast oscillations for the 
#                                 exact case. I don't remember it ever working. Eventually I declared it unnecessary.
#    find_real_env_start() -- Gives the time to start following the envelope created by the fast oscillations.
#    find_t_points(n_points, t_max, t_start, T) -- Creates a numpy array of points for which to integrate or plot, 
#                                                  for which the spacing is guaranteed to be a multiple of T. Used
#                                                  to keep the evaluation at the same point on the fast oscillations. 

# 6 ----- BVP matching
#    find_root_rand(method = None, tol = None, ratio = 1.0)
#       -- Tries to find a solution to the BVP using the indicated algorithm and starting point.
#          The starting point is given by the linear BVP solution divided by ratio.
#    find_root_many(tol_root_finder = None, tol_bvp_matching = None)
#       -- Tries to find the solution to the BVP using multiple algorithms and starting points. 
#          Returns a list of found solutions.
#    check_solution_for_boundary_matching(deriv_psi = None, tol = None)
#       -- Checks if the psi derivative solves the BVP (based on the modeled solution).
#    find_root_min(tol_root_finder = None, tol_bvp_matching = None)
#       -- Sets the derivative to the solution of the BVP.

# 7 ----- Finding current
#    current_old() -- Returns the current of the linear BVP-solving case with the loop parameters.
#    current_bvp() -- Solves the bvp and returns the current of this nonlinear solution using the model
#                     for current of current_alt.
#    current_new() -- Returns the current of the nonlinear solution of the saved loop, based
#                     on the constant term of the exact solution.
#                     This estimate for current is most likely incorrect.
#    current_alt() -- Returns the current of the nonlinear solution of the saved loop, based
#                     on the hypothesized form of this current. It is likely correct.
#    current_calc(psi=None, psi_pr=None)   -- Returns the current of the nonlinear solution of the loop
#                                             at the saved or indicated psi and psi prime values, based on 
#                                             the equation to calculate current. 
#                                             




# ---- LOOP CLASS ----------
class loop:

    # constructor
    # On creation, it takes R, B, dk, mu, k, and amp as parameters.
    # R is radius as a percentage of R_max.
    # B is the magnetic field as a percentage of B_max.
    # The wave function of the electron is k_el = k + dk/R_max /2.0.
    # k defaults to kFAu (assuming gold as the conductor).
    # psi(0) = amp
    # psi'(0) is initially defined based on the known solution for mu = 0.
    def __init__(self, R, B, dk, mu, k = kFAu, amp=1.):

        # parameters
        self.R   = R * R_max #R in micrometers
        self.k   = k + dk/R_max/2.0 #dk adds to min k
        self.B   = B * B_max #scales with 130 G
        self.mu  = mu
        self.amp = complex(amp)
        self.dk  = dk
        self.k0  = k 

        # defaults
        # These are changed by hand or by changing the indicated constants in the code.
        # They are used only if the indicated parameter is not passed to the methods.
        self.k_model_bvp = default_k_model_bvp
        self.k_model_ivp = default_k_model_ivp
        self.M_model     = default_M_model

        self.tol_root_finder            = default_tol_root_finder
        self.tol_bvp_matching           = default_tol_bvp_matching
        self.default_rootfinder         = default_rootfinder
        self.default_integration_method = default_integration_method
        self.default_no_ave_init_osc    = default_number_averaged_initial_oscillations
        self.default_no_int_points      = default_number_of_points_from_integration
        self.default_recovery_rate      = default_number_fast_oscillations_between_recoveries

        self.solve_mu_0 = True # Set to False by hand to save some time estimating the half period of fast oscillations.

        # This indicates which steps have been taken so far in the solution process.
        self.deriv_set  = False   # False = default derivative, True = chosen derivative
        self.bvp_deriv  = None  # This has not yet been calculated.
        self.clear_solution()  # Initializes variables to be unset.

        self.calcInit() # Calculate many derived quantities.

    def ivp_solved(self, ivp_type = "er"):
        if self.ivp[ivp_type] is None: 
            return False
        else:
            return True

    # Remove the IVP solutions (since they may be large), and indicate that nothing has been solved 
    # (as any previous solution may be erroneous now).
    def clear_solution(self):
        self.k_ee_estimated = False
        self.k_0_estimated  = False

        self.ivp = {"er": None,   # with e-e interaction, recovered solution
                    "ed": None,   # with e-e interaction, decreasing solution
                    "em": None,   # with e-e interaction, recovered solution, with spacing based on predicted k
                    "0r": None,   # without e-e interaction, recovered solution
                    "0d": None}   # without e-e interaction, decreasing solution
        self.percent_R_solved = {"er": None,   # with e-e interaction, recovered solution
                    "ed": None,   # with e-e interaction, decreasing solution
                    "em": None,   # with e-e interaction, recovered solution, with spacing based on predicted k
                    "0r": None,   # without e-e interaction, recovered solution
                    "0d": None}   # without e-e interaction, decreasing solution

        self.bvp_solved = False

        self.solu    = None
        self.solu_d  = None
        self.solu_m  = None
        self.solu0   = None
        self.solu0_r = None

    # The default -1 means that that parameter is not to be changed.
    # Otherwise, the parameter is updated to the given values.
    # Then all initialization steps are to be performed with the new parameters, 
    # clearing the old data, which is now incorrect.
    def update_params(self, R=-1., B=-1., dk=-1., mu=-1., k=-1., amp=-1.):
        if R > 0.:
            self.R = R * R_max
        if B > 0.:           
            self.B = B * B_max
        if amp > 0.:
            self.amp = complex(amp)
        if mu > 0.:
            self.mu = mu
        if k > 0. and dk > 0.:
            self.k = k + dk /R_max/ 2.0
            self.dk = dk
            self.k0 = k
        elif dk > 0.:
            self.k = self.k0 + dk / R_max/ 2.0
            self.dk = dk
        elif k > 0.:
            self.k = k + self.dk / R_max / 2.0
            self.k0 = k

        self.clear_solution() # Clear old solutions.
        self.deriv_set = False # False = default derivative, True = chosen derivative
        self.bvp_deriv = None # Clear old solution.

        self.calcInit() # Recalculate derived quantities.
        
    # Sets a bunch of constants based on the data defining the loop.
    def calcInit(self):
        self.lngt     = 2 * pi * self.R
        self.period_k = 2 * self.R * self.k
        self.T0 = 2. * pi / self.k
        
        self.M      = self.B*self.R*phi0inv
        self.denrem = self.rem(self.R*self.M)
        self.aj     = self.aj_calc()
        self.bj     = self.bj_calc()
        self.aj0    = self.aj
        self.bj0    = self.bj
        
        self.psi0_deriv_0 = self.psi_prime_0()

        self.T_fast_mod      = pi / self.k_model_ivp(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_fast_mod_true = pi / self.k_model_bvp(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_slow_mod      = 2 * pi / self.M_model(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        

    # This method is for setting the initial psi prime value of the loop.
    # It is meant to be called numerous times.
    def setDeriv(self, p_prime):
        self.psi0_deriv_0 = p_prime

        self.aj = self.ajN_calc()
        self.bj = self.bjN_calc()

        self.clear_solution()
        self.deriv_set = True # False = default derivative, True = chosen derivative

        self.T_fast_mod      = pi / self.k_model_ivp(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_fast_mod_true = pi / self.k_model_bvp(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_slow_mod      = 2 * pi / self.M_model(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
                

    # 2 ----- Analytic calculations 
    # of the case without e-e interaction
    # and for the models with e-e interaction.

    # aj and bj terms, so that aj + bj = amp
    def rem(self, number):
        resnum = np.modf(number)
        return resnum[0]
    
    def ajdiv1r(self):
        numrem = self.rem(self.B*np.square(self.R)*phi0inv-self.R*self.k)
        denom = np.exp(2j*pi*self.denrem) * 2j * np.sin(2*pi*self.k*self.R)
        num = 1 - np.exp(2j*pi*(numrem))
        return - num / denom
    
    def bjdiv1r(self):
        numrem = self.rem(self.B*np.square(self.R)*phi0inv+self.R*self.k)
        denom = np.exp(2j*pi*self.denrem) * 2j * np.sin(2*pi*self.k*self.R)
        num = 1 - np.exp(2j*pi*(numrem))
        return num / denom
    
    def normToAmp(self):
        return self.amp / (self.ajdiv1r()+self.bjdiv1r())
    
    def aj_calc(self):
        return self.ajdiv1r()*self.normToAmp()
    
    def bj_calc(self):
        return self.bjdiv1r()*self.normToAmp()
    
    # Used when setting psi prime with setDeriv.
    def ajN_calc(self):
        den = 1.0 / 2.0 / self.k
        reA = den * (np.imag(self.psi0_deriv_0) - self.M + self.k)
        imA = - den * np.real(self.psi0_deriv_0)

        ret = 0.5*((self.psi0_deriv_0 - 1j * self.M * self.amp)/(1j*self.k)+self.amp)

        return ret
    
    def bjN_calc(self):
        den = 1.0 / 2.0 / self.k
        reB = den * (-np.imag(self.psi0_deriv_0) + self.M + self.k)
        imB = - den * np.real(self.psi0_deriv_0)

        ret = 0.5*(self.amp - (self.psi0_deriv_0 - 1j * self.M * self.amp)/(1j*self.k))

        return ret
    
    # Exact solution with the current derivative
    def psij(self, x):
        return self.aj*np.exp(1j*x*(self.k+self.M)) + self.bj*np.exp(1j*x*(-self.k+self.M))
    
    # Modelled solution with the current derivative, with error
    def psij_pred(self, x):
        kn = pi / self.T_fast_mod
        Mn = 2 * pi / self.T_slow_mod
        return self.aj*np.exp(1j*x*(kn+Mn)) + self.bj*np.exp(1j*x*(-kn+Mn))

    # Modelled solution with the current derivative, without error
    def psij_pred_true(self, x):
        kn = pi / self.T_fast_mod_true
        Mn = 2 * pi / self.T_slow_mod
        return self.aj*np.exp(1j*x*(kn+Mn)) + self.bj*np.exp(1j*x*(-kn+Mn))
    
    # Exact solution of the BVP
    def psij0(self, x):
        return self.aj0*np.exp(1j*x*(self.k+self.M)) + self.bj0*np.exp(1j*x*(-self.k+self.M))
    
    # Derivative of the exact solution of the BVP
    def psi_prime_0(self):
        return 1j * (self.aj0*(self.k+self.M)+self.bj0*(-self.k+self.M))
    
    # This is the max absolute value of psi' without e-e interaction which solves the BVP.
    def psi_prime_0_max(self):
        #psi = e^(iMx)*[ae^(ikx)+be^(-ikx)]
        #psi' = iM*psi + ik* e^(iMx)*[ae^(ikx)-be^(-ikx)]
        #abs(psi') = abs[M *(ae^(ikx)+be^(-ikx)) + k* (ae^(ikx)-be^(-ikx)]
        # <= (M+k)*[abs(a)+abs(b)]
        d_max = (self.M+self.k)*(np.abs(self.aj0)+np.abs(self.bj0))
        return d_max
    
    # This is the max absolute value of psi for the current derivative.
    def amp_max(self):
        return np.sqrt(np.abs(self.aj)**2 + np.abs(self.bj)**2 + 2*np.abs(self.aj*np.conj(self.bj)))
    
    # This is the max absolute value of psi without e-e interaction which solves the BVP.
    def amp_max_0(self):
        return np.sqrt(np.abs(self.aj0)**2 + np.abs(self.bj0)**2 + 2*np.abs(self.aj0*np.conj(self.bj0)))
    
    # 3 ---- ODE methods

    # Translates the intuitive plot_code system into the unintuitive integer code system for solve in solve_ivp.
    def get_solve_code(solve_arr):
        code = -1
        if "er" in solve_arr: code = code * -1
        if "ed" in solve_arr: code = code * 2
        if "0r" in solve_arr: code = code * 5
        if "0d" in solve_arr: code = code * 3
        if "em" in solve_arr: code = code * 7
        return code

    # Solves the indicated IVPs using the indicated numerical ODE solver.
    def solve_ivp(self, n = None, percent_range = 1.0, method = None, rtol = rtol, atol = atol, solve=1):
        # I have changed how I set defaults values here, to allow editing of the defaults
        # in either the object or the code.
        if method is None:
            method = self.default_integration_method
        if n is None:
            n = self.default_no_int_points

        if (solve % 3 == 0 or solve % 5 == 0) and (self.solve_mu_0 == False):
            self.solve_mu_0 = True
            self.find_fast_oscillations()

        if self.k_ee_estimated == False:
            self.find_fast_oscillations()
 
        t0h = self.sth
        tfh = self.lngt * percent_range  # t_evalh[-1]
        #t0l = self.stl
        #tfl = self.lngt * percent_range  #t_evall[-1]

        if solve % 3 == 0 or solve % 5 == 0:
            t0h0 = self.sth0
            tfh0 = self.lngt * percent_range #t_vals[1]
            #t0l0 = self.stl0
            #tfl0 = self.lngt * percent_range #t_vals[3]

        #calculate y0
        y0h = self.amp
        yp0h = self.psi0_deriv_0
        #y0l = self.amp
        #yp0l = self.psi0_deriv_0

        y0h0, yp0h0 = self.amp, self.psi0_deriv_0 #self.calc_yval_old(n, t0h0)
        #y0l0, yp0l0 = self.amp, self.psi0_deriv_0 #self.calc_yvals_old(n, t0l0)

        # call the solver multiple times  
        # with e-e-interaction -- divided into steps (+), original(2), 
        # with predicted k_fast (7)
        # without e-e-interaction -- original(3), divided into steps(5)
        if solve > 0:      
            self.solu = self.ivp_solver_steps(t0h, tfh, y0h, yp0h, n, fullSol = False, ee_int = True, 
                                              method = method, rtol = rtol, atol = atol)
            #self.soll = self.ivp_solver_steps(t0l, tfl, y0l, yp0l, n, fullSol = False, ee_int = True, 
            #                                   method = method, rtol = rtol, atol = atol)
            self.percent_R_solved["er"] = percent_range
            self.ivp["er"] = self.solu
        if abs(solve)%2 == 0:
            self.solu_d = self.call_ivp_solver(t0h, tfh, y0h, yp0h, n, fullSol = False, ee_int = True, 
                                               method = method, rtol = rtol, atol = atol)
            #self.soll_d = self.call_ivp_solver(t0l, tfl, y0l, yp0l, n, fullSol = False, ee_int = True, 
            #                                    method = method, rtol = rtol, atol = atol)
            self.percent_R_solved["ed"] = percent_range
            self.ivp["ed"] = self.solu_d
        if abs(solve)%3 == 0:
            self.solu0 = self.call_ivp_solver(t0h0, tfh0, y0h0, yp0h0, n, fullSol = False, ee_int = False, 
                                              method = method, rtol = rtol, atol = atol)
            #self.soll0 = self.call_ivp_solver(t0l0, tfl0, y0l0, yp0l0, n, fullSol = False, ee_int = False, 
            #                                   method = method, rtol = rtol, atol = atol)
            self.percent_R_solved["0d"] = percent_range
            self.ivp["0d"] = self.solu0
        if abs(solve)%5 == 0:
            self.solu0_r = self.ivp_solver_steps(t0h0, tfh0, y0h0, yp0h0, n, fullSol = False, ee_int = False, 
                                                 method = method, rtol = rtol, atol = atol)
            #self.soll0_r = self.ivp_solver_steps(t0l0, tfl0, y0l0, yp0l0, n, fullSol = False, ee_int = False, 
            #                                      method = method, rtol = rtol, atol = atol)
            self.percent_R_solved["0r"] = percent_range
            self.ivp["0r"] = self.solu0_r
        if abs(solve)%7 == 0: 
            self.solu_m = self.ivp_solver_steps(t0h, tfh, y0h, yp0h, n, fullSol = False, ee_int = True, 
                                              method = method, rtol = rtol, atol = atol, estimate_k = True)
            #self.soll_f = self.call_ivp_solver(t0l, tfl, y0l, yp0l, n, fullSol = False, ee_int = True, 
            #                                   method = method, rtol = rtol, atol = atol)  
            self.percent_R_solved["em"] = percent_range
            self.ivp["em"] = self.solu_m      

    # This is designed to correct for the drop in power over continuous cycles by just adding it back in.
    # Calls the solver multiple times for shorter intervals.
    def ivp_solver_steps(self, t0, tf, y0, yp0, n=None, ee_int=True, m = None, method = None, rtol = rtol, atol = atol, fullSol = False, estimate_k = False):

        if method is None:
            method = self.default_integration_method
        if n is None:
            n = self.default_no_int_points
        if m is None:
            m = 4

        # Find all of the points we wish to solve for.
        if ee_int:
            if estimate_k:
                t_eval_full = self.find_t_points(n, t_max = tf, t_start = t0, T = 2*self.T_fast_mod)
            else:
                t_eval_full = self.find_t_points(n, t_max = tf, t_start = t0, T = 2*self.T_fast)
        else:
            t_eval_full = self.find_t_points(n,t_max = tf, t_start = t0, T = self.T_fast0)

        # desired step size for our solver
        first_step = max_step = t_eval_full[1]-t_eval_full[0]

        # initialization
        t0s = t0
        y0s = y0
        yp0s = yp0
        sol = []
        start_high = [[],[]]
        end_high = [[],[]]
        i = 1
        if m > len(t_eval_full): m = len(t_eval_full)
        if m == len(t_eval_full): tfs = tf
        else: tfs = t_eval_full[m + 1]

        t0_start = False

        # Solve until the end of the chain.
        while tfs <= tf:
            
            # define points to find
            if i+m+1 > len(t_eval_full): 
                t_eval_s = t_eval_full[i:]
                tfs = tf
            else: 
                t_eval_s = t_eval_full[i:i+m] 
                tfs = t_eval_full[i+m]

            if len(t_eval_s) == 0:
                break 

            # Solve the next step.
            sol.append(self.call_ivp_solver(t0s, tfs, y0s, yp0s, fullSol = False, t0_start=t0_start, ee_int = ee_int, method = method, 
                        rtol = rtol, atol = atol, first_step=first_step, max_step = max_step, t_eval = t_eval_s))
            
            # Too short
            if len(sol[-1]['y']) == 0:
                sol.pop()
                i += m
                continue

            # event trigger problems 
            if np.shape(sol[-1]['y_events'][0])[0] < 2:
                sol.pop()
                i += m
                continue

            # find final solutions
            yfs = sol[-1]['y_events'][0][-1,0]
            ypfs = sol[-1]['y_events'][0][-1,1]
            tfs = sol[-1]['t_events'][0][-1]

            y0a = sol[-1]['y_events'][0][0,0]
            y0a2 = sol[-1]['y_events'][0][1,0]

            t0_start = True
            
            # Choose the same abs event side.
            
            if abs(yfs) < abs(y0a) and abs(yfs) < abs(y0a2):
                rto = min(abs(y0a), abs(y0a2)) /abs(yfs)
                end_high[0].append(False)
            else:
                rto = max(abs(y0a), abs(y0a2))/abs(yfs)
                end_high[0].append(True)
            if abs(y0a) < abs(y0a2):
                start_high[0].append(False)
            else:
                start_high[0].append(True)

            # Choose the same re event side.
            y_event_first = np.real(sol[-1]['y_events'][1][0,0])
            y_event_last = np.real(sol[-1]['y_events'][1][-1,0])

            if y_event_last < 0:
                end_high[1].append(False)
            else:
                end_high[1].append(True)
            if y_event_first < 0:
                start_high[1].append(False)
            else:
                start_high[1].append(True)

            # Rescale for the new run.

            i += m
            t0s = tfs
            y0s = yfs * rto
            yp0s = ypfs * rto


        # combine full solution
        # t, y, t_events, y_events, status   
        # t

        sol_t_list = []
        sol_y_list = []
        sol_out_tev = []
        sol_out_yev = []

        for ev in sol:
            sol_t_list.append(ev['t'][1:])
            sol_y_list.append(ev['y'][:,1:])

        if len(sol) == 0:
            return {
                't': np.array([0]),
                'y': np.array([0,0]),
                'status': [3],
                't_events': [np.array([0]), np.array([0])],
                'y_events': [np.array([0,0]),np.array([0,0])]
            }    

        for i in range(len(sol[0]['t_events'])):
            sol_tev_list = []
            sol_yev_list = []

            for j,ev in enumerate(sol):

                if j > 0 and end_high[i][j-1]==start_high[i][j]:
                    sol_tev_list.append(ev['t_events'][i][1:])
                    sol_yev_list.append(ev['y_events'][i][1:, :])
                else:
                    sol_tev_list.append(ev['t_events'][i])
                    sol_yev_list.append(ev['y_events'][i])


            sol_out_tev.append(np.concatenate(tuple(sol_tev_list)))
            sol_out_yev.append(np.concatenate(tuple(sol_yev_list), axis = 0))


        sol_out_t = np.concatenate(tuple(sol_t_list))
        sol_out_y = np.concatenate(tuple(sol_y_list), axis = 1)
        
        sol_out_status = []
        for ev in sol:
            sol_out_status.append(ev['status'])

        
        return {
            't': sol_out_t,
            'y': sol_out_y,
            'status': sol_out_status,
            't_events': sol_out_tev,
            'y_events': sol_out_yev
        }

    # wrapper for the library call to the IVP solver
    def call_ivp_solver(self, t0, tf, y0, yp0, n = 0, fullSol = True, t0_start=False, ee_int = True, method = None, 
                        rtol = rtol, atol = atol, first_step=None, max_step = np.inf, t_eval = None):

        if method is None:
            method = self.default_integration_method

        # choose solver
        if ee_int:
            y_hand = psi_deriv
        else:
            y_hand = psi_deriv_old
            
        # range, initial values, and parameters
        if t0_start:
            t_range = [t0,tf]
        else: 
            t_range = [0.0, tf] #sequence
        y_0 = np.array([y0, yp0]) #array_like, shape (n,) 
        arg = (self.k, self.B, self.R, self.mu, 1000.) #tuple
        
        # events
        elist = [deriv_amp, deriv_real]
        
        # call solver
        if fullSol:
            sol = scipy.integrate.solve_ivp(y_hand, t_range, y_0, method=method,first_step=first_step, 
                                            max_step=max_step, t_eval=t_eval, args=arg, events=elist, rtol = rtol, atol = atol) 
        else:
            sol = scipy.integrate.solve_ivp(y_hand, t_range, y_0, method=method,first_step=first_step, 
                                            max_step=max_step, t_eval=t_eval, args=arg, events=elist, rtol = rtol, atol = atol) 
        # return the solution
        return sol

    # 4 ------ Calculations for the fast oscillations ------

    # These find the (half) period and starting point of the fast oscillations

    # This estimates the half period of the fast oscillations by averaging the spacing between n 
    # oscillations of the squared absolute value of psi, found from our ODE solver.
    # self.solve_mu_0 can be flipped to False to remove our linear solution estimate and speed up 
    # calculations. We estimate this because T0 doesn't correctly estimate the period due to error. 
    def find_fast_oscillations(self, n=None, method = None, rtol = rtol, atol = atol):

        if method is None:
            method = self.default_integration_method
        if n is None:
            n = self.default_no_ave_init_osc

        # choose percent_range

        maxX = n*self.T0
        per_range = maxX / self.lngt

        sol1 = self.call_ivp_solver(0.0, self.lngt *per_range, self.amp, self.psi0_deriv_0, 
                                    method = method, rtol = rtol, atol = atol) # find the first n cycles
        if self.solve_mu_0: 
            sol2 = self.call_ivp_solver(0.0, self.lngt *per_range, self.amp, self.psi0_deriv_0, ee_int = False,
                                    method = method, rtol = rtol, atol = atol) # find the first n cycles

        # find the period of the events
        self.T_fast = self.find_period_fast_oscillations(n, sol1)
        self.A_max = self.find_amplitude_fast_oscillations(sol1)
        if self.solve_mu_0: self.T_fast_0_calc = self.find_period_fast_oscillations(n, sol2)
        self.T_fast0 = self.T0
        self.T_fast_ex = self.T0

        # find the starting point of the events
        start_ee = self.find_start_fast_oscillations(n, sol1)

        if self.solve_mu_0: 
            start_0 =  self.find_start_fast_oscillations(n, sol2)

        self.stl = start_ee[0]
        self.sth = start_ee[1]
        if self.solve_mu_0:
            self.stl0 = start_0[0]
            self.sth0 = start_0[1]
            self.k_0_estimated  = True

        self.stl_ex, self.stu_ex = self.find_start_exact()

        self.k_ee_estimated = True

    # return t_min, t_max
    def find_start_exact(self):
        # find extreema -- maximum, minimum, and two midpoints

        x1 = pi - np.angle(self.aj*np.conj(self.bj))/2.0
        x2 = x1 - pi / 2.0
        x1 = x1 / self.k
        x2 = x2 / self.k

        # order solutions
        if self.psij(x1) > self.psij(x2):
            x3 = x1
            x1 = x2
            x2 = x3

        return x1, x2

    # The first time our absolute value squared of our oscillations reach their minimum and maximum 
    # values, as extracted from our given ODE solution.
    def find_start_fast_oscillations(self,n,sol):
        t_1 = sol['t_events'][0][0]
        t_2 = sol['t_events'][0][1]
        y_1 = np.abs(sol['y_events'][0][0][0])
        y_2 = np.abs(sol['y_events'][0][1][0])
        
        if np.abs(y_2) < np.abs(y_1):
            t_low = t_2
            t_high = t_1
        else:
            t_low = t_1
            t_high = t_2
        return [t_low, t_high]
        
    # Estimate the period of oscillation from our given solution, calculated by estimating the 
    # first n events (extrema of our absolute value).
    def find_period_fast_oscillations(self,n,sol):
        t = sol['t_events'][0]
        t2 = t[1:n]
        t1 = t[0:n-1]
        td = t2 - t1
        dt = np.average(td)
        return dt
    
    # Find the maximum absolute value recorded in our events.
    def find_amplitude_fast_oscillations(self,sol):
        amp = np.abs(sol['y_events'][0][:, 0])
        max = np.max(amp)
        return max
    
    def find_period_shift_exact(self):
        fast = self.T0
        slow = 2.*pi / self.M
        pos = 2.*pi/(self.k + self.M)
        neg = 2.*pi/(self.k - self.M)
        return fast, slow, pos, neg
    
    def find_slow_oscillations_start(self):
        den = 2 * self.M
        x1 = 1j * (np.log(self.aj)+np.log(self.bj)) / den # max
        x1 = np.real(x1)
        dx = pi / den
        T = pi / den
        nT = np.floor(x1 / (2. * T))
        x2 = x1 - nT * 2. * T
        x3 = x2 + T
        if x3 > 2.* T:
            x3 = x3 - 2. * T

        return x3, x2

    def find_real_env_start(self):
        xTmin, xTmax = self.find_slow_oscillations_start()
        xtmin, xtmax = self.find_start_exact()
        t = self.T0

        ymin = self.psij(xtmin)
        ymax = self.psij(xtmax)
        y = max(abs(ymin), abs(ymax))
        if y == ymin: xt = xtmin
        else: xt = xtmax

        n_osc = np.ceil((xTmin - xt) / t)
        x_st1 = xt + n_osc * t
        if self.psij(x_st1) == 0: x_st1 += t/4
        n_osc = np.ceil((xTmax - xt) / t)
        x_st2 = xt + n_osc * t
        if self.psij(x_st2) == 0: x_st2 += t/4
        y_st = max(abs(self.psij(x_st1)), abs(self.psij(x_st2)))
        if y_st == abs(self.psij(x_st1)): x_st = x_st1
        else: x_st = x_st2
        return x_st


    # Gives the target real(psi'(0)) and its corresponding t start values
    # for the upper and lower envelopes.
    def find_t_points(self, n_points, t_max, t_start, T):
        p0 = t_start
        nT = np.floor((t_max - t_start) / T)  
        count = np.floor(nT / (n_points - 1)) 
        
        pf = t_start + count * (n_points -1) * T
        x1 = np.linspace(p0, pf, n_points)
        return x1

    
    # 6 -- BVP methods

    # This will find a random solution to our BVP. Changing method and prep will change which solution
    # the root finder converges to. I can set this up more intelligently, but I want to have a solution
    # as fast as possible, and further examination can wait.
    def find_root_rand(self, method = None, tol = None, ratio = 1.0):
        mu = self.mu
        dk = self.dk
        B = self.B
        R = self.R

        if tol is None:
            tol = self.tol_root_finder
        if method is None:
            method = self.default_rootfinder


        model_k = self.k_model_bvp
        model_M = self.M_model

        #kfun = lambda dpsi, mu, dk, B, R, A, k0: self.k

        # Find the root of the linear equation
        #yy = lambda x: fun(x, mu, dk, B, R, k_calc_f=kfun, M_calc_f=M_calc_0)
        #jj = lambda x: jac(x, mu, dk, B, R, k_calc_f=k_calc_0, M_calc_f=M_calc_0, dk_calc_f=dk_calc_0, dM_calc_f=dM_calc_0)

        der_0 = self.psi_prime_0()
        #sol = scipy.optimize.root(yy, [-1e12, 1e12], method='hybr')

        xs = np.array([np.real(der_0), np.imag(der_0)])
        #print(sol.x)

        # Now find the root for the nonlinear solution
        yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=model_k, M_calc_f=model_M)
        #jj = lambda x: jac(x, mu, dk, B, R)

        sol = scipy.optimize.root(yy, xs/ratio, method=method, tol=tol)
        dpsi0c = sol.x

        deriv_psi = dpsi0c[0] + 1j * dpsi0c[1]
        #deriv_psi_o = xs[0]+1j*xs[1]

        return deriv_psi

    # This will find multiple solutions to our BVP. It appears to work well enough to enable a way around our problematic 
    # solutions. The solution with the smallest amplitude is our most important one, and not all solutions are found, only
    # only the smallest of them. The main issue with this algorithim is the time requirement, which will be a big cost on a 
    # big grid. 
    def find_root_many(self, tol_root_finder = None, tol_bvp_matching = None):

        if tol_root_finder is None:
            tol_root_finder = self.tol_root_finder
        if tol_bvp_matching is None:
            tol_bvp_matching = self.tol_bvp_matching

        mu = self.mu
        dk = self.dk
        B = self.B
        R = self.R

        model_k = self.k_model_bvp
        model_M = self.M_model

        der_0 = self.psi_prime_0()
        xs = np.array([np.real(der_0), np.imag(der_0)])

        method = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']
        prep = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0]

        # Now find the root for the nonlinear solution
        yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=model_k, M_calc_f=model_M)
        #jj = lambda x: jac(x, mu, dk, B, R)

        dpsi0c = []

        for mm in method:
            for pp in prep:
                try:
                    sol = scipy.optimize.root(yy, xs/pp, method=mm, tol=tol_root_finder)
                    dpsi0c.append(sol.x)
                except:
                    pass

        dpsi0f = []

        # Clear unsuccessful variants
        for ii in dpsi0c:
            deriv_psi = ii[0] + 1j * ii[1]
            try:
                self.setDeriv(deriv_psi)
                sol = self.psij_pred_true(2*pi*self.R) - 1.0
                if np.abs(sol) < tol_bvp_matching:
                    new = True
                    for jj in dpsi0f:
                        if np.abs((jj - deriv_psi)/(jj+deriv_psi)) < tol_bvp_matching:
                            new = False
                            break
                    if new: dpsi0f.append(deriv_psi)
            except:
                pass
            
        return dpsi0f

    # Checks to see if the derivative (pased or saved) is a solution to the BVP within the given tolerance.
    def check_solution_for_boundary_matching(self, deriv_psi = None, tol = None):

        if tol is None:
            tol = self.tol_bvp_matching

        if deriv_psi is not None:
            self.setDeriv(deriv_psi)
        sol = self.psij_pred_true(2*pi*self.R) - 1.0

        if np.abs(sol) < tol:
            return True
        else:
            return False
    
    # This will set the derivative to the correct solution of the BVP.
    # I don't actually use it in the code.
    # The fact that the BVP is solved will be noted, and the solution saved.
    # bvp_solved updates when setDeriv is called, but bvp_deriv only updates for update_params.
    def find_root_min(self, tol_root_finder = None, tol_bvp_matching = None):

        if tol_root_finder is None:
            tol_root_finder = self.tol_root_finder
        if tol_bvp_matching is None:
            tol_bvp_matching = self.tol_bvp_matching

        deriv_list = self.find_root_many(tol_root_finder, tol_bvp_matching)
        A_list = []
        for deriv_psi in deriv_list:
            self.setDeriv(deriv_psi)
            A_list.append(self.l_calc.amp_max())
        if len(A_list) > 0:
            i = np.argmin(A_list)
            self.setDeriv(deriv_list[i])
            self.bvp_solved = True
            self.bvp_deriv = deriv_list[i]

    # 7 -- Current for exact solution and new solution

    # This is the current of the linear solution to the BVP.
    def current_old(self):
        ajsq = np.square(np.real(self.aj0)) + np.square(np.imag(self.aj0))
        bjsq = np.square(np.real(self.bj0)) + np.square(np.imag(self.bj0))
        return ajsq - bjsq

    # This is the current we expect for the nonlinear solution.
    def current_bvp(self):
        if self.bvp_solved == False:
            if self.bvp_deriv is None:
                self.find_root_min()
            else:
                self.setDeriv(self.bvp_deriv)
        ajsq = np.square(np.real(self.aj)) + np.square(np.imag(self.aj))
        bjsq = np.square(np.real(self.bj)) + np.square(np.imag(self.bj))
        return ajsq - bjsq
    

    # These methods below are not set up to check if the solution given solves the bvp, so be careful

    def current_new(self):
        kn = pi / self.T_fast_mod
        mn = 2* pi /self.T_slow_mod
        ko = self.k
        mo = self.M

        ajsq = np.square(np.real(self.aj)) + np.square(np.imag(self.aj))
        bjsq = np.square(np.real(self.bj)) + np.square(np.imag(self.bj))

        # This is the term independent of position from the solution for current
        # given modeled nonlinear solution with the fast oscillations assumed
        # as sinusoidal.
        cur_new = (ajsq + bjsq)*(mn-mo)/ko + kn/ko * (ajsq - bjsq)

        return cur_new
    
    # I suspect that this is the actual current due to how the convergence appears to be structured.
    def current_alt(self):
        ajsq = np.square(np.real(self.aj)) + np.square(np.imag(self.aj))
        bjsq = np.square(np.real(self.bj)) + np.square(np.imag(self.bj))
        return ajsq - bjsq

    # This third one is defined at our starting point (or another point for passed values) instead of averaged.
    def current_calc(self, psi=None, psi_pr=None):
        if psi is None: psi = self.amp
        if psi_pr is None: psi_pr = self.psi0_deriv_0
        t1 = 1/ self.k / 2j
        t2 = -2j * self.B * self.R * phi0inv
        return t1*(np.conj(psi)*psi_pr - psi*np.conj(psi_pr) + t2*np.conj(psi)*psi)
    




