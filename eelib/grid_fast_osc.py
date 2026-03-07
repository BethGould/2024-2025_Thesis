# class grid_fast_osc 
#
# Author: Elizabeth Gould
# Date Last Edit: 03.03.2026
#
# This code builds a grid of solutions to the IVP, in order to calculate t_fast (= 2*pi / k), 
# the period of the fast oscillations. Here we can vary R, B, k, dk, mu, A, and Psi'(0) and 
# construct either a grid of values, or generate a set of random values. 
#
# The Psi'(0) grid is spaced sinusoidally. The mu grid is spaced logrithmically. The other grids 
# are spaced linearly. Note that one of the results of this is that mu is taken as a exponent of 
# powers of 10 for the grid, while it is taken as a raw number for Monte Carlo runs.

# Grid run code:
# > gridl = eelib.grid_fast_osc(R, B, dk, mu)
# > gridl.makeGridPoints(mu=mu_r, B=b_r, num = [n])
# > gridl.gridFastOsc()

# Monte Carlo run code:
# > gridl = eelib.grid_fast_osc(R, B, dk, mu)
# > gridl.makeMCPoints(mu=mu_r, B=b_r, num = n)
# > gridl.mcFastOsc()

# __init__(self, R=1.0, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.) --
#       Creates a loop object with the given parameters. Note that the passed values of parameters
#       are relevant only for those parameters which are not to be varied as varied parameters will 
#       be replaced during the calculations.

# __repr__(self)
# __str__(self)
#       Prints a description of the object. 

# clear_calcs(self) -- Clears the data. 

# setIntegratorParameters(self, solve_mu_0 = None, n_sm = None, method = None, rtol = None, atol = None)
# The default parameter values are: solve_mu_0 = True, n_sm = 20, method = 'RK45', rtol = rtol, atol = atol.
# n_sm counts how many oscillations of |Psi|^2 to count in order to estimate t / 2, (which is pi / k).
# If solve_mu_0 = True, the code will estimate the fast oscillations for the case without ee interaction as well.
# Parameters not given will not be changed.

# makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), R = (-1.0, -1.0), 
#                       A = (-1.0, -1.0), k0 = (-1.0, -1.0), ang_lim = 0.1, num = [10])
# makeMCPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), R = (-1.0, -1.0), 
#                       A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = 1000)

# These define the various values of parameters for which to analyze the IVP. If a set of points is not included,
# it will not be varied, instead using the value of the parameter passed in initialization. Otherwise, the values
# will be varied from the minimum value given to the maximum value given. 
#   !!! Note -- mu is varied based on exponents of 10 for the grid points, but as a pure number for Monte Carlo 
#               analysis. I know this is an issue, but it how things are written.
#   num is a positive integer indicating the number of points for the Monte Carlo case.
#   For a grid, num must be a list of positive integers. A list size of one will set all dimensions of the grid to
# this number, for num ^ n points. You can send two numbers to provide separate lengths for derivatives and other
# numbers, or more to provide a value for every parameter separately. (The real and imaginary derivatives must be 
# equal, and they will never take two parameters, only one.)

# gridFastOsc(self) does the work of the calcuation if given a grid, with all of the time spent here.
# mcFastOsc(self) does the work of the calculation if given a random assortment of points to run.
# runCalc(self) does the work of the calculation for either case.

# Results are saved in the following variables:
# self.fast_osc_t = fast_oscillation_period
# self.fast_osc_a = fast_oscillation_amplitude
# self.fast_osc_t_0 = fast_oscillation_period calculated without ee interaction, for error-checking

# For a grid run, the saved arrays are indicated as follows:  [n_mu, n_dk, n_b, n_r, n_a, n_k0, n_dr, n_di]
# The order might be n_di, n_dr. I know I invert real and imaginary indicies at some point, but I think this 
# one is fine. I am not sure, however.

# Parameters are saved in the following variables.
# - Monte Carlo Table:
# self.val_table    -- The table of random values of parameters for the Monte Carlo variant.
# - Grid Tables:
# magnetic_field_strength  = self.mfs
# electron_wavenumber      = self.ew
# electron_wavenumber_diff = self.ewd
# nonlinearity_strength    = self.nls
# starting_amplitude       = self.amp
# ring_radius              = self.rr
# d0_grid[idr,idi]
# starting_derivative_real = self.dr
# starting_derivative_imag = self.di

import numpy as np
from eelib.consts import pi, kFAu, rtol, atol
from eelib.loop import loop
import time
import gc

class grid_fast_osc:

    # ------ INITIALIZATION --------------

    # R, B, dk, mu, k, and amp must be given if they are fixed.
    # k defaults to gold, amp defaults to 1.
    def __init__(self, R=0.5, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.):
        self.l_calc = loop(R, B, dk, mu, k, amp)

        self.R = R
        self.B = B
        self.dk = dk
        self.k = k
        self.mu = mu
        self.a = amp

        self.is_grid = False
        self.is_mc = False
        self.calculated = False

        self.setIntegratorParameters(solve_mu_0 = True, n_sm = 20, method = 'RK45', rtol = rtol, atol = atol)

    # There are many parameters for the integrator, which I tend to just set to the default and ignore.
    # Here they can be set once, and used every time.
    def setIntegratorParameters(self, solve_mu_0 = None, n_sm = None, method = None, rtol = None, atol = None):
        if solve_mu_0 is not None:
            self.l_calc.solve_mu_0 = solve_mu_0
        if n_sm is not None:
            self.n_sm = n_sm
        if method is not None:
            self.method = method
        if rtol is not None:
            self.rtol = rtol
        if atol is not None:
            self.atol = atol

    # ------------ STRING OUTPUT ---------------
    # Because we want to know what our object is.
    def __repr__(self):
        if self.is_grid:
            if self.calculated:
                str = "Grid object to measure fast oscillations:\n"
            else:
                str = "Uncalculated grid object to measure fast oscillations:\n"

            # Indicate our parameters for our grid.
            if self.num_mu > 1:
                str = str + f"mu has {self.num_mu} points from {self.nls[0]} to {self.nls[-1]}.\n"
            else:
                str = str + f"mu is: {self.mu}\n"
            if self.num_dk > 1:
                str = str + f"dk has {self.num_dk} points from {self.ewd[0]} to {self.ewd[-1]}.\n"
            else:
                str = str + f"dk is: {self.dk}\n"
            if self.num_b > 1:
                str = str + f"B has {self.num_b} points from {self.mfs[0]} to {self.mfs[-1]}.\n"
            else:
                str = str + f"B is: {self.B}\n"
            if self.num_r > 1:
                str = str + f"R has {self.num_r} points from {self.rr[0]} to {self.rr[-1]}.\n"
            else:
                str = str + f"R is: {self.R}\n"
            if self.num_a > 1:
                str = str + f"A has {self.num_a} points from {self.amp[0]} to {self.amp[-1]}.\n"
            else:
                str = str + f"A is: {self.amp}\n"
            if self.num_k0 > 1:
                str = str + f"k0 has {self.num_k0} points from {self.ew[0]} to {self.ew[-1]}.\n"
            else:
                str = str + f"k0 is: {self.k}\n"
            str = str + f"Grid size: {self.grid_size}\n"
            str = str + f"Total number of parameters: {self.num_mu*self.num_dk*self.num_k0*self.num_b*self.num_a*self.num_r*self.grid_size*self.grid_size}"
            return str
        elif self.is_mc:
            if self.calculated:
                str = f"Monte Carlo object to measure fast oscillations:\n"
            else:
                str = f"Uncalculated Monte Carlo object to measure fast oscillations:\n"
            # The following output is not eligant.
            str = str + f"mu has points from {np.min(self.val_table[:,0])} to {np.max(self.val_table[:,0])}.\n"
            str = str + f"dk has points from {np.min(self.val_table[:,1])} to {np.max(self.val_table[:,1])}.\n"
            str = str + f"B has points from {np.min(self.val_table[:,2])} to {np.max(self.val_table[:,2])}.\n"
            str = str + f"R has points from {np.min(self.val_table[:,3])} to {np.max(self.val_table[:,3])}.\n"
            str = str + f"A has points from {np.min(self.val_table[:,4])} to {np.max(self.val_table[:,4])}.\n"
            str = str + f"k0 has points from {np.min(self.val_table[:,5])} to {np.max(self.val_table[:,5])}.\n"
            str = str + f"Number of points: {self.val_table.shape[0]}"
            return str
        else:
            return "Empty object to measure fast oscillations:\nR is %s, B is %s, dk is %s, k is %s, mu is %s, A is %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # Because we want to know what our object is.
    def __str__(self):
        if self.is_grid:
            if self.calculated:
                str = "Grid object to measure fast oscillations:\n"
            else:
                str = "Uncalculated grid object to measure fast oscillations:\n"

            #  Indicate our parameters for our grid.
            if self.num_mu > 1:
                str = str + f"mu has {self.num_mu} points from {self.nls[0]} to {self.nls[-1]}.\n"
            else:
                str = str + f"mu is: {self.mu}\n"
            if self.num_dk > 1:
                str = str + f"dk has {self.num_dk} points from {self.ewd[0]} to {self.ewd[-1]}.\n"
            else:
                str = str + f"dk is: {self.dk}\n"
            if self.num_b > 1:
                str = str + f"B has {self.num_b} points from {self.mfs[0]} to {self.mfs[-1]}.\n"
            else:
                str = str + f"B is: {self.B}\n"
            if self.num_r > 1:
                str = str + f"R has {self.num_r} points from {self.rr[0]} to {self.rr[-1]}.\n"
            else:
                str = str + f"R is: {self.R}\n"
            if self.num_a > 1:
                str = str + f"A has {self.num_a} points from {self.amp[0]} to {self.amp[-1]}.\n"
            else:
                str = str + f"A is: {self.amp}\n"
            if self.num_k0 > 1:
                str = str + f"k0 has {self.num_k0} points from {self.ew[0]} to {self.ew[-1]}.\n"
            else:
                str = str + f"k0 is: {self.k}\n"
            str = str + f"Grid size: {self.grid_size}\n"
            str = str + f"Total number of parameters: {self.num_mu*self.num_dk*self.num_k0*self.num_b*self.num_a*self.num_r*self.grid_size*self.grid_size}"
            return str
        elif self.is_mc:
            if self.calculated:
                str = f"Monte carlo object to measure fast oscillations:\n"
            else:
                str = f"Uncalculated monte carlo object to measure fast oscillations:\n"
            # The following output is not eligant.
            str = str + f"mu has points from {np.min(self.val_table[:,0])} to {np.max(self.val_table[:,0])}.\n"
            str = str + f"dk has points from {np.min(self.val_table[:,1])} to {np.max(self.val_table[:,1])}.\n"
            str = str + f"B has points from {np.min(self.val_table[:,2])} to {np.max(self.val_table[:,2])}.\n"
            str = str + f"R has points from {np.min(self.val_table[:,3])} to {np.max(self.val_table[:,3])}.\n"
            str = str + f"A has points from {np.min(self.val_table[:,4])} to {np.max(self.val_table[:,4])}.\n"
            str = str + f"k0 has points from {np.min(self.val_table[:,5])} to {np.max(self.val_table[:,5])}.\n"
            str = str + f"Number of points: {self.val_table.shape[0]}"
            return str
        else:
            return "Empty object to measure fast oscillations:\nR is %s, B is %s, dk is %s, k is %s, mu is %s, A is %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # ------------ CHOOSE POINTS (GRID) -----------------

    # def makeGridPoints(self, m_min, m_max, k_min, k_max, a_min, a_max, mu_min, mu_max, num = 10):
    # In order -- mu, dk, B, R, M, A, k0
    # ang_lim is used for avoiding zero values where it may be an issue.
    # num length -- 1, 2 (grid vs non-grid), or n = defined vars + 1 for grid
    def makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), R = (-1.0, -1.0), 
                       A = (-1.0, -1.0), k0 = (-1.0, -1.0), ang_lim = 0.1, num = [10]):
        
        # Determines with which parameters to build the grid.
        # For dk, I need something else to indicate that it is not needed, as negative values in theory could be useful.
        # However, I assume dk goes from 0 to 1.
        # There is a wider range of error-checking which can be performed at this step, limiting our ranges.
        # I don't know why I didn't just use None.
        var_set = [False,False,False,False,False,False]
        if mu[0] <  0.0: var_set[0] = True
        if dk[0] > -0.1: var_set[1] = True
        if B[0]  > -0.1: var_set[2] = True
        if R[0]  > -0.1: var_set[3] = True
        if A[0]  > -0.1: var_set[4] = True
        if k0[0] > -0.1: var_set[5] = True

        self.var_set = var_set # so I can use this later

        # Put the number of grid points to use into our object.
        self.set_nums(var_set, num)

        # non-linear strength / mu
        if var_set[0]:
            self.nls = np.logspace(mu[0], mu[1], num=self.num_mu)
        else: 
            self.nls = np.array([self.mu])

        # electron wavenumber / k
        if var_set[1]:
            self.ewd = np.linspace(dk[0], dk[1], num=self.num_dk)
        else:
            self.ewd = np.array([self.dk])
        if var_set[5]:
            self.ew = np.linspace(k0[0], k0[1], num=self.num_k0)
        else:
            self.ew = np.array([self.k])

        # oscillations due to magnetic field, magnetic field strenth and ring radius
        if var_set[2]:
            self.mfs = np.linspace(B[0], B[1], num=self.num_b)
        else:
            self.mfs = np.array([self.B])
        if var_set[3]:
            self.rr = np.linspace(R[0], R[1], num=self.num_r)
        else:
            self.rr = np.array([self.R])

        # Initial values for Psi, Psi'
        if var_set[4]:
            self.amp = np.linspace(A[0], A[1], num=self.num_a)
        else:
            self.amp = np.array([self.a])

        self.dlim = self.l_calc.psi_prime_0_max()
        self.ang_lim = ang_lim
        self.makeDerivPoints()

        if self.calculated: self.clear_calcs()

        self.is_grid = True
        self.is_mc = False
        self.calculated = False

        gc.collect()

    # This is separated for clarity of previous code as this is long and repetitive.
    # It is not to be run independently.
    def set_nums(self, var_set, num):

        self.num_mu = 1
        self.num_a = 1
        self.num_b = 1
        self.num_r = 1
        self.num_dk = 1
        self.num_k0 = 1

        if len(num) < 1: 
            raise ValueError("Numbers of points have not been indicated.")
        elif len(num) == 1:
            self.grid_size = num[0]
            self.num = num[0]
            if var_set[0]: self.num_mu = num[0]
            if var_set[4]: self.num_a  = num[0]
            if var_set[2]: self.num_b  = num[0]
            if var_set[3]: self.num_r  = num[0]
            if var_set[1]: self.num_dk = num[0]
            if var_set[5]: self.num_k0 = num[0]
        # separating derivitive from other parameters
        elif len(num) == 2:
            if max(var_set) == False: 
                raise ValueError("Expecting only one value for num, as only the grid will be built.")
            self.grid_size = num[1]
            self.num = num[0]
            if var_set[0]: self.num_mu = num[0]
            if var_set[4]: self.num_a  = num[0]
            if var_set[2]: self.num_b  = num[0]
            if var_set[3]: self.num_r  = num[0]
            if var_set[1]: self.num_dk = num[0]
            if var_set[5]: self.num_k0 = num[0]
        # separating derivitive from other parameters
        else:
            if max(var_set) == False: 
                raise ValueError("Expecting only one value for num, as only the grid will be built.")
            i = 0
            j = 0
            ln = len(num)
            if var_set[i]: 
                self.num_mu = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                self.num_dk = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                self.num_b = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                if j == ln: 
                    raise ValueError(f"Not enough values passed for num. Stopped at parameter index {i} (R) and number array {j}.")
                self.num_r = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                if j == ln: 
                    raise ValueError(f"Not enough values passed for num. Stopped at parameter index {i} (A0) and number array {j}.")
                self.num_a = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                if j == ln: 
                    raise ValueError(f"Not enough values passed for num. Stopped at parameter index {i} (k0) and number array {j}.")
                self.num_k0 = num[j]
                j += 1
            i += 1

            if j == ln: 
                raise ValueError(f"Not enough values passed for num. Stopped at parameter index {i} (derivitive) and number array {j}.")
            self.grid_size = num[j]
            j += 1
            if j != ln:
                raise ValueError(f"Too many values passed for num. j = {j}, i = {i}, ln = {ln}")

    # Now to make the derivative grid.
    # Inputs:
    #       self.ang_lim   -- should remain constant, prevents zero values; 
    #       self.dlim      -- should change and be recorded in an array; scale of derivative
    #       self.grid_size -- must remain constant, set at start
    def makeDerivPoints(self):
        # From the derivative grid for t = 0.
        # Note that this uses the given values to choose the ideal points.
        # The points should instead be based off of the other values.
        x = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        x = np.cos(x) * self.dlim 
        y = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        y = np.cos(y) * self.dlim
        xx, yy = np.meshgrid(x, y)
        d0_grid = xx + 1j * yy

        d0_grid = d0_grid.T

        self.d0_grid = d0_grid

    # Clear memory
    def clear_calcs(self):
        self.fast_osc_t   = None
        self.fast_osc_t_0 = None
        self.fast_osc_a   = None
        gc.collect()


    # ------------ CHOOSE POINTS (MONTE CARLO) -----------------

    # Monte Carlo points take ranges of parameters like the grid, but since they are random,
    # it will output an array of random points, rather than building a grid. num is the full 
    # number of points, not for each parameter like is the case for the grid.
    def makeMCPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), R = (-1.0, -1.0), 
                       A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = 1000):
        
        # Determines with which parameters to build the grid.
        # For dk, I need something else to indicate that it is not needed, as negative values in theory could be useful
        # however, I assume dk goes from 0 to 1.
        # There is a wider range of error-checking which can be performed at this step, limiting our ranges.
        var_set = [False,False,False,False,False,False]
        if mu[0] > -0.1: var_set[0] = True
        if dk[0] > -0.1: var_set[1] = True
        if B[0]  > -0.1: var_set[2] = True
        if R[0]  > -0.1: var_set[3] = True
        if A[0]  > -0.1: var_set[4] = True
        if k0[0] > -0.1: var_set[5] = True

        self.var_set = var_set # so I can use this later

        # Create a table of values. 
        self.num = num
        self.val_table = np.zeros((num, 8))

        rng = np.random.default_rng()

        # non-linear strength / mu
        if var_set[0]:
            for ii in range(num):
                self.val_table[ii,0] = mu[0] + rng.random()*(mu[1]-mu[0])
        else: 
            self.val_table[:,0] = self.mu

        # electron wavenumber / k
        if var_set[1]:
            for ii in range(num):
                self.val_table[ii,1] = dk[0] + rng.random()*(dk[1]-dk[0])
        else: 
            self.val_table[:,1] = self.dk
        if var_set[5]:
            for ii in range(num):
                self.val_table[ii,5] = k0[0] + rng.random()*(k0[1]-k0[0])
        else:
            self.val_table[:,5] = self.k

        # oscillations due to magnetic field, magnetic field strength and ring radius
        if var_set[2]:
            for ii in range(num):
                self.val_table[ii,2] = B[0] + rng.random()*(B[1]-B[0])
        else:
            self.val_table[:,2] = self.B
        if var_set[3]:
            for ii in range(num):
                self.val_table[ii,3] = R[0] + rng.random()*(R[1]-R[0])
        else:
            self.val_table[:,3] = self.R

        # Initial values for Psi, Psi'
        if var_set[4]:
            for ii in range(num):
                self.val_table[ii,4] = A[0] + rng.random()*(A[1]-A[0])
        else:
            self.val_table[:,4] = self.a

        self.dlim = self.l_calc.psi_prime_0_max()

        for ii in range(num):
            self.val_table[ii,6] = -self.dlim + rng.random()*2*self.dlim
            self.val_table[ii,7] = -self.dlim + rng.random()*2*self.dlim

        # Free up memory after clearing the old data, as it is no longer valid
        if self.calculated: self.clear_calcs()

        self.is_grid = False
        self.is_mc = True
        self.calculated = False

        gc.collect()


    # ------------- FIND VALUES ON GRID ---------------

    # Wrapper for the two following functions to run the code.
    def runCalc(self):
        if self.is_grid: 
            self.gridFastOsc()
        elif self.is_mc: 
            self.mcFastOsc()
        else: 
            print("Points to calculate have not yet been set.")

    # Perform the calculation on the grid.
    # This take time, but no longer requires parameters. It is separate from the parameters due to this requirement.
    def gridFastOsc(self):

        # Old calculation already exists.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # The grid is not set.
        if not self.is_grid:
            raise Exception("gridFastOsc requires forming the grid first. Try calling makeGridPoints.") 

        # Timing
        start_time = time.time()
        print('Begin grid build: ', time.time() - start_time)

        # Initialize grids
        num_dk = self.num_dk
        num_k0 = self.num_k0
        num_b = self.num_b
        num_r = self.num_r
        num_mu = self.num_mu
        num_a = self.num_a
        num_d = self.grid_size

        # magnetic_field_strength  = self.mfs
        # electron_wavenumber      = self.ew
        # electron_wavenumber_diff = self.ewd
        # nonlinearity_strength    = self.nls
        # starting_amplitude       = self.amp
        # ring_radius              = self.rr
        # d0_grid[idr,idi]
        # starting_derivative_real = self.dr
        # starting_derivative_imag = self.di

        # saved data
        fast_oscillation_period = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        fast_oscillation_amplitude = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        fast_oscillation_period_0 = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))

        # Indicies for each parameter to vary.
        ib = 0
        ir = 0
        im = 0
        ik = 0
        ik0 = 0
        ia = 0
        idr = 0
        idi = 0

        # Solve for each point on the grid.
        # The 8 levels are only a problem if they are all used.
        print("Number of periods to calculate:", num_dk*num_k0*num_b*num_r*num_mu*num_a*num_d*num_d)
        for im in range(num_mu):
            for ik in range(num_dk):
                print(f"mu: {im}, dk: {ik}, Time: {time.time() - start_time}")
                for ib in range(num_b):
                    for ir in range(num_r):    
                        for ia in range(num_a):
                            for ik0 in range(num_k0):
                                self.l_calc.update_params(R=self.rr[ir], B=self.mfs[ib], dk=self.ewd[ik], 
                                                  mu=self.nls[im], k = self.ew[ik0], amp=self.amp[ia])
                                # And for the derivative.
                                for idr in range(num_d):
                                    for idi in range(num_d):
                                        # Calculate the derivative from our parameters, if it is not as given.
                                        self.l_calc.setDeriv(self.d0_grid[idr,idi], n = self.n_sm, method = self.method,
                                                             rtol = self.rtol, atol = self.atol)
                                        # Estimate k. This is no longer automatic, and it is required. 
                                        self.l_calc.find_fast_oscillations(self.n_sm, method = self.method, 
                                                                           rtol = self.rtol, atol = self.atol)
                                        #print(self.l_calc.T_fast * 2, self.l_calc.A_max)
                                        fast_oscillation_period[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.T_fast * 2
                                        fast_oscillation_amplitude[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.A_max  
                                        fast_oscillation_period_0[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.T_fast_0_calc * 2

        # Transfer our local variables to object variables, saving our data.
        self.fast_osc_t = fast_oscillation_period
        self.fast_osc_a = fast_oscillation_amplitude
        self.fast_osc_t_0 = fast_oscillation_period_0    # because I want to know the error.

        # Indicate that this function has been run.
        self.calculated = True

        # And end timing.
        print("Done grid build: ", time.time() - start_time)

    # Now our run for randomly determined points.
    def mcFastOsc(self):

        # Old calculation already exists.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # The grid is not set.
        if not self.is_mc:
            raise Exception("mcFastOsc requires choosing the points first. Try calling makeMCPoints.") 

        # Timing
        start_time = time.time()
        print('Begin grid build: ', time.time() - start_time)

        # Retrieve number of points.
        num = self.num

        # magnetic_field_strength  = self.mfs
        # electron_wavenumber      = self.ew
        # electron_wavenumber_diff = self.ewd
        # nonlinearity_strength    = self.nls
        # starting_amplitude       = self.amp
        # ring_radius              = self.rr
        # d0_grid[idr,idi]
        # starting_derivative_real = self.dr
        # starting_derivative_imag = self.di

        # self.val_table

        # results
        fast_oscillation_period = np.zeros((num))
        fast_oscillation_amplitude = np.zeros((num))
        fast_oscillation_period_0 = np.zeros((num))

        # Solve for each randomly determined point.
        print("Number of periods to calculate:", num)
        for ii in range(num):
            self.l_calc.update_params(R=self.val_table[ii,3], B=self.val_table[ii,2], dk=self.val_table[ii,1], 
                                      mu=self.val_table[ii,0], k = self.val_table[ii,5], amp=self.val_table[ii,4])
            # calculate derivative from our parameters if it is not as given
            self.l_calc.setDeriv(self.val_table[ii,6]+ 1.j *self.val_table[ii,7], n = self.n_sm, method = self.method,
                                                                        rtol = self.rtol, atol = self.atol)
            # Estimate k. This is no longer automatic, and it is required.
            self.l_calc.find_fast_oscillations(self.n_sm, method = self.method, rtol = self.rtol, atol = self.atol)

            # Save our results.
            fast_oscillation_period[ii] = self.l_calc.T_fast * 2
            fast_oscillation_amplitude[ii] = self.l_calc.A_max  
            fast_oscillation_period_0[ii] = self.l_calc.T_fast_0_calc * 2

        # Transfer our local variables to object variables, saving our data.
        self.fast_osc_t = fast_oscillation_period
        self.fast_osc_a = fast_oscillation_amplitude
        self.fast_osc_t_0 = fast_oscillation_period_0    # because I want to know the error.

        # Indicate that this function has been run.
        self.calculated = True

        # And end timing.
        print("Done grid build: ", time.time() - start_time)