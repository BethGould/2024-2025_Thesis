# class grid_slow_osc 
#
# Author: Elizabeth Gould
# Date Last Edit: 09.03.2026
#
# This code builds a grid of solutions to the IVP, in order to calculate t_slow (= 2*pi / M), 
# the period of the slow oscillations. Here we can vary R, B, k, dk, mu, A, and Psi'(0) and 
# construct either a grid of values, or generate a set of random values. 
# 
# The Psi'(0) grid is spaced sinusoidally. The mu grid is spaced logrithmically. The other grids 
# are spaced linearly. Note that one of the results of this is that mu is taken as a exponent of 
# powers of 10 for the grid, while it is taken as a raw number for Monte Carlo runs.
# 
# grid_slow_osc is built off of grid_fast_osc, inhereting the routines for forming the grid, 
# as the structure is identical. 

# __init__(self, R, B, dk, mu, k = kFAu, amp=1., grid_size = 9, ang_lim = 0.1)
# makeDerivPoints(self) -- called by init
# makeGridPoints(self, a_min, a_max, mu_min, mu_max, num = 10, num_m = 10)
# gridFastOsc(self, n = 20, method = 'RK45', rtol = rtol, atol = atol)

# need to run init -> makeGridPoints -> gridSlowOsc

# Grid run code:
# > gridl = eelib.grid_slow_osc(R, B, dk, mu)
# > gridl.makeGridPoints(mu=mu_r, B=b_r, num = [n])
# > gridl.gridSlowOsc()

# Monte Carlo run code:
# > gridl = eelib.grid_slow_osc(R, B, dk, mu)
# > gridl.makeMCPoints(mu=mu_r, B=b_r, num = n)
# > gridl.mcSlowOsc()

# __init__(self, R=1.0, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.) --
#       Creates a loop object with the given parameters. Note that the passed values of parameters
#       are relevant only for those parameters which are not to be varied as varied parameters will 
#       be replaced during the calculations.

# __repr__(self)
# __str__(self)
#       Prints a description of the object. 

# save_solution -- Boolean set to false by default. If true, the full solutions to the IVP will be saved. 
#                  This is used as the solutions can be large, and are rarely needed.

# clear_calcs(self) -- Clears the data. 
# setIntegratorParameters(self, solve_mu_0 = None, n_sm = None, method = None, rtol = None, atol = None, R_max = None, n_lg = None)
# The default parameter values are: solve_mu_0 = False, n_sm = 20, method = 'RK45', rtol = rtol, atol = atol, R_max = 1.0, n_lg = 1000.
# n_sm counts how many oscillations of |Psi|^2 to count in order to estimate t / 2, (which is pi / k).
# n_lg counts how many points to sample of Psi when the fast oscillations reaches its maximum value. This will calculate the curve 
# for the slow oscillations. 
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

# gridSlowOsc(self) does the work of the calcuation if given a grid, with all of the time spent here.
# mcSlowOsc(self) does the work of the calculation if given a random assortment of points to run.
# runCalc(self) does the work of the calculation for either case.

# Results are saved in the following variables:
# self.slow_osc_k   -- Estimate for our wavenumber for the slow oscillations. (Found from fitting to a sine function.)
# self.slow_osc_a   -- Estimate our slow oscillation amplitude.
# self.slow_osc_th  -- Estimate our slow oscillation angle offset from a sine function.
# self.slow_osc_sol -- Saves our full solution. Note that this is saved as a list rather than an array like the other solutions.
# self.slow_osc_i   -- The i-th element of this list matches the full solution to the array indicies of the other estimates, 
#                      as given by the contained list.
#                      For i[j] = [n_mu, n_dk, n_b, n_r, n_a, n_k0, n_d, n_d] (where j and all ns are integers),
#                      sol[j] is the full solution for the estimated k[i], a[i], and th[i].
#                      This last variable is used only for the grid solution, as all variables have one index for the Monte 
#                      Carlo case.

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
from eelib.consts import kFAu, rtol, atol
from eelib.loop import loop
from eelib.grid_fast_osc import grid_fast_osc
from eelib.fitted_functions import fit_sin
import time
import gc

class grid_slow_osc(grid_fast_osc):

    # ------ INITIALIZATION --------------

    # I have parameters here set to remove a call to the integrator, hopefully saving some time.
    def __init__(self, R=1.0, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.):
        super().__init__(R,B,dk,mu,k,amp)

        self.setIntegratorParameters(solve_mu_0=False, R_max = 1.0, n_lg = 1000)
        self.save_solution = False # Tests with the solution saved show that it is too large.

    # Because we want to know what our object is.
    def __repr__(self):
        if self.is_grid:
            if self.calculated:
                str = "Grid object to measure slow oscillations:\n"
            else:
                str = "Uncalculated grid object to measure slow oscillations:\n"

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
                str = f"Monte Carlo object to measure slow oscillations:\n"
            else:
                str = f"Uncalculated Monte Carlo object to measure slow oscillations:\n"
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
            return "Empty object to measure slow oscillations:\nR is %s, B is %s, dk is %s, k is %s, mu is %s, A is %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # Because we want to know what our object is.
    def __str__(self):
        if self.is_grid:
            if self.calculated:
                str = "Grid object to measure slow oscillations:\n"
            else:
                str = "Uncalculated grid object to measure slow oscillations:\n"

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
                str = f"Monte carlo object to measure slow oscillations:\n"
            else:
                str = f"Uncalculated monte carlo object to measure slow oscillations:\n"
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
            return "Empty object to measure slow oscillations:\nR is %s, B is %s, dk is %s, k is %s, mu is %s, A is %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # ------ MODIFIED FUNCTIONS -------

    # This is changed, because the data is saved in a different manner.
    def clear_calcs(self):
        self.slow_osc_k   = None
        self.slow_osc_a   = None
        self.slow_osc_th  = None
        self.slow_osc_i   = None
        self.slow_osc_sol = None
        gc.collect()

    # R_max is not relevant for the fast oscillations, as only the first n fast oscillations are calculated for this 
    # estimate. So this works as the previous case, but now with an R_max parameter, to allow overcalculating, for 
    # better M estimates.
    def setIntegratorParameters(self, solve_mu_0 = None, n_sm = None, method = None, rtol = None, atol = None, R_max = None, n_lg = None):
        super().setIntegratorParameters(solve_mu_0, n_sm, method, rtol, atol)
        if n_lg is not None:
            self.n_lg = n_lg
        if R_max is not None:
            self.R_max = R_max

    # ------------- FIND VALUES ON GRID ---------------

    # Wrapper for the two following functions to run the code.
    def runCalc(self):
        if self.is_grid: 
            self.gridSlowOsc()
        elif self.is_mc: 
            self.mcSlowOsc()
        else: 
            print("Points to calculate have not yet been set.")

    # Remove the inherited function.
    def gridFastOsc(self):
        print("Incorrect type of object. Try gridSlowOsc instead.")

    # Perform the calculation on the grid.
    # This take time, but no longer requires parameters. It is separate from the parameters due to this requirement.
    def gridSlowOsc(self):

        # If old calculation already exists.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # If the grid is not set.
        if not self.is_grid:
            raise Exception("gridSlowOsc requires forming the grid first. Try calling makeGridPoints.") 

        # Integration parameters.
        pr = self.R_max # can be changed if more cycles needed
        plot_code = 1   # only plot er, saving time

        # Timing
        start_time = time.time()
        print('Begin grid build: ', time.time() - start_time)

        # Initialize grids
        num_dk = self.num_dk
        num_k0 = self.num_k0
        num_b  = self.num_b
        num_r  = self.num_r
        num_mu = self.num_mu
        num_a  = self.num_a
        num_d  = self.grid_size

        # saves solution for analysis
        sol_er_u = None

        # saved data
        slow_oscillation_wavenumber = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        slow_oscillation_amplitude  = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        slow_oscillation_theta      = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))

        slow_osc_sv_ind  = [] # indicies to match full solution to other saved data
        slow_osc_sv_data = [] # full solution

        # Indicies for each parameter to vary.
        ib  = 0
        ir  = 0
        im  = 0
        ik  = 0
        ik0 = 0
        ia  = 0
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
                                        self.l_calc.setDeriv(self.d0_grid[idr,idi])
                                        # Estimate k. This is no longer automatic, and it is required. 
                                        self.l_calc.find_fast_oscillations(self.n_sm, method = self.method, 
                                                                           rtol = self.rtol, atol = self.atol)
                                        self.l_calc.solve_ivp(n = self.n_lg, percent_range=pr, method = self.method,
                                                             rtol = self.rtol, atol = self.atol, solve = plot_code)

                                        # Retrieve our solution
                                        sol_er_u = self.l_calc.solu
                                        # Fit to a sinusoid.
                                        fit = fit_sin(sol_er_u)
                                        if fit is not None:
                                            slow_oscillation_wavenumber[im,ik,ib,ir,ia,ik0,idr,idi] = fit[1]
                                            slow_oscillation_amplitude[im,ik,ib,ir,ia,ik0,idr,idi] = fit[0]
                                            slow_oscillation_theta[im,ik,ib,ir,ia,ik0,idr,idi] = fit[2]

                                        if self.save_solution:
                                        # Save our full data as well.
                                            slow_osc_sv_ind.append([im,ik,ib,ir,ia,ik0,idr,idi])
                                            slow_osc_sv_data.append(sol_er_u.copy())

        # Transfer our local variables to object variables, saving our data.
        self.slow_osc_k  = slow_oscillation_wavenumber  # Estimate for our wavenumber for the slow oscillations (found from fitting a sin).
        self.slow_osc_a  = slow_oscillation_amplitude   # Maximum absolute value.
        self.slow_osc_th = slow_oscillation_theta
        if self.save_solution:
            self.slow_osc_i  = slow_osc_sv_ind              # Indicies for our spacings, due to the alternative storage mechanism.
            self.slow_osc_sol  = slow_osc_sv_data           # Save full solution.

        # Indicate that this function has been run.
        self.calculated = True

        # And end timing.
        print("Done grid build: ", time.time() - start_time)

    # Delete the inherited function.
    def mcFastOsc(self):
        print("Wrong type of object. Try mcSlowOsc.")

    # Now our run for randomly determined points.
    def mcSlowOsc(self):

        # Old calculation already exists.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # The grid is not set.
        if not self.is_mc:
            raise Exception("mcSlowOsc requires choosing the points first. Try calling makeMCPoints.") 

        # Integration parameters.
        pr = self.R_max # can be changed if more cycles needed
        plot_code = 1   # only plot er, saving time

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

        # Saves solution for analysis.
        sol_er_u = None
        sol_er_u_save = []

        slow_oscillation_wavenumber = np.zeros((num))
        slow_oscillation_theta = np.zeros((num))
        slow_oscillation_amplitude = np.zeros((num))

        # Solve for each rndomly determined point.
        print("Number of periods to calculate:", num)
        for ii in range(num):
            self.l_calc.update_params(R=self.val_table[ii,3], B=self.val_table[ii,2], dk=self.val_table[ii,1], 
                                      mu=self.val_table[ii,0], k = self.val_table[ii,5], amp=self.val_table[ii,4])
            # Calculate derivative from our parameters if it is not as given.
            self.l_calc.setDeriv(self.val_table[ii,6]+ 1.j *self.val_table[ii,7])
            # Estimate k. This is no longer automatic, and it is required.
            self.l_calc.find_fast_oscillations(self.n_sm, method = self.method, rtol = self.rtol, atol = self.atol)
            self.l_calc.solve_ivp(n = self.n_lg, percent_range=pr, method = self.method,
                                                             rtol = self.rtol, atol = self.atol, solve = plot_code)

            # Retrieve our solution.
            sol_er_u = self.l_calc.solu

            # Fit to a sinusoidal function.
            fit = fit_sin(sol_er_u)
            if fit is not None:
                slow_oscillation_wavenumber[ii] = fit[1]
                slow_oscillation_amplitude[ii] = fit[0]
                slow_oscillation_theta[ii] = fit[2]

            # And also save our full solution.
            if self.save_solution: sol_er_u_save.append(sol_er_u.copy())

        # Transfer our local variables to object variables, saving our data.
        self.slow_osc_k  = slow_oscillation_wavenumber  # Estimate for our wavenumber for the slow oscillations (found from fitting a sin).
        self.slow_osc_a  = slow_oscillation_amplitude   # Maximum absolute value.
        self.slow_osc_th = slow_oscillation_theta
        if self.save_solution: self.slow_osc_sol  = sol_er_u_save  # Save full solution.

        # Indicate that this function has been run.
        self.calculated = True

        # And end timing.
        print("Done grid build: ", time.time() - start_time)
