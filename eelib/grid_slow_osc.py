# grid_fast_osc class 
# Runs a grid to calculate t_fast.
# Fixed R, B, k, dk
# Grid on mu, amp, psi'(0)
# psi'(0) grid sinusoidal. mu grid logrithmic, amp grid linear.

# __init__(self, R, B, dk, mu, k = kFAu, amp=1., grid_size = 9, ang_lim = 0.1)
# makeDerivPoints(self) -- called by init
# makeGridPoints(self, a_min, a_max, mu_min, mu_max, num = 10, num_m = 10)
# gridFastOsc(self, n = 20, method = 'RK45', rtol = rtol, atol = atol)

# need to run init -> makeGridPoints -> gridFastOsc

# init forms the derivative grid and creates a loop with the given parameters

# makeGridPoints forms the grid for other parameters which vary
# num is the number of amp values
# num_mu is the number of mu values

# gridFastOsc does the work of calcuation
# time is spent here

# I should add a Monte Carlo variant.

import numpy as np
from scipy.optimize import curve_fit
from eelib.consts import pi, kFAu, rtol, atol
from eelib.loop import loop
from eelib.grid_fast_osc import grid_fast_osc
from eelib.deriv_functions import cos_fit, full_fit, sin_fit
import time
import gc


class grid_slow_osc(grid_fast_osc):

    # ------ INITIALIZATION --------------

    # I have parameters here set to remove a call to the integrator, hopefully saving some time.
    def __init__(self, R=1.0, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.):
        super().__init__(R,B,dk,mu,k,amp)

        self.setIntegratorParameters(solve_mu_0=False)

    # Because we want to know what our object is.
    def __repr__(self):
        if self.is_grid:
            if self.calculated:
                str = "Grid object to measure slow oscillations:\n"
            else:
                str = "Uncalculated grid object to measure slow oscillations:\n"

            # add our parameters for ......
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
                str = "Monte carlo object to measure slow oscillations"
            else:
                str = "Uncalculated monte carlo object to measure slow oscillations"
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

            # add our parameters for ......
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
                str = "Monte carlo object to measure slow oscillations"
            else:
                str = "Uncalculated monte carlo object to measure slow oscillations"
            return str
        else:
            return "Empty object to measure slow oscillations:\nR is %s, B is %s, dk is %s, k is %s, mu is %s, A is %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # ------ MODIFIED FUNCTIONS -------

    # because the data is saved in different places ....
    def clear_calcs(self):
        self.slow_osc_k   = None
        self.slow_osc_a   = None
        self.slow_osc_th  = None
        self.slow_osc_i   = None
        self.slow_osc_sol = None

    def setIntegratorParameters(self, solve_mu_0 = False, n = 20, method = 'RK45', rtol = rtol, atol = atol, R_max = 1.0):
        super().setIntegratorParameters(solve_mu_0, n, method, rtol, atol)
        self.R_max = R_max

    # ------------- FIND VALUES ON GRID ---------------

    # def calculate_fast_t(self):
    # if self.is_grid: gridFastOsc()
    # if self.is_mc: mcFastOsc()

    def gridFastOsc(self):
        print("Incorrect type of object. Try gridSlowOsc instead.")

    # Perform the calculation on the grid.
    # This take time, but no longer requires parameters. It is separate from the parameters due to this.
    # I need to add functionality which uses self.is_grid, ...
    def gridSlowOsc(self):

        # Old calculation already exists.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # The grid is not set.
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

        ib  = 0
        ir  = 0
        im  = 0
        ik  = 0
        ik0 = 0
        ia  = 0
        idr = 0
        idi = 0

        # solve for each point on the grid
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
                                # And for the derivative
                                for idr in range(num_d):
                                    for idi in range(num_d):
                                        # calculate derivative from our parameters if it is not as given
                                        self.l_calc.setDeriv(self.d0_grid[idr,idi], n = self.n, method = self.method,
                                                             rtol = self.rtol, atol = self.atol)
                                        self.l_calc.solve_ivp(n = self.n, percent_range=pr, method = self.method,
                                                             rtol = self.rtol, atol = self.atol, solve = plot_code)

                                        sol_er_u = self.l_calc.solu
                                        fit = fit_sin(sol_er_u)
                                        if fit is not None:
                                            slow_oscillation_wavenumber[im,ik,ib,ir,ia,ik0,idr,idi] = fit[1]
                                            slow_oscillation_amplitude[im,ik,ib,ir,ia,ik0,idr,idi] = fit[0]
                                            slow_oscillation_theta[im,ik,ib,ir,ia,ik0,idr,idi] = fit[2]

                                        slow_osc_sv_ind.append([im,ik,ib,ir,ia,ik0,idr,idi])
                                        slow_osc_sv_data.append(sol_er_u.copy())

        self.slow_osc_k  = slow_oscillation_wavenumber  # estimate for our wavenumber for the slow oscillations (found from fitting a sin)
        self.slow_osc_a  = slow_oscillation_amplitude   # maximum absolute value
        self.slow_osc_th = slow_oscillation_theta
        self.slow_osc_i  = slow_osc_sv_ind             # indicies for our spacings, due to the alternative storage mechanism
        self.slow_osc_sol  = slow_osc_sv_data          # save full solution

        self.calculated = True

        print("Done grid build: ", time.time() - start_time)


    def mcFastOsc(self):
        print("Wrong type of object. Try mcSlowOsc.")

    def mcSlowOsc(self):

        # Old calculation already exists.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # The grid is not set.
        if not self.is_mc:
            raise Exception("mcSlowOsc requires choosing the points first. Try calling makeMCPoints.") 

        # Integration parameters.
        pr = 1.0 # can be changed if more cycles needed
        plot_code = 1 # only plot er, saving time

        # Timing
        start_time = time.time()
        print('Begin grid build: ', time.time() - start_time)

        # Initialize grids
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

        # saves solution for analysis
        sol_er_u = None

        slow_oscillation_period = np.zeros((num))
        slow_oscillation_amplitude = np.zeros((num))
        #fast_oscillation_period_0 = np.zeros((num))

        # solve for each point on the grid
        print("Number of periods to calculate:", num)
        for ii in range(num):
            self.l_calc.update_params(R=self.val_table[ii,3], B=self.val_table[ii,2], dk=self.val_table[ii,1], 
                                      mu=self.val_table[ii,0], k = self.val_table[ii,5], amp=self.val_table[ii,4])
            # calculate derivative from our parameters if it is not as given
            self.l_calc.setDeriv(self.val_table[ii,6]+ 1.j *self.val_table[ii,7], n = self.n, method = self.method,
                                                                        rtol = self.rtol, atol = self.atol)
            
            self.l_calc.solve_ivp(n = self.n, percent_range=pr, method = self.method,
                                                             rtol = self.rtol, atol = self.atol, solve = plot_code)
        
            sol_er_u = self.l_calc.solu

            slow_oscillation_period[ii] = find_period(sol_er_u)
            slow_oscillation_amplitude[ii] = find_amp(sol_er_u) 
            #fast_oscillation_period_0[ii] = self.l_calc.T_fast_0_calc * 2

        self.fast_osc_t = slow_oscillation_period
        self.fast_osc_a = slow_oscillation_amplitude
        #self.fast_osc_t_0 = fast_oscillation_period_0    # because I want to know the error.

        self.calculated = True

        print("Done grid build: ", time.time() - start_time)


# ---------- PERIOD AND AMPLITUDE CALCULATIONS -------------

# Ok. Here is code to fit a sin function
# I may need to fit to only half a period in order to ensure one-to-one behavior
# all of our parameters must be set close to the expected value for this to work well

def fit_sin(sol):
    # estimate our parameters (amp, M, theta)
    # because we need to start close to the expected value to get a correct fit
    amp, TT, rt_st = find_fit_params(sol)

    # extract our x and psi values from our solution
    xdata = sol['t']
    ydata = np.real(sol['y'][0])

    # rescale our x_0 to be an angle shift
    if rt_st is None:
        return None
    
    MM = 2 * pi / TT
    theta = - MM * xdata[rt_st]

    # prevent our function fitter from deviating too far from our expected values
    # I need to choose resonable numbers here. 
    amp_tol = 0.1
    M_tol = 0.1
    th_tol = 0.1 * pi

    # min, max, and start values for our fonction fitter, in the correct form
    min_arr = np.array([(1.0 - amp_tol) * amp, MM - M_tol * MM , theta - th_tol])
    max_arr = np.array([(1.0 + 2 * amp_tol) * amp, MM + M_tol * MM, theta + th_tol])
    start_arr = np.array([amp, MM, theta])

    # and call scipy
    popt, pcov = curve_fit(sin_fit, xdata, ydata, p0=start_arr, bounds=(min_arr, max_arr))
    return popt


# These assume nice behavior. So far, it works well, but I can think of cases when it won't.
# They will estimate the fitting sin function 

# -- amp estimate --
# given a solution, estimate the amplitude as the furthest distance in the solution from 0
# the actual amplitude should be slightly more than this
def find_amp(sol):
    arr = np.real(sol['y_events'][1][:, 0])
    return np.max(np.abs(arr))

# used for both x_0 estimation and period estimation
# y_points = y1_list_0 = np.real(sol['y'][0])
# given the y points, find the indicies of the roots
def find_root_points(y_points):
    y_mult = np.array(y_points[:-1])*np.array(y_points[1:])
    zero_index = np.nonzero(y_mult == 0)
    root_index = np.nonzero(y_mult < 0)
    full_index = list(root_index[0]) + list(zero_index[0])
    full_index.sort()
    return full_index

# -- x_0 estimate --
# here I want to find the first two x values where the function changes sign 
# and return the one where the sign goes from negative to positive
# since the sin function starts at 0 and increases from there
def find_root_start(sol):
    y1_list = np.real(sol['y'][0])
    root_ls = find_root_points(y1_list)
    if len(root_ls) < 2:
        rt_start = None
    elif y1_list[root_ls[0]+1] > 0: 
        rt_start = root_ls[0]
    else: 
        rt_start = root_ls[1]
    return rt_start

# -- period estimate --
# the function goes through 0 twice in a period, 0->0->0 for sin
def find_period(sol):
    t_list = sol['t']
    y1_list = np.real(sol['y'][0])
    root_list = find_root_points(y1_list)
    t_root_list = []
    for i in root_list: 
        t_root_list.append(t_list[i])
    if len(t_root_list) == 0:
        return None
    t_diff = np.array(t_root_list[1:])- np.array(t_root_list[:-1])
    t_dif_ave = np.average(t_diff)
    return t_dif_ave * 2

# call and return the result of all three estimates
def find_fit_params(sol):
    amp = find_amp(sol)
    rt_start = find_root_start(sol)
    MM = find_period(sol)
    return amp, MM, rt_start        
        
