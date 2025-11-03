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
from eelib.consts import pi, kFAu, rtol, atol
from eelib.loop import loop
from eelib.grid_fast_osc import grid_fast_osc
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
        self.slow_osc_t   = None
        self.slow_osc_a   = None
        self.slow_osc_i   = None
        self.slow_osc_m   = None
        self.slow_osc_s   = None

    def setIntegratorParameters(self, solve_mu_0 = False, n = 20, method = 'RK45', rtol = rtol, atol = atol, R_max = 1.0):
        super().setIntegratorParameters(solve_mu_0, n, method, rtol, atol)
        self.R_max = R_max

    # ---------- PERIOD AND AMPLITUDE CALCULATIONS -------------


    # Note that this calculation has high error. However, it is likely not used.
    def find_amp(self, sol):
        arr = np.real(sol['y_events'][1][:, 0])
        return np.max(np.abs(arr))

    # nn is the number of divisions within which to search for a maximum
    def find_period(self, sol, nn=60):

        diff_arr = self.find_gaps(sol, nn)
        
        # because I need an average over an even number of cycles
        if len(diff_arr)%2 == 0:
            return np.average(diff_arr), np.std(diff_arr)
        else:
            return np.average([np.average(diff_arr[:-1]), np.average(diff_arr[1:])]), np.std(diff_arr)

    def find_gaps(self, sol, nn=60):

        # In order to shorten the chosen array.
        arr = np.real(sol['y_events'][1][::2, 0])
        arr_t = np.real(sol['t_events'][1][::2])

        leng = len(arr)

        # max and min storage arrays
        ind_arr_max = []
        ind_arr_max_s = []
        ind_arr_min = []
        ind_arr_min_s = []

        # fill the storage arrays
        # since I am looking for peaks in the data, and neither local nor global maximum are actually the correct value
        # I separate the data into chunks and search for a global max for each chunk. 
        # I then compare to chuncks shifted by half a chunk size.
        # Then I throw away the boundary half chunks.
        for i in range(nn):
            ind_arr_max.append(np.argmax(arr[int(leng/nn)*i:int(leng/nn)*(i+1)]))
            ind_arr_max[-1] += int(leng/nn)*i
        for i in range(nn-1):
            ind_arr_max_s.append(np.argmax(arr[int(leng/nn*(i+0.5)):int(leng/nn*(i+1.5))]))
            ind_arr_max_s[-1] += int(leng/nn*(i+0.5))
        for i in range(nn):
            ind_arr_min.append(np.argmin(arr[int(leng/nn)*i:int(leng/nn)*(i+1)]))
            ind_arr_min[-1] += int(leng/nn)*i
        for i in range(nn-1):
            ind_arr_min_s.append(np.argmin(arr[int(leng/nn*(i+0.5)):int(leng/nn*(i+1.5))]))
            ind_arr_min_s[-1] += int(leng/nn*(i+0.5))

        # I need to check which curve I am following. I think my period is 4 times the distance between extrema.

        # calculate extrema
        # if the maxima on the shifted set is the same as that for the unshifted set, then it is a true local maxima
        # if not, it is just the edge of the data
        ind_arr_max_int = np.intersect1d(ind_arr_max, ind_arr_max_s)
        ind_arr_min_int = np.intersect1d(ind_arr_min, ind_arr_min_s)
        arr_minmax = np.union1d(ind_arr_min_int, ind_arr_max_int)

        # print extreema
        #print('Max:', np.max(arr[arr_minmax]), "Min:", np.min(arr[arr_minmax]))
        
        diff_arr = []

        for i in range(len(arr_t[arr_minmax])-1):
            dif = arr_t[arr_minmax][i+1]-arr_t[arr_minmax][i]
            diff_arr.append(dif)

        return diff_arr


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
        plot_code = 1 # only plot er, saving time

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

        # saves solution for analysis
        sol_er_u = None

        # saved data
        slow_oscillation_period = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        slow_oscillation_amplitude = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        slow_oscillation_std = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))

        slow_osc_sv_ind = []
        slow_osc_sv_measures = []

        ib = 0
        ir = 0
        im = 0
        ik = 0
        ik0 = 0
        ia = 0
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
                                        #print(self.l_calc.T_fast * 2, self.l_calc.A_max)

                                        sol_er_u = self.l_calc.solu

                                        slow_oscillation_period[im,ik,ib,ir,ia,ik0,idr,idi], slow_oscillation_std[im,ik,ib,ir,ia,ik0,idr,idi] = self.find_period(sol_er_u)
                                        slow_oscillation_amplitude[im,ik,ib,ir,ia,ik0,idr,idi] = self.find_amp(sol_er_u) 
                                        #slow_oscillation_period_0[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.T_fast_0_calc * 2

                                        slow_osc_sv_ind.append([im,ik,ib,ir,ia,ik0,idr,idi])
                                        slow_osc_sv_measures.append(self.find_gaps(sol_er_u))

                                        #print()

        self.slow_osc_t = slow_oscillation_period        # estimate for our quarter period for the slow oscillations (note the factor of 2*2 = 4)
        self.slow_osc_a = slow_oscillation_amplitude     # maximum absolute value
        self.slow_osc_s = slow_oscillation_std / slow_oscillation_period # standard deviation, in order to check validity of our estimates
        self.slow_osc_i = slow_osc_sv_ind           # indicies for our spacings, due to the alternative storage mechanism
        self.slow_osc_m = slow_osc_sv_measures      # save our estimates for spacing between extrema

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

            slow_oscillation_period[ii] = self.find_period(sol_er_u)
            slow_oscillation_amplitude[ii] = self.find_amp(sol_er_u) 
            #fast_oscillation_period_0[ii] = self.l_calc.T_fast_0_calc * 2

        self.fast_osc_t = slow_oscillation_period
        self.fast_osc_a = slow_oscillation_amplitude
        #self.fast_osc_t_0 = fast_oscillation_period_0    # because I want to know the error.

        self.calculated = True

        print("Done grid build: ", time.time() - start_time)