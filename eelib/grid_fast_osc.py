# class grid_fast_osc 
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
import time
import gc


class grid_fast_osc:

    # ------ INITIALIZATION --------------

    # R, B, dk, mu, k, and amp must be given if they are fixed
    # k defaults to gold, amp defaults to 1
    def __init__(self, R=0.5, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.):
        self.l_calc = loop(R, B, dk, mu, k, amp) # I should probably do this with inheritance

        self.R = R
        self.B = B
        self.dk = dk
        self.k = k
        self.mu = mu
        self.a = amp

        self.is_grid = False
        self.is_mc = False
        self.calculated = False

        self.setIntegratorParameters()

    # There are many parameters for the integrator, which I tend to just set to the default and ignore.
    # Here they can be set once, and used every time.
    def setIntegratorParameters(self, solve_mu_0 = True, n = 20, method = 'RK45', rtol = rtol, atol = atol):
        self.l_calc.solve_mu_0 = solve_mu_0
        self.n = n
        self.method = method
        self.rtol = rtol
        self.atol = atol


    # ------------ CHOOSE POINTS (GRID) -----------------

    #def makeGridPoints(self, m_min, m_max, k_min, k_max, a_min, a_max, mu_min, mu_max, num = 10):
    # In order -- mu, dk, B, R, M, A, k0
    # num length -- 1, 2 (grid vs non-grid), n = defined vars + 1 for grid
    def makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), R = (-1.0, -1.0), 
                       A = (-1.0, -1.0), k0 = (-1.0, -1.0), ang_lim = 0.1, num = [10]):
        
        # Determines with which parameters to build the grid.
        # For dk, I need something else to indicate that it is not needed, as negative values in theory could be useful
        # however, I assume dk goes from 0 to 1.
        # There is a wider range of error-checking which can be performed at this step, limiting our ranges.
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
        #if self.is_mc: self.clear_mc_list()

        self.is_grid = True
        self.is_mc = False
        self.calculated = False

        gc.collect()

    # separated for clarity of previous code as this is long and repetitive
    # it is not to be run independently
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

    # inputs:
    #       self.ang_lim -- should remain constant; 
    #       self.dlim -- should change and be recorded in an array; scale of derivative
    #       self.grid_size -- must remain constant, set at start
    def makeDerivPoints(self):
        # form the derivative grid for t = 0
        # note that this uses the given values to choose the ideal points
        # the points should instead be based off of the other values
        x = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        x = np.cos(x) * self.dlim 
        y = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        y = np.cos(y) * self.dlim
        xx, yy = np.meshgrid(x, y)
        d0_grid = xx + 1j * yy
        self.d0_grid = d0_grid

    def clear_calcs(self):
        self.fast_osc_t   = None
        self.fast_osc_t_0 = None
        self.fast_osc_a   = None


    # ------------ CHOOSE POINTS (MONTE CARLO) -----------------

    # This is the alternative variant for function fitting. It works better for a combined regression task, 
    # but is not so clear when fitting the parameters separately, where I prefer the option of keeping other
    # parameters fixed. I have yet to implement this variant. (MC + stabalized LR or MCMC)


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
        #if self.is_mc: self.clear_mc_list()

        self.is_grid = False
        self.is_mc = True
        self.calculated = False

        gc.collect()




    # ------------- FIND VALUES ON GRID ---------------

    # def calculate_fast_t(self):
    # if self.is_grid: gridFastOsc()
    # if self.is_mc: mcFastOsc()

    # Perform the calculation on the grid.
    # This take time, but no longer requires parameters. It is separate from the parameters due to this.
    # I need to add functionality which uses self.is_grid, ...
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

        fast_oscillation_period = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        fast_oscillation_amplitude = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        fast_oscillation_period_0 = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))

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
                                        #print(self.l_calc.T_fast * 2, self.l_calc.A_max)
                                        fast_oscillation_period[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.T_fast * 2
                                        fast_oscillation_amplitude[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.A_max  
                                        fast_oscillation_period_0[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.T_fast_0_calc * 2

        self.fast_osc_t = fast_oscillation_period
        self.fast_osc_a = fast_oscillation_amplitude
        self.fast_osc_t_0 = fast_oscillation_period_0    # because I want to know the error.

        self.calculated = True

        print("Done grid build: ", time.time() - start_time)


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

        fast_oscillation_period = np.zeros((num))
        fast_oscillation_amplitude = np.zeros((num))
        fast_oscillation_period_0 = np.zeros((num))

        # solve for each point on the grid
        print("Number of periods to calculate:", num)
        for ii in range(num):
            self.l_calc.update_params(R=self.val_table[ii,3], B=self.val_table[ii,2], dk=self.val_table[ii,1], 
                                      mu=self.val_table[ii,0], k = self.val_table[ii,5], amp=self.val_table[ii,4])
            # calculate derivative from our parameters if it is not as given
            self.l_calc.setDeriv(self.val_table[ii,6]+ 1.j *self.val_table[ii,7], n = self.n, method = self.method,
                                                                        rtol = self.rtol, atol = self.atol)
            fast_oscillation_period[ii] = self.l_calc.T_fast * 2
            fast_oscillation_amplitude[ii] = self.l_calc.A_max  
            fast_oscillation_period_0[ii] = self.l_calc.T_fast_0_calc * 2

        self.fast_osc_t = fast_oscillation_period
        self.fast_osc_a = fast_oscillation_amplitude
        self.fast_osc_t_0 = fast_oscillation_period_0    # because I want to know the error.

        self.calculated = True

        print("Done grid build: ", time.time() - start_time)