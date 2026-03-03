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
import time
import gc
import warnings

class grid_BVP:

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

    # Because we want to know what our object is.
    def __repr__(self):
        if self.is_grid:
            if self.calculated:
                str = "Grid object to measure current:\n"
            else:
                str = "Uncalculated grid object to measure current:\n"

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
            str = str + f"Total number of parameters: {self.num_mu*self.num_dk*self.num_k0*self.num_b*self.num_a*self.num_r}"
            return str
        elif self.is_mc:
            if self.calculated:
                str = "Monte carlo object to measure current"
            else:
                str = "Uncalculated monte carlo object to measure current"
            return str
        else:
            return "Empty object to measure current:\nR is %s, B is %s, dk is %s, k is %s, mu is %s, A is %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # Because we want to know what our object is.
    def __str__(self):
        if self.is_grid:
            if self.calculated:
                str = "Grid object to measure current:\n"
            else:
                str = "Uncalculated grid object to measure current:\n"

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
            str = str + f"Total number of parameters: {self.num_mu*self.num_dk*self.num_k0*self.num_b*self.num_a*self.num_r}"
            return str
        elif self.is_mc:
            if self.calculated:
                str = "Monte carlo object to measure current"
            else:
                str = "Uncalculated monte carlo object to measure current"
            return str
        else:
            return "Empty object to measure current:\nR is %s, B is %s, dk is %s, k is %s, mu is %s, A is %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)



    # ------------ CHOOSE POINTS (GRID) -----------------

    #def makeGridPoints(self, m_min, m_max, k_min, k_max, a_min, a_max, mu_min, mu_max, num = 10):
    # In order -- mu, dk, B, R, M, A, k0
    # num length -- 1, 2 (grid vs non-grid), n = defined vars + 1 for grid
    def makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), R = (-1.0, -1.0), 
                       A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = [10]):
        
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

            if j != ln:
                raise ValueError(f"Too many values passed for num. j = {j}, i = {i}, ln = {ln}")

    # inputs:
    #       self.ang_lim -- should remain constant; 
    #       self.dlim -- should change and be recorded in an array; scale of derivative
    #       self.grid_size -- must remain constant, set at start


    def clear_calcs(self):
        self.derivs   = None


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

        #print(var_set)

        self.var_set = var_set # so I can use this later

        # Create a table of values. 
        self.num = num
        self.val_table = np.zeros((num, 6))

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
    def gridBVP(self):

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

        # magnetic_field_strength  = self.mfs
        # electron_wavenumber      = self.ew
        # electron_wavenumber_diff = self.ewd
        # nonlinearity_strength    = self.nls
        # starting_amplitude       = self.amp
        # ring_radius              = self.rr

        #fast_oscillation_period = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0))
        #fast_oscillation_amplitude = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0))
        #fast_oscillation_period_0 = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0))

        found_derivs_array = []

        ib = 0
        ir = 0
        im = 0
        ik = 0
        ik0 = 0
        ia = 0

        # solve for each point on the grid
        # The 8 levels are only a problem if they are all used.
        print("Number of periods to calculate:", num_dk*num_k0*num_b*num_r*num_mu*num_a)
        for im in range(num_mu):
            for ik in range(num_dk):
                print(f"mu: {im}, dk: {ik}, Time: {time.time() - start_time}")
                for ib in range(num_b):
                    for ir in range(num_r):    
                        for ia in range(num_a):
                            for ik0 in range(num_k0):
                                # set the loop
                                self.l_calc.update_params(R=self.rr[ir], B=self.mfs[ib], dk=self.ewd[ik], 
                                                  mu=self.nls[im], k = self.ew[ik0], amp=self.amp[ia])
                                # find the roots which solve our BVP; this is the part which takes time (~30s)
                                with warnings.catch_warnings(action="ignore"):
                                    deriv_found = self.l_calc.find_root_many()
                                # numbers here are for the solution without ee interaction
                                psiprime0 = self.l_calc.psi_prime_0()
                                exactend = self.l_calc.psij0(2*pi*self.l_calc.R)
                                i0 = self.l_calc.current_old()
                                a0 = self.l_calc.aj0
                                b0 = self.l_calc.bj0
                                A_max_0 = self.l_calc.amp_max_0()
                                for deriv in deriv_found:
                                    self.l_calc.setDeriv(deriv)
                                    new_element = {"R": self.rr[ir],
                                                   "B": self.mfs[ib],
                                                   "dk": self.ewd[ik],
                                                   "mu": self.nls[im],
                                                   "k": self.ew[ik0],
                                                   "A": self.amp[ia],
                                                   "dpsi0": psiprime0,
                                                   "Exact End": exactend,
                                                   "a0": a0,
                                                   "b0": b0,
                                                   "A max 0": A_max_0,
                                                   "I0": i0,
                                                   "dpsi": deriv,
                                                   "New End": self.l_calc.psij_pred_true(2*pi*self.l_calc.R),
                                                   "a": self.l_calc.aj,
                                                   "b": self.l_calc.bj,
                                                   "A max new": self.l_calc.amp_max(),
                                                   "I v1": self.l_calc.current_new(),
                                                   "I v2": self.l_calc.current_alt(),
                                                   "I v3": self.l_calc.current_calc(),
                                                   "effective mu": self.l_calc.mu * self.l_calc.amp_max() ** 2
                                    }
                                    found_derivs_array.append(new_element)


        self.derivs = found_derivs_array

        self.calculated = True

        print("Done grid build: ", time.time() - start_time)


    def mcBVP(self):

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

        # self.val_table

        found_derivs_array = []
        
        # solve for each point on the grid
        print("Number of periods to calculate:", num)
        for ii in range(num):
            self.l_calc.update_params(R=self.val_table[ii,3], B=self.val_table[ii,2], dk=self.val_table[ii,1], 
                                      mu=self.val_table[ii,0], k = self.val_table[ii,5], amp=self.val_table[ii,4])
            # calculate derivative from our parameters if it is not as given

            # find the roots which solve our BVP; this is the part which takes time (~30s)
            with warnings.catch_warnings(action="ignore"):
                deriv_found = self.l_calc.find_root_many()
            # numbers here are for the solution without ee interaction
            psiprime0 = self.l_calc.psi_prime_0()
            exactend = self.l_calc.psij0(2*pi*self.l_calc.R)
            i0 = self.l_calc.current_old()
            a0 = self.l_calc.aj0
            b0 = self.l_calc.bj0
            A_max_0 = self.l_calc.amp_max_0()
            for deriv in deriv_found:
                self.l_calc.setDeriv(deriv)
                new_element = {"R": self.val_table[ii,3],
                               "B": self.val_table[ii,2],
                               "dk": self.val_table[ii,1],
                               "mu": self.val_table[ii,0],
                               "k": self.val_table[ii,5],
                               "A": self.val_table[ii,4],
                               "dpsi0": psiprime0,
                               "Exact End": exactend,
                               "a0": a0,
                               "b0": b0,
                               "A max 0": A_max_0,
                               "I0": i0,
                               "dpsi": deriv,
                               "New End": self.l_calc.psij_pred_true(2*pi*self.l_calc.R),
                               "a": self.l_calc.aj,
                               "b": self.l_calc.bj,
                               "A max new": self.l_calc.amp_max(),
                               "I v1": self.l_calc.current_new(),
                               "I v2": self.l_calc.current_alt(),
                               "I v3": self.l_calc.current_calc(),
                               "effective mu": self.l_calc.mu * self.l_calc.amp_max() ** 2
                              }
                found_derivs_array.append(new_element)

        self.derivs = found_derivs_array
        self.calculated = True

        print("Done grid build: ", time.time() - start_time)