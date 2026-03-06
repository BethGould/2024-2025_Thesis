# class deriv_grid
#  
# Takes fixed R, B, k, dk, mu, and amp, and creates a sinusoidal-spaced grid of 
# grid_size real and imaginary values. There are grid_size^2 runs of the loop.
# The runs are saved, and may be plotted later with the same object. 
# Plots allow for the possibility to plot the expected curve as given by our model.
# This code is only for visual analysis of the IVP results.
# Sample plotting script:
#
# * first set R (% R_max), B (% R_max), k (dk, 0-1 is one period for R = R_max)
# * mu (raw, ranges of 10^-10 to 10^-6 appear relevant, preferably 10^-8 to 10^-7)
# * n_g (number of points in one direction), no (plot number for plot naming)
# 
# > loopl = eelib.deriv_grid(R, B, k, mu, grid_size=n_g) # create object
# > loopl.derivGrid()                                    # run object

# PLOTTING FOR GRIDS (Large number of saved pdf files.)
# > for i in range(n_g):
# >     for j in range(n_g):
# >         loopl.plot_abs(i,j,k,R,B,mu,no)
# >         loopl.plot_real(i,j,k,R,B,mu,no)

# ----- Methods ---- 
# __init__(self, R, B, dk, mu, k = kFAu, amp=1., grid_size = 9, ang_lim = 0.1)
# derivGrid(self, n = 200, pr=1.0, method = 'RK45', rtol = rtol, atol = atol, trim = 16, to_plot = ['er','0r','0x','em'])
# plot_real(self, i, j, k, R, B, mu, no)
# plot_abs(self, i, j, k, R, B, mu, no)
#
# __repr__(self)
# __str__(self)
# change_parameters(self, R=None, B=None, dk=None, mu=None, k=None, amp = None, grid_size = None, ang_lim = None)
# clear_run(self)
#
# find_root_points(self,y_points)
# find_root_start(self,y_points)
# find_root_dif(self,y_points)
# find_amp(self, sol)

# change_parameters clears all the data and resets the object as if created with the given parameters.
# In this case, any parameters not given will be unchanged.
# clear_run just clears the data from the integration run of derivGrid. Nothing else is changed.

# to_plot can accept the following codes (assuming the code hasn't been broken, which is a possibility).
# 'er' -- Amplitude recovered (with error deleted) solution with electron-electron interaction.
# 'ed' -- Amplitude decreasing (erroneous) solution with electron-electron interaction.
# '0r' -- Amplitude recovered (with error deleted) solution without electron-electron interaction.
# '0d' -- Amplitude decreasing (erroneous) solution without electron-electron interaction.
# '0x' -- Exact solution without electron-electron interaction. 
# 'em' -- Modeled solution with electron-electron interaction.
# Desired codes are passed as a list of two character strings. Unrecognized codes are most likely ignored.

# derivGrid runs the integration, the most intensive part.
# n and trim reduce the number of points on the plots.
# pr is the percentage of the radius to evaluate. It can also be set greater that 1.0 to get more data.
# method, rtol, atol control the integrator.
# to_plot gives the versions of integration to evaluate, with all saved and plotted.

# plot_real, plot_abs plot and save the indicated loop. 
# i, j are the indicies in the grid of the desired loop.
# k, R, B, mu are for labeling the plot.
# no is an index used to label the plots, allowing for unique labeling.

# find_root_points, find_root_start_find_root_dif_ and find_amp are used by the code to fit the slow 
# oscillations to a sine function. They don't interact with the object, so external use is irrelevant. 

# Error handling has not been added to this class, so it may function incorrectly if used incorrectly.


# --- LIBRARIES ---
import numpy as np
import matplotlib.pyplot as plt
from eelib.consts import pi, kFAu, rtol, atol, R_max, B_max
from eelib.loop import loop
import time

class deriv_grid:

    # Initialize the grid by creating a loop object and choosing derivitive points on the grid.

    def __init__(self, R, B, dk, mu, k = kFAu, amp=1., grid_size = 9, ang_lim = 0.1):
        self.l_calc = loop(R, B, dk, mu, k, amp)  # loop class for calculations
        self.grid_size = grid_size
        self.dlim = self.l_calc.psi_prime_0_max() # max(abs(psi'(0)))

        self.ang_lim = ang_lim
        self.R = R
        self.B = B
        self.dk = dk
        self.mu = mu
        self.k = k
        self.amp = amp
        
        # form the derivative grid for t = 0
        # ang_limit helps prevent issues, while the max grid size is determined by the maximum amplitude
        # of the exact solution. 
        x = np.linspace(pi-ang_lim, ang_lim, num=self.grid_size, dtype = complex)
        x = np.cos(x) * self.dlim 
        y = np.linspace(pi-ang_lim, ang_lim, num=self.grid_size, dtype = complex)
        y = np.cos(y) * self.dlim
        xx, yy = np.meshgrid(x, y)
        d0_grid = xx + 1j * yy
        self.d0_grid = d0_grid

        self.run = False

    # Outputs for the print statement.
    def __repr__(self):
        if self.run:
            str = "Object with a grid of loop runs for various derivatives:\n"
        else:
            str = "Object to construct a grid of loop runs for various derivatives:\n"

        # And show our parameters.
        str = str + f"mu is: {self.mu}\n"
        str = str + f"dk is: {self.dk}\n"
        str = str + f"B is: {self.B}\n"
        str = str + f"R is: {self.R}\n"
        str = str + f"A is: {self.amp}\n"
        str = str + f"k0 is: {self.k}\n"
        str = str + f"Grid size (in each direction -- real and imaginary): {self.grid_size}"
        return str

    def __str__(self):
        if self.run:
            str = "Object with a grid of loop runs for various derivatives:\n"
        else:
            str = "Object to construct a grid of loop runs for various derivatives:\n"
            
        # And show our parameters.
        str = str + f"mu is: {self.mu}\n"
        str = str + f"dk is: {self.dk}\n"
        str = str + f"B is: {self.B}\n"
        str = str + f"R is: {self.R}\n"
        str = str + f"A is: {self.amp}\n"
        str = str + f"k0 is: {self.k}\n"
        str = str + f"Grid size is (in each direction -- real and imaginary): {self.grid_size}"
        return str

    # Code to reset the grid object and change parameters.
    def change_parameters(self, R=None, B=None, dk=None, mu=None, k=None, amp = None, grid_size = None, ang_lim = None):
        # Clear the data for the old parameters.
        self.clear_run()

        # Change parameters which are set in the code.
        if R is not None: 
            self.R = R
        if B is not None:
            self.B = B
        if dk is not None:
            self.dk = dk
        if mu is not None: 
            self.mu = mu
        if amp is not None: 
            self.amp = amp
        if k is not None:
            self.k = k
        if grid_size is not None:
            self.grid_size = grid_size
        if ang_lim is not None:
            self.ang_lim = ang_lim

        # Apply this change in parameters:
        self.l_calc.update_params(self.R, self.B, self.dk, self.mu, self.k, self.amp)
        self.dlim = self.l_calc.psi_prime_0_max()
        self.d0_grid = None

        # form the derivative grid for t = 0
        # ang_limit helps prevent issues, while the max grid size is determined by the maximum amplitude
        # of the exact solution. 
        x = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        x = np.cos(x) * self.dlim 
        y = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        y = np.cos(y) * self.dlim
        xx, yy = np.meshgrid(x, y)
        d0_grid = xx + 1j * yy
        self.d0_grid = d0_grid

    # Delete all the data. 
    def clear_run(self):
        self.run = False

        self.s_grid_er_u = None
        self.s_grid_ed_u = None
        self.s_grid_em_u = None
        self.s_grid_M_pred = None
        self.s_grid_0d_u = None
        self.s_grid_0r_u = None
        self.s_grid_0x_u_tc = None
        self.s_grid_0x_l_tc = None
        self.s_grid_0x_u_tcm = None
        self.s_grid_0x_l_tcm = None
        self.s_grid_0x_u_ta = None
        self.s_grid_0x_l_ta = None
        self.s_grid_0x_u_tr = None
        self.s_grid_0x_l_tr = None
        self.s_grid_0x_u_yc = None
        self.s_grid_0x_l_yc = None
        self.s_grid_0x_u_ycm = None
        self.s_grid_0x_l_ycm = None
        self.s_grid_0x_u_ya = None
        self.s_grid_0x_l_ya = None
        self.s_grid_0x_u_yr = None
        self.s_grid_0x_l_yr = None

    # This just creates a grid of values for various psi'_0 values.
    # Due to running the integrator multiple times, it can be quite slow. Therefore, the program uses
    # the time library to calculate the time required regularly, to allow for keeping track of how much
    # time is left. 
    # n -- number of points to plot, used for untriggered, predefined points
    # pr -- percent_range
    # method, rtol (relative tolerance), atol (absolute tolerance)
    # trim -- needs to be even, every x point is used for the plots, used for triggered points
    # to_plot -- list of strings with choises for plotting
    # n_start -- number of oscillations to use in averaging to estimate k for the fast oscillations

    def derivGrid(self, n = 200, pr=1.0, method = 'RK45', rtol = rtol, atol = atol, trim = 16, 
                  to_plot = ['er', '0r', '0x', 'em'], n_start=20):

        if self.run:
            print("This object has already calculated the grid.")
            return None

        # save imput parameters
        self.trim = trim
        self.to_plot = to_plot

        # start timing
        start_time = time.time()

        # Convert the clear plot code system of the deriv_grid class to the difficult to understand one of the loop class.
        plot_code = -1
        if 'er' in to_plot: plot_code *= -1
        if 'ed' in to_plot: plot_code *= 2
        if '0d' in to_plot: plot_code *= 3
        if '0r' in to_plot: plot_code *= 5
        if 'em' in to_plot: plot_code *= 7

        # Places for our solutions (which will be lists of lists of solutions).
        '''
        u, l -- upper, lower
        e -- with ee-interaction
        0 -- without ee-interaction
        d -- full calculation
        r -- piecewise, restored power
        m -- piecewise, restored power, triggered by estimated fast period
        x -- exact
        tc, ta, tr -- time for calculated, abs triggered, real triggered
        yc, ya, yr -- value for calculated, abs triggered, real triggered
        '''

        if 'er' in to_plot: s_grid_er_u = []
        if 'er' in to_plot: s_grid_er_l = []
        if 'ed' in to_plot: s_grid_ed_u = []
        if 'ed' in to_plot: s_grid_ed_l = []
        if 'em' in to_plot: s_grid_em_u = []
        if 'em' in to_plot: s_grid_em_l = []

        if 'er' in to_plot: s_grid_M_pred = []

        if '0d' in to_plot: s_grid_0d_u = []
        if '0d' in to_plot: s_grid_0d_l = []
        if '0r' in to_plot: s_grid_0r_u = []
        if '0r' in to_plot: s_grid_0r_l = []

        if '0x' in to_plot: s_grid_0x_u_tc = []
        if '0x' in to_plot: s_grid_0x_l_tc = []
        if '0x' in to_plot: s_grid_0x_u_tcm = []
        if '0x' in to_plot: s_grid_0x_l_tcm = []
        if '0r' in to_plot:
            if '0x' in to_plot: s_grid_0x_u_ta = []
            if '0x' in to_plot: s_grid_0x_l_ta = []
            if '0x' in to_plot: s_grid_0x_u_tr = []
            if '0x' in to_plot: s_grid_0x_l_tr = []
        if '0x' in to_plot: s_grid_0x_u_yc = []
        if '0x' in to_plot: s_grid_0x_l_yc = []
        if '0x' in to_plot: s_grid_0x_u_ycm = []
        if '0x' in to_plot: s_grid_0x_l_ycm = []
        if '0r' in to_plot:
            if '0x' in to_plot: s_grid_0x_u_ya = []
            if '0x' in to_plot: s_grid_0x_l_ya = []
            if '0x' in to_plot: s_grid_0x_u_yr = []
            if '0x' in to_plot: s_grid_0x_l_yr = []

        # Begin the time-consuming part of the code
        print('Begin grid build: ', time.time() - start_time)
        print('Plot Code:', to_plot, plot_code)
        
        # Solve for each point on the grid
        for i in range(self.grid_size):
            # The grid is stored on a two-dimensional list. This creates the second dimension.
            if 'er' in to_plot: s_grid_er_u.append([])
            if 'er' in to_plot: s_grid_er_l.append([])
            if 'ed' in to_plot: s_grid_ed_u.append([])
            if 'ed' in to_plot: s_grid_ed_l.append([])
            if 'em' in to_plot: s_grid_em_u.append([])
            if 'em' in to_plot: s_grid_em_l.append([])

            if 'er' in to_plot: s_grid_M_pred.append([])

            if '0d' in to_plot: s_grid_0d_u.append([])
            if '0d' in to_plot: s_grid_0d_l.append([])
            if '0r' in to_plot: s_grid_0r_u.append([])
            if '0r' in to_plot: s_grid_0r_l.append([])

            if '0x' in to_plot: 
                s_grid_0x_u_tc.append([])
                s_grid_0x_l_tc.append([])
                s_grid_0x_u_tcm.append([])
                s_grid_0x_l_tcm.append([])
                if '0r' in to_plot:
                    s_grid_0x_u_ta.append([])
                    s_grid_0x_l_ta.append([])
                    s_grid_0x_u_tr.append([])
                    s_grid_0x_l_tr.append([])
                s_grid_0x_u_yc.append([])
                s_grid_0x_l_yc.append([])
                s_grid_0x_u_ycm.append([])
                s_grid_0x_l_ycm.append([])
                if '0r' in to_plot:
                    s_grid_0x_u_ya.append([])
                    s_grid_0x_l_ya.append([])
                    s_grid_0x_u_yr.append([])
                    s_grid_0x_l_yr.append([])
            
            # Output our current stage of calculation and the time so far taken.
            # I switched this from counting from 0 to counting from 1 to be more intuitive. 
            print("i = ", i + 1, "of ", self.grid_size, "time: ", time.time() - start_time)

            for j in range(self.grid_size):
                # This is the one and only place where we call the integrator, which is slow. 
                # Note that find_fast_oscillations has been moved here from setDeriv, as it was 
                # slowing down my BVP matching algorithm. 
                self.l_calc.setDeriv(self.d0_grid[j,i]) 
                # Note that all the old runs were from code in the format self.d0_grid[j,i],
                # which gives the imaginary derivative variations as the first index and real variations as the second.
                self.l_calc.find_fast_oscillations(n=n_start, method = method, rtol = rtol, atol = atol)
                self.l_calc.solve_ivp(n = n, percent_range = pr, method = method, rtol = rtol, atol = atol, solve=plot_code)

                # Output our current stage of calculation and the time so far taken.
                # I switched this from counting from 0 to counting from 1 to be more intuitive. 
                print("Done, i, j = ", i+1,j+1, "time: ", time.time() - start_time)

                # Now we are appending all of our solutions to our saved solution lists, chosen based on our plot_code.
                if 'er' in to_plot: s_grid_er_u[i].append(self.l_calc.solu)
                if 'ed' in to_plot: s_grid_ed_u[i].append(self.l_calc.solu_d)
                if 'em' in to_plot: s_grid_em_u[i].append(self.l_calc.solu_m)

                if '0d' in to_plot: s_grid_0d_u[i].append(self.l_calc.solu0)
                if '0r' in to_plot: s_grid_0r_u[i].append(self.l_calc.solu0_r)

                if '0x' in to_plot:
                    T_arr = self.l_calc.find_period_shift_exact() # fast, slow, pos, neg

                    # I need start times for the real amplitude following cases
                    # they are from s_grid_0x_u_tc[np.argmax(abs(s_grid_0x_u_yc))]
                    # but then need to caclulate backwards
                    xT = self.l_calc.find_real_env_start()

                    s_grid_0x_u_tc[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stu_ex, T_arr[0]))
                    s_grid_0x_l_tc[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stl_ex, T_arr[0]))
                    s_grid_0x_u_tcm[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, xT, T_arr[2]))
                    s_grid_0x_l_tcm[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, xT, T_arr[3]))

                    if '0r' in to_plot:
                        # triggered times
                        s_grid_0x_u_ta[i].append(self.l_calc.solu0_r['t_events'][0][0::trim])
                        s_grid_0x_l_ta[i].append(self.l_calc.solu0_r['t_events'][0][1::trim])
                        s_grid_0x_u_tr[i].append(self.l_calc.solu0_r['t_events'][1][0::trim])
                        s_grid_0x_l_tr[i].append(self.l_calc.solu0_r['t_events'][1][1::trim])

                    # y values
                    s_grid_0x_u_yc[i].append(self.l_calc.psij(s_grid_0x_u_tc[i][j]))
                    s_grid_0x_l_yc[i].append(self.l_calc.psij(s_grid_0x_l_tc[i][j]))
                    s_grid_0x_u_ycm[i].append(self.l_calc.psij(s_grid_0x_u_tcm[i][j]))
                    s_grid_0x_l_ycm[i].append(self.l_calc.psij(s_grid_0x_l_tcm[i][j]))
                    
                    if '0r' in to_plot:
                        s_grid_0x_u_ya[i].append(self.l_calc.psij(s_grid_0x_u_ta[i][j]))
                        s_grid_0x_l_ya[i].append(self.l_calc.psij(s_grid_0x_l_ta[i][j]))
                        s_grid_0x_u_yr[i].append(self.l_calc.psij(s_grid_0x_u_tr[i][j]))
                        s_grid_0x_l_yr[i].append(self.l_calc.psij(s_grid_0x_l_tr[i][j]))
                    
                if 'er' in to_plot: s_grid_M_pred[i].append(self.l_calc.T_slow_mod)

        # And now we need to save our solutions (because I, for some reason, chose not to save them earlier).
        if 'er' in to_plot: self.s_grid_er_u = s_grid_er_u
        if 'ed' in to_plot: self.s_grid_ed_u = s_grid_ed_u
        if 'em' in to_plot: self.s_grid_em_u = s_grid_em_u

        if 'er' in to_plot: self.s_grid_M_pred = s_grid_M_pred

        if '0d' in to_plot: self.s_grid_0d_u = s_grid_0d_u
        if '0r' in to_plot: self.s_grid_0r_u = s_grid_0r_u

        if '0x' in to_plot: self.s_grid_0x_u_tc = s_grid_0x_u_tc
        if '0x' in to_plot: self.s_grid_0x_l_tc = s_grid_0x_l_tc
        if '0x' in to_plot: self.s_grid_0x_u_tcm = s_grid_0x_u_tcm
        if '0x' in to_plot: self.s_grid_0x_l_tcm = s_grid_0x_l_tcm
        if '0r' in to_plot:
            if '0x' in to_plot: self.s_grid_0x_u_ta = s_grid_0x_u_ta
            if '0x' in to_plot: self.s_grid_0x_l_ta = s_grid_0x_l_ta
            if '0x' in to_plot: self.s_grid_0x_u_tr = s_grid_0x_u_tr
            if '0x' in to_plot: self.s_grid_0x_l_tr = s_grid_0x_l_tr
        if '0x' in to_plot: self.s_grid_0x_u_yc = s_grid_0x_u_yc
        if '0x' in to_plot: self.s_grid_0x_l_yc = s_grid_0x_l_yc
        if '0x' in to_plot: self.s_grid_0x_u_ycm = s_grid_0x_u_ycm
        if '0x' in to_plot: self.s_grid_0x_l_ycm = s_grid_0x_l_ycm
        if '0r' in to_plot:
            if '0x' in to_plot: self.s_grid_0x_u_ya = s_grid_0x_u_ya
            if '0x' in to_plot: self.s_grid_0x_l_ya = s_grid_0x_l_ya
            if '0x' in to_plot: self.s_grid_0x_u_yr = s_grid_0x_u_yr
            if '0x' in to_plot: self.s_grid_0x_l_yr = s_grid_0x_l_yr

        # And now we output the full timing.
        print("Done grid build: ", time.time() - start_time)
        self.run = True


# ---------

    # These functions are for use with grid plotting (sine fitting), and are not for external use.
    # They assume nice behavior. So far, it works well, but I can think of cases when it won't.
    def find_root_points(self,y_points):
        y_mult = np.array(y_points[:-1])*np.array(y_points[1:])
        zero_index = np.nonzero(y_mult == 0)
        root_index = np.nonzero(y_mult < 0)
        return root_index[0]

    def find_root_start(self,y_points):
        root_ls = self.find_root_points(y_points)
        if y_points[root_ls[0]+1] > 0: 
            rt_start = root_ls[0]
        else: 
            rt_start = root_ls[1]
        return rt_start

    def find_root_dif(self,y_points):
        root_list = self.find_root_points(y_points)
        y_diff = np.array(root_list[1:])- np.array(root_list[:-1])
        return np.average(y_diff)
    
    def find_amp(self, sol):
        arr = np.real(sol['y_events'][1][:, 0])
        return np.max(np.abs(arr))

# plotting the grid
    def plot_real(self, i, j, k, R, B, mu, no):

        # Because I still don't want to write self.* all the time.
        to_plot = self.to_plot

        # currently the below parameters are used for labeling the graph, 
        # and should be shorter than the actual parameters
        #k = self.l_calc.dk
        #R = self.l_calc.R
        #B = self.l_calc.B
        #mu = self.l_calc.mu

        # This will calculate the model for the curve for slow oscillations. It is a sine curve
        # which models sampling at the same part of every fast oscillation period. The fast 
        # oscillation model is given by the 'em' curve, which integrates the solution, providing 
        # outputs at every time given by our modeled k, allowing for the difference between the 
        # actual and modeled k values to be observed.
        if 'er' in to_plot: 
            M_pred = self.s_grid_M_pred[i][j]
            A_pred = self.find_amp(self.s_grid_er_u[i][j])
            sol_t = np.real(self.s_grid_er_u[i][j]['t']) 
            sol_y = np.real(self.s_grid_er_u[i][j]['y'][0])
            ind_st = self.find_root_start(sol_y)
            t0_pred = sol_t[ind_st]
            t_pred = sol_t
            y_pred = A_pred * np.sin(2*pi / M_pred * (t_pred-t0_pred))

        #position arrays
        #estimated, triggered buy abs, triggered by real (both) -- c, a, r
        if 'er' in to_plot: tuerr = np.real(self.s_grid_er_u[i][j]['t_events'][1][0::self.trim])
        if '0r' in to_plot: tu0rr = np.real(self.s_grid_0r_u[i][j]['t_events'][1][0::self.trim])
        if 'ed' in to_plot: tuedr = np.real(self.s_grid_ed_u[i][j]['t_events'][1][0::self.trim])
        if '0d' in to_plot: tu0dr = np.real(self.s_grid_0d_u[i][j]['t_events'][1][0::self.trim])
        if 'er' in to_plot: tlerr = np.real(self.s_grid_er_u[i][j]['t_events'][1][1::self.trim])
        if '0r' in to_plot: tl0rr = np.real(self.s_grid_0r_u[i][j]['t_events'][1][1::self.trim])
        if 'ed' in to_plot: tledr = np.real(self.s_grid_ed_u[i][j]['t_events'][1][1::self.trim])
        if '0d' in to_plot: tl0dr = np.real(self.s_grid_0d_u[i][j]['t_events'][1][1::self.trim])

        # abs triggered
        if 'er' in to_plot: tuera = np.real(self.s_grid_er_u[i][j]['t_events'][0][0::self.trim])
        if '0r' in to_plot: tu0ra = np.real(self.s_grid_0r_u[i][j]['t_events'][0][0::self.trim])
        if 'ed' in to_plot: tueda = np.real(self.s_grid_ed_u[i][j]['t_events'][0][0::self.trim])
        if '0d' in to_plot: tu0da = np.real(self.s_grid_0d_u[i][j]['t_events'][0][0::self.trim])
        if 'er' in to_plot: tlera = np.real(self.s_grid_er_u[i][j]['t_events'][0][1::self.trim])
        if '0r' in to_plot: tl0ra = np.real(self.s_grid_0r_u[i][j]['t_events'][0][1::self.trim])
        if 'ed' in to_plot: tleda = np.real(self.s_grid_ed_u[i][j]['t_events'][0][1::self.trim])
        if '0d' in to_plot: tl0da = np.real(self.s_grid_0d_u[i][j]['t_events'][0][1::self.trim])
        
        #value arrays
        if 'er' in to_plot: suerr = np.real(self.s_grid_er_u[i][j]['y_events'][1][0::self.trim, 0])
        if 'ed' in to_plot: suedr = np.real(self.s_grid_ed_u[i][j]['y_events'][1][0::self.trim, 0])
        if 'er' in to_plot: slerr = np.real(self.s_grid_er_u[i][j]['y_events'][1][1::self.trim, 0])
        if 'ed' in to_plot: sledr = np.real(self.s_grid_ed_u[i][j]['y_events'][1][1::self.trim, 0])
        if '0r' in to_plot: su0rr = np.real(self.s_grid_0r_u[i][j]['y_events'][1][0::self.trim, 0])
        if '0d' in to_plot: su0dr = np.real(self.s_grid_0d_u[i][j]['y_events'][1][0::self.trim, 0])
        if '0r' in to_plot: sl0rr = np.real(self.s_grid_0r_u[i][j]['y_events'][1][1::self.trim, 0])
        if '0d' in to_plot: sl0dr = np.real(self.s_grid_0d_u[i][j]['y_events'][1][1::self.trim, 0])

        # abs triggered
        if 'er' in to_plot: suera = np.real(self.s_grid_er_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'ed' in to_plot: sueda = np.real(self.s_grid_ed_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'er' in to_plot: slera = np.real(self.s_grid_er_u[i][j]['y_events'][0][1::self.trim, 0])
        if 'ed' in to_plot: sleda = np.real(self.s_grid_ed_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0r' in to_plot: su0ra = np.real(self.s_grid_0r_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0d' in to_plot: su0da = np.real(self.s_grid_0d_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0r' in to_plot: sl0ra = np.real(self.s_grid_0r_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0d' in to_plot: sl0da = np.real(self.s_grid_0d_u[i][j]['y_events'][0][1::self.trim, 0])

        # exact solutions
        if '0x' in to_plot: 
            tu0xc = np.real(self.s_grid_0x_u_tc[i][j])
            tl0xc = np.real(self.s_grid_0x_l_tc[i][j])
            su0xc = np.real(self.s_grid_0x_u_yc[i][j])
            sl0xc = np.real(self.s_grid_0x_l_yc[i][j])
            if '0r' in to_plot:
                tu0xr = np.real(self.s_grid_0x_u_tr[i][j])
                tl0xr = np.real(self.s_grid_0x_l_tr[i][j])
                su0xr = np.real(self.s_grid_0x_u_yr[i][j])
                sl0xr = np.real(self.s_grid_0x_l_yr[i][j])
        
        #--PLOTTING--
        # Plots the values for points triggered by the absolute value reaching a maximum,
        # thus showing the envelope of psi.
        fig, ax = plt.subplots()
        ax.set_ylabel('Real Part of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, abs triggered, dk={k}, R={R}, B={B}, \u03BC={mu}")

        # I form my label based on a list which is appended, so that only the plots which exist will
        # be plotted.
        h_list = []

        if 'er' in to_plot: 
            if np.max(suera) > np.max(slera):
                line1, = ax.plot(tuera, suera, color = 'red', label = 'with e-e interaction')
            else:
                line1, = ax.plot(tlera, slera, color = 'red', label = 'with e-e interaction')
            line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')
            h_list.append(line1)
            h_list.append(line13)
        if 'ed' in to_plot: 
            line3, = ax.plot(tueda, sueda, color = 'orange', label = 'with e-e interaction')
            line4, = ax.plot(tleda, sleda, color = 'orange', label = 'with e-e interaction')
            h_list.append(line3)
        if '0r' in to_plot: 
            if np.max(su0ra) > np.max(sl0ra):
                line5, = ax.plot(tu0ra, su0ra, color = 'green', label = 'without e-e interaction')
            else:
                line5, = ax.plot(tl0ra, sl0ra, color = 'green', label = 'without e-e interaction')
            h_list.append(line5)
        if '0d' in to_plot: 
            line7, = ax.plot(tu0da, su0da, color = 'blue', label = 'without e-e interaction')
            line8, = ax.plot(tl0da, sl0da, color = 'blue', label = 'without e-e interaction')
            h_list.append(line7)
        if '0x' in to_plot: 
            if np.max(su0xc) > np.max(sl0xc):
                line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'exact solution')
            else:
                line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'exact solution')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_real_a.pdf", format='pdf')
        plt.close()

        # This curve is determined by plotting cases when the derivative of the real part is zero.
        fig, ax = plt.subplots()
        ax.set_ylabel('Real Part of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, triggered, dk={k}, R={R}, B={B}, \u03BC={mu}")

        h_list = []

        if 'er' in to_plot: 
            line1, = ax.plot(tuerr, suerr, color = 'red', label = 'with e-e interaction')
            line2, = ax.plot(tlerr, slerr, color = 'red', label = 'with e-e interaction')
            line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')
            h_list.append(line1)
            h_list.append(line13)
        if 'ed' in to_plot: 
            line3, = ax.plot(tuedr, suedr, color = 'orange', label = 'with e-e interaction')
            line4, = ax.plot(tledr, sledr, color = 'orange', label = 'with e-e interaction')
            h_list.append(line3)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0rr, su0rr, color = 'green', label = 'without e-e interaction')
            line6, = ax.plot(tl0rr, sl0rr, color = 'green', label = 'without e-e interaction')
            h_list.append(line5)
        if '0d' in to_plot: 
            line7, = ax.plot(tu0dr, su0dr, color = 'blue', label = 'without e-e interaction')
            line8, = ax.plot(tl0dr, sl0dr, color = 'blue', label = 'without e-e interaction')
            h_list.append(line7)
        if '0x' in to_plot: 
            if np.max(su0xc) > np.max(sl0xc):
                line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'exact solution')
            else:
                line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'exact solution')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_real_r.pdf", format='pdf')
        plt.close()

        # This also triggers on the maxima of the real values and also uses these calculated points for 
        # the points at which to calculate the exact solution. This will ensure that the exact solution also
        # produces an envelope rather than a sinusoid.
        fig, ax = plt.subplots()
        ax.set_ylabel('Real Part of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, triggered, dk={k}, R={R}, B={B}, \u03BC={mu}")

        h_list = []

        if 'er' in to_plot: 
            line1, = ax.plot(tuerr, suerr, color = 'red', label = 'with e-e interaction')
            line2, = ax.plot(tlerr, slerr, color = 'red', label = 'with e-e interaction')
            h_list.append(line1)
        if 'ed' in to_plot: 
            line3, = ax.plot(tuedr, suedr, color = 'orange', label = 'with e-e interaction')
            line4, = ax.plot(tledr, sledr, color = 'orange', label = 'with e-e interaction')
            h_list.append(line3)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0rr, su0rr, color = 'green', label = 'without e-e interaction')
            line6, = ax.plot(tl0rr, sl0rr, color = 'green', label = 'without e-e interaction')
            h_list.append(line5)
        if '0d' in to_plot: 
            line7, = ax.plot(tu0dr, su0dr, color = 'blue', label = 'without e-e interaction')
            line8, = ax.plot(tl0dr, sl0dr, color = 'blue', label = 'without e-e interaction')
            h_list.append(line7)
        if '0x' in to_plot: 
            if '0r' in to_plot:
                line9, = ax.plot(tu0xr, su0xr, color = 'purple', label = 'exact solution')
                line10, = ax.plot(tl0xr, sl0xr, color = 'purple', label = 'exact solution')
                h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_real_rx.pdf", format='pdf')
        plt.close()


    # --- PLOTS OF ABSOLUTE VALUE ---
    def plot_abs(self, i, j, k, R, B, mu, no):

        to_plot = self.to_plot

        # currently the below parameters are used for labeling the graph, 
        # and should be shorter than the actual parameters
        #k = self.l_calc.dk
        #R = self.l_calc.R
        #B = self.l_calc.B
        #mu = self.l_calc.mu

        #position arrays
        # estimated, triggered buy abs, triggered by real (both) -- e, a, r
        # here everything is triggered by absolute value
        # This should create horizontal lines on the graph. 
        if 'er' in to_plot: tuera = np.abs(self.s_grid_er_u[i][j]['t_events'][0][0::self.trim])
        if '0r' in to_plot: tu0ra = np.abs(self.s_grid_0r_u[i][j]['t_events'][0][0::self.trim])
        if 'ed' in to_plot: tueda = np.abs(self.s_grid_ed_u[i][j]['t_events'][0][0::self.trim])
        if '0d' in to_plot: tu0da = np.abs(self.s_grid_0d_u[i][j]['t_events'][0][0::self.trim])
        if 'er' in to_plot: tlera = np.abs(self.s_grid_er_u[i][j]['t_events'][0][1::self.trim])
        if '0r' in to_plot: tl0ra = np.abs(self.s_grid_0r_u[i][j]['t_events'][0][1::self.trim])
        if 'ed' in to_plot: tleda = np.abs(self.s_grid_ed_u[i][j]['t_events'][0][1::self.trim])
        if '0d' in to_plot: tl0da = np.abs(self.s_grid_0d_u[i][j]['t_events'][0][1::self.trim])

        # Try to also plot the versions which are calculated. 
        # The accuracy of the calculation is seen by the drop in what should be a straight line.
        if 'er' in to_plot: tuerc = np.abs(self.s_grid_er_u[i][j]['t'])
        if '0r' in to_plot: tu0rc = np.abs(self.s_grid_0r_u[i][j]['t'])
        if 'em' in to_plot: tuemc = np.abs(self.s_grid_em_u[i][j]['t'])

        #value arrays
        if 'er' in to_plot: suera = np.abs(self.s_grid_er_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'ed' in to_plot: sueda = np.abs(self.s_grid_ed_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'er' in to_plot: slera = np.abs(self.s_grid_er_u[i][j]['y_events'][0][1::self.trim, 0])
        if 'ed' in to_plot: sleda = np.abs(self.s_grid_ed_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0r' in to_plot: su0ra = np.abs(self.s_grid_0r_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0d' in to_plot: su0da = np.abs(self.s_grid_0d_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0r' in to_plot: sl0ra = np.abs(self.s_grid_0r_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0d' in to_plot: sl0da = np.abs(self.s_grid_0d_u[i][j]['y_events'][0][1::self.trim, 0])

        # Try to also plot the versions which are calculated.
        if 'er' in to_plot: suerc = np.abs(self.s_grid_er_u[i][j]['y'][0])
        if '0r' in to_plot: su0rc = np.abs(self.s_grid_0r_u[i][j]['y'][0])
        if 'em' in to_plot: suemc = np.abs(self.s_grid_em_u[i][j]['y'][0])

        # exact solutions
        # On the real grid, the calculated version looks the best.
        if '0x' in to_plot: 
            tu0xa = np.abs(self.s_grid_0x_u_tc[i][j])
            tl0xa = np.abs(self.s_grid_0x_l_tc[i][j])
            su0xa = np.abs(self.s_grid_0x_u_yc[i][j])
            sl0xa = np.abs(self.s_grid_0x_l_yc[i][j])


        # ------ PLOTTING ------
        # --- calculated ---
        fig, ax = plt.subplots()
        ax.set_ylabel('Absolute value of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, calculated, dk={k}, R={R}, B={B}, \u03BC={mu}")
        
        h_list = []

        if 'er' in to_plot: 
            line1, = ax.plot(tuerc, suerc, color = 'red', label = 'with e-e interaction')
            h_list.append(line1)
        if 'em' in to_plot: 
            line2, = ax.plot(tuemc, suemc, color = 'goldenrod', label = 'with e-e interaction')
            h_list.append(line2)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0rc, su0rc, color = 'green', label = 'without e-e interaction')
            h_list.append(line5)
        if '0x' in to_plot: 
            line9, = ax.plot(tu0xa, su0xa, color = 'purple', label = 'exact solution')
            line10, = ax.plot(tl0xa, sl0xa, color = 'purple', label = 'exact solution')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_abs_c.pdf", format='pdf')
        plt.close()
        
        # --- abs triggered envelope ---
        fig, ax = plt.subplots()
        ax.set_ylabel('Absolute value of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, triggered, dk={k}, R={R}, B={B}, \u03BC={mu}")

        h_list = []

        if 'er' in to_plot: 
            line1, = ax.plot(tuera, suera, color = 'red', label = 'with e-e interaction')
            line2, = ax.plot(tlera, slera, color = 'red', label = 'with e-e interaction')
            h_list.append(line1)
        if 'ed' in to_plot: 
            line3, = ax.plot(tueda, sueda, color = 'orange', label = 'with e-e interaction')
            line4, = ax.plot(tleda, sleda, color = 'orange', label = 'with e-e interaction')
            h_list.append(line3)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0ra, su0ra, color = 'green', label = 'without e-e interaction')
            line6, = ax.plot(tl0ra, sl0ra, color = 'green', label = 'without e-e interaction')
            h_list.append(line5)
        if '0d' in to_plot: 
            line7, = ax.plot(tu0da, su0da, color = 'blue', label = 'without e-e interaction')
            line8, = ax.plot(tl0da, sl0da, color = 'blue', label = 'without e-e interaction')
            h_list.append(line7)
        if '0x' in to_plot: 
            line9, = ax.plot(tu0xa, su0xa, color = 'purple', label = 'exact solution')
            line10, = ax.plot(tl0xa, sl0xa, color = 'purple', label = 'exact solution')
            h_list.append(line9)

        ax.legend(handles=h_list)

        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_abs_a.pdf", format='pdf')
        plt.close()