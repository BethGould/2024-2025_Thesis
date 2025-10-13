import numpy as np
import matplotlib.pyplot as plt
from eelib.consts import pi, kFAu, rtol, atol
from eelib.loop import loop
import time


class grid_fast_osc:
    def __init__(self, R, B, dk, mu, k = kFAu, amp=1., grid_size = 9, ang_lim = 0.1):
        self.l_calc = loop(R, B, dk, mu, k, amp) # I should probably do this with inheritance
        self.grid_size = grid_size

        self.dlim = self.l_calc.psi_prime_0_max()
        self.ang_lim = ang_lim
        self.makeDerivPoints()

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

    #def makeGridPoints(self, m_min, m_max, k_min, k_max, a_min, a_max, mu_min, mu_max, num = 10):
    def makeGridPoints(self, a_min, a_max, mu_min, mu_max, num = 10, num_m = 10):
        #self.mfp = np.linspace(m_min, m_max, num=num)
        #self.ew = np.linspace(k_min, k_max, num=num)
        self.nls = np.logspace(mu_min, mu_max, num=num_m)
        self.amp = np.linspace(a_min, a_max, num=num)
        self.num = num
        self.num_m = num_m

    def gridFastOsc(self, n = 20, method = 'RK45', rtol = rtol, atol = atol):

        # Timing
        start_time = time.time()
        print('Begin grid build: ', time.time() - start_time)

        # Initialize grids
        #num_m = self.num
        #num_k = self.num
        num_mu = self.num_m
        num_a = self.num
        num_d = self.grid_size

        # need to deterimine how to store the starting derivative as a function of the other parameters
        #magnetic_field_period = self.mfp
        #electron_wavenumber = self.ew
        nonlinearity_strength = self.nls
        starting_amplitude = self.amp
        # d0_grid[idr,idi]
        #starting_derivative_real = self.dr
        #starting_derivative_imag = self.di

        # I may need Monte Carlo .... probably

        #fast_oscillation_period = np.zeros((num_m, num_k, num_mu, num_a, num_d, num_d))
        #fast_oscillation_amplitude = np.zeros((num_m, num_k, num_mu, num_a, num_d, num_d))
        fast_oscillation_period = np.zeros((num_mu, num_a, num_d, num_d))
        fast_oscillation_amplitude = np.zeros((num_mu, num_a, num_d, num_d))

        # solve for each point on the grid
        #for im in range(num_m):
        #    for ik in range(num_k):
        for imu in range(num_mu):
            for ia in range(num_a):    
                self.l_calc.update_params(#B=magnetic_field_period[im], dk=electron_wavenumber[ik], 
                                                  mu=nonlinearity_strength[imu], amp=starting_amplitude[ia])
                print(imu, ia, time.time() - start_time)
                # And for the derivative
                for idr in range(num_d):
                    for idi in range(num_d):
                        # calculate derivative from our parameters if it is not as given
                        self.l_calc.setDeriv(self.d0_grid[idr,idi])
                        #print(self.l_calc.T_fast * 2, self.l_calc.A_max)
                        fast_oscillation_period[imu,ia,idr,idi] = self.l_calc.T_fast * 2
                        fast_oscillation_amplitude[imu,ia,idr,idi] = self.l_calc.A_max                

        self.fast_osc_t = fast_oscillation_period
        self.fast_osc_a = fast_oscillation_amplitude

        print("Done grid build: ", time.time() - start_time)


# ---------

# plotting the grid
    def plot_real(self, i, j, k, R, B, mu, no):

        to_plot = self.to_plot

        # currently the below parameters are used for labeling the graph, 
        # and should be shorter than the actual parameters
        #k = self.l_calc.dk
        #R = self.l_calc.R
        #B = self.l_calc.B
        #mu = self.l_calc.mu

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
            tu0xr = np.real(self.s_grid_0x_u_tr[i][j])
            tl0xr = np.real(self.s_grid_0x_l_tr[i][j])
            su0xr = np.real(self.s_grid_0x_u_yr[i][j])
            sl0xr = np.real(self.s_grid_0x_l_yr[i][j])
            tu0xcm = np.real(self.s_grid_0x_u_tcm[i][j])
            tl0xcm = np.real(self.s_grid_0x_l_tcm[i][j])
            su0xcm = np.real(self.s_grid_0x_u_ycm[i][j])
            sl0xcm = np.real(self.s_grid_0x_l_ycm[i][j])
        
        #--PLOTTING--
        #absolute value triggered envelope
        fig, ax = plt.subplots()
        ax.set_ylabel('Real Part of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, abs triggered, dk={k}, R={R}, B={B}, \u03BC={mu}")

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
            line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'exact solution')
            line10, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'exact solution')
            #line11, = ax.plot(tu0xcm, su0xcm, color = 'purple', label = 'exact solution')
            #line12, = ax.plot(tl0xcm, sl0xcm, color = 'purple', label = 'exact solution')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_real_a.pdf", format='pdf')
        plt.close()

        # real triggered envelope
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
            line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'exact solution')
            line10, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'exact solution')
            #line11, = ax.plot(tu0xcm, su0xcm, color = 'purple', label = 'exact solution')
            #line12, = ax.plot(tl0xcm, sl0xcm, color = 'purple', label = 'exact solution')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_real_r.pdf", format='pdf')
        plt.close()

        # real triggered envelope with triggered exact solution
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
            line9, = ax.plot(tu0xr, su0xr, color = 'purple', label = 'exact solution')
            line10, = ax.plot(tl0xr, sl0xr, color = 'purple', label = 'exact solution')
            #line11, = ax.plot(tu0xcm, su0xcm, color = 'black', label = 'exact solution')
            #line12, = ax.plot(tl0xcm, sl0xcm, color = 'gray', label = 'exact solution')
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
        if 'er' in to_plot: tuera = np.abs(self.s_grid_er_u[i][j]['t_events'][0][0::self.trim])
        if '0r' in to_plot: tu0ra = np.abs(self.s_grid_0r_u[i][j]['t_events'][0][0::self.trim])
        if 'ed' in to_plot: tueda = np.abs(self.s_grid_ed_u[i][j]['t_events'][0][0::self.trim])
        if '0d' in to_plot: tu0da = np.abs(self.s_grid_0d_u[i][j]['t_events'][0][0::self.trim])
        if 'er' in to_plot: tlera = np.abs(self.s_grid_er_u[i][j]['t_events'][0][1::self.trim])
        if '0r' in to_plot: tl0ra = np.abs(self.s_grid_0r_u[i][j]['t_events'][0][1::self.trim])
        if 'ed' in to_plot: tleda = np.abs(self.s_grid_ed_u[i][j]['t_events'][0][1::self.trim])
        if '0d' in to_plot: tl0da = np.abs(self.s_grid_0d_u[i][j]['t_events'][0][1::self.trim])

        # Try to also plot the versions which are calculated
        if 'er' in to_plot: tuerc = np.abs(self.s_grid_er_u[i][j]['t'])
        if '0r' in to_plot: tu0rc = np.abs(self.s_grid_0r_u[i][j]['t'])
        if 'ed' in to_plot: tuedc = np.abs(self.s_grid_ed_u[i][j]['t'])
        if '0d' in to_plot: tu0dc = np.abs(self.s_grid_0d_u[i][j]['t'])
        
        #value arrays
        if 'er' in to_plot: suera = np.abs(self.s_grid_er_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'ed' in to_plot: sueda = np.abs(self.s_grid_ed_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'er' in to_plot: slera = np.abs(self.s_grid_er_u[i][j]['y_events'][0][1::self.trim, 0])
        if 'ed' in to_plot: sleda = np.abs(self.s_grid_ed_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0r' in to_plot: su0ra = np.abs(self.s_grid_0r_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0d' in to_plot: su0da = np.abs(self.s_grid_0d_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0r' in to_plot: sl0ra = np.abs(self.s_grid_0r_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0d' in to_plot: sl0da = np.abs(self.s_grid_0d_u[i][j]['y_events'][0][1::self.trim, 0])

        # Try to also plot the versions which are calculated
        if 'er' in to_plot: suerc = np.abs(self.s_grid_er_u[i][j]['y'][0])
        if '0r' in to_plot: su0rc = np.abs(self.s_grid_0r_u[i][j]['y'][0])
        if 'ed' in to_plot: suedc = np.abs(self.s_grid_ed_u[i][j]['y'][0])
        if '0d' in to_plot: su0dc = np.abs(self.s_grid_0d_u[i][j]['y'][0])

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
        #if 'ed' in to_plot: 
        #    line3, = ax.plot(tuedc, suedc, color = 'orange', label = 'with e-e interaction')
        #    h_list.append(line3)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0rc, su0rc, color = 'green', label = 'without e-e interaction')
            h_list.append(line5)
        #if '0d' in to_plot: 
        #    line7, = ax.plot(tu0dc, su0dc, color = 'blue', label = 'without e-e interaction')
        #    h_list.append(line7)
        if '0x' in to_plot: 
            line9, = ax.plot(tu0xa, su0xa, color = 'purple', label = 'exact solution')
            line10, = ax.plot(tl0xa, sl0xa, color = 'purple', label = 'exact solution')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        #plt.show()
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
            #line11, = ax.plot(tu0xam, su0xam, color = 'purple', label = 'exact solution')
            #line12, = ax.plot(tl0xam, sl0xam, color = 'purple', label = 'exact solution')
            h_list.append(line9)

        ax.legend(handles=h_list)

        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_abs_a.pdf", format='pdf')
        plt.close()

# ----------------------------------
    # This is for calculating the period of the slow oscillations
    # I will need to construct something to instead function fit with Scipy
    # They are periodic, but not sinisoidal

    # nn is the number of slices for dividing the function
    # needs to be high enough to minimize the boundary size and ensure that all extreema are found
    # but still small enough to contain many points
    # right now the data is printed on the screen, not saved to file
    def calc_period(self, nn=40):
        hand_list = []
        hand_string = []
        if 'er' in self.to_plot:
            hand_list.append(self.s_grid_er_u)
            hand_string.append('ee, recovered')
        if 'ed' in self.to_plot:
            hand_list.append(self.s_grid_ed_u)
            hand_string.append('ee, decreasing')
        if '0r' in self.to_plot:
            hand_list.append(self.s_grid_0r_u)
            hand_string.append('0, recovered')
        if '0d' in self.to_plot:
            hand_string.append('0, decreasing')
            hand_list.append(self.s_grid_0d_u)

        for i1, ii in enumerate(hand_list):
            print(hand_string[i1])
            for jj in range(self.grid_size):
                for kk in range(self.grid_size):

                    # choose the array to evaluate
                    arr = np.real(ii[jj][kk]['y_events'][1][::2, 0])
                    arr_t = np.real(ii[jj][kk]['t_events'][1][::2])

                    leng = len(arr)

                    # max and min storage arrays
                    ind_arr_max = []
                    ind_arr_max_s = []
                    ind_arr_min = []
                    ind_arr_min_s = []

                    # fill the storage arrays
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

                    print("For array:", jj, kk)

                    # print absolute maxima and minima
                    '''
                    for i in range(nn):
                        if ind_arr_max[i] in ind_arr_max_s:
                            print(arr_t[ind_arr_max[i]], arr[ind_arr_max[i]])

                    for i in range(nn):
                        if ind_arr_min[i] in ind_arr_min_s:
                            print(arr_t[ind_arr_min[i]], arr[ind_arr_min[i]])
                    '''

                    # calculate extrema
                    ind_arr_max_int = np.intersect1d(ind_arr_max, ind_arr_max_s)
                    ind_arr_min_int = np.intersect1d(ind_arr_min, ind_arr_min_s)
                    arr_minmax = np.union1d(ind_arr_min_int, ind_arr_max_int)

                    # print extreema
                    #print(arr_t[arr_minmax])
                    print('Max:', np.max(arr[arr_minmax]), "Min:", np.min(arr[arr_minmax]))


                    print('Difference: ')
                    for i in range(len(arr_t[arr_minmax])-1):
                        print(arr_t[arr_minmax][i+1]-arr_t[arr_minmax][i])