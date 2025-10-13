
import numpy as np
import matplotlib.pyplot as plt
from eelib.consts import pi, kFAu, rtol, atol
from eelib.loop import loop
import time


class deriv_grid:
    def __init__(self, R, B, dk, mu, k = kFAu, amp=1., grid_size = 9, ang_lim = 0.1):
        self.l_calc = loop(R, B, dk, mu, k, amp) # I should probably do this with inheritance
        self.grid_size = grid_size
        self.dlim = self.l_calc.psi_prime_0_max()
        
        # form the derivative grid for t = 0
        x = np.linspace(pi-ang_lim, ang_lim, num=self.grid_size, dtype = complex)
        x = np.cos(x) * self.dlim 
        y = np.linspace(pi-ang_lim, ang_lim, num=self.grid_size, dtype = complex)
        y = np.cos(y) * self.dlim
        xx, yy = np.meshgrid(x, y)
        d0_grid = xx + 1j * yy
        self.d0_grid = d0_grid

    #this just creates a grid of values for various psi'_0 values
    # n -- 
    # pr -- 
    # method, rtol (relative tolerance), atol (absolute tolerance)
    # trim -- needs to be even, every x point is used for the plots
    # to_plot -- list of strings with choises for plotting

    def derivGrid(self, n = 200, pr=1.0, method = 'RK45', rtol = rtol, atol = atol, trim = 16, to_plot = ['er','ed','0d', '0r', '0x']):

        self.trim = trim
        self.to_plot = to_plot

        start_time = time.time()

        plot_code = -1
        if 'er' in to_plot: plot_code *= -1
        if 'ed' in to_plot: plot_code *= 2
        if '0d' in to_plot: plot_code *= 3
        if '0r' in to_plot: plot_code *= 5

        # place for our solutions
        '''
        u, l -- upper, lower
        e -- with ee-interaction
        0 -- without ee-interaction
        d -- full calculation
        r -- piecewise, restored power
        f -- envelope-following
        m -- modified-library envelope-following
        x -- exact
        tc, ta, tr -- time for calculated, abs triggered, real triggered
        yc, ya, yr -- value for calculated, abs triggered, real triggered
        '''

        if 'er' in to_plot: s_grid_er_u = []
        if 'er' in to_plot: s_grid_er_l = []
        if 'ed' in to_plot: s_grid_ed_u = []
        if 'ed' in to_plot: s_grid_ed_l = []
        if 'ef' in to_plot: s_grid_ef_u = []
        if 'ef' in to_plot: s_grid_ef_l =[]
        if 'em' in to_plot: s_grid_em_u = []
        if 'em' in to_plot: s_grid_em_l = []

        if '0d' in to_plot: s_grid_0d_u = []
        if '0d' in to_plot: s_grid_0d_l = []
        if '0r' in to_plot: s_grid_0r_u = []
        if '0r' in to_plot: s_grid_0r_l = []
        if '0f' in to_plot: s_grid_0f_u =[]
        if '0f' in to_plot: s_grid_0f_l = []
        if '0m' in to_plot: s_grid_0m_u = []
        if '0m' in to_plot: s_grid_0m_l = []

        if '0x' in to_plot: s_grid_0x_u_tc = []
        if '0x' in to_plot: s_grid_0x_l_tc = []
        if '0x' in to_plot: s_grid_0x_u_tcm = []
        if '0x' in to_plot: s_grid_0x_l_tcm = []
        if '0x' in to_plot: s_grid_0x_u_ta = []
        if '0x' in to_plot: s_grid_0x_l_ta = []
        if '0x' in to_plot: s_grid_0x_u_tr = []
        if '0x' in to_plot: s_grid_0x_l_tr = []
        if '0x' in to_plot: s_grid_0x_u_yc = []
        if '0x' in to_plot: s_grid_0x_l_yc = []
        if '0x' in to_plot: s_grid_0x_u_ycm = []
        if '0x' in to_plot: s_grid_0x_l_ycm = []
        if '0x' in to_plot: s_grid_0x_u_ya = []
        if '0x' in to_plot: s_grid_0x_l_ya = []
        if '0x' in to_plot: s_grid_0x_u_yr = []
        if '0x' in to_plot: s_grid_0x_l_yr = []

        print('Begin grid build: ', time.time() - start_time)
        print('Plot Code:', to_plot, plot_code)
        
        # solve for each point on the grid
        for i in range(self.grid_size):

            if 'er' in to_plot: s_grid_er_u.append([])
            if 'er' in to_plot: s_grid_er_l.append([])
            if 'ed' in to_plot: s_grid_ed_u.append([])
            if 'ed' in to_plot: s_grid_ed_l.append([])
            if 'ef' in to_plot: s_grid_ef_u.append([])
            if 'ef' in to_plot: s_grid_ef_l.append([])
            if 'em' in to_plot: s_grid_em_u.append([])
            if 'em' in to_plot: s_grid_em_l.append([])

            if '0d' in to_plot: s_grid_0d_u.append([])
            if '0d' in to_plot: s_grid_0d_l.append([])
            if '0r' in to_plot: s_grid_0r_u.append([])
            if '0r' in to_plot: s_grid_0r_l.append([])
            if '0f' in to_plot: s_grid_0f_u.append([])
            if '0f' in to_plot: s_grid_0f_l.append([])
            if '0m' in to_plot: s_grid_0m_u.append([])
            if '0m' in to_plot: s_grid_0m_l.append([])

            if '0x' in to_plot: 
                s_grid_0x_u_tc.append([])
                s_grid_0x_l_tc.append([])
                s_grid_0x_u_tcm.append([])
                s_grid_0x_l_tcm.append([])
                s_grid_0x_u_ta.append([])
                s_grid_0x_l_ta.append([])
                s_grid_0x_u_tr.append([])
                s_grid_0x_l_tr.append([])
                s_grid_0x_u_yc.append([])
                s_grid_0x_l_yc.append([])
                s_grid_0x_u_ycm.append([])
                s_grid_0x_l_ycm.append([])
                s_grid_0x_u_ya.append([])
                s_grid_0x_l_ya.append([])
                s_grid_0x_u_yr.append([])
                s_grid_0x_l_yr.append([])
            
            print("i = ", i, "of ", self.grid_size-1, "time: ", time.time() - start_time)

            for j in range(self.grid_size):
                self.l_calc.setDeriv(self.d0_grid[i,j])
                self.l_calc.solve_ivp(n = n, percent_range = pr, method = method, rtol = rtol, atol = atol, solve=plot_code)

                print("Done, i, j = ", i,j, "time: ", time.time() - start_time)

                if 'er' in to_plot: s_grid_er_u[i].append(self.l_calc.solu)
                #if 'er' in to_plot: s_grid_er_l[i].append(self.l_calc.soll)
                if 'ed' in to_plot: s_grid_ed_u[i].append(self.l_calc.solu_d)
                #if 'ed' in to_plot: s_grid_ed_l[i].append(self.l_calc.soll_d)
                if 'ef' in to_plot: s_grid_ef_u[i].append(self.l_calc.solu_f)
                #if 'ef' in to_plot: s_grid_ef_l[i].append(self.l_calc.soll_f)
                if 'em' in to_plot: s_grid_em_u[i].append(self.l_calc.solu_m)
                #if 'em' in to_plot: s_grid_em_l[i].append(self.l_calc.soll_m)

                if '0d' in to_plot: s_grid_0d_u[i].append(self.l_calc.solu0)
                #if '0d' in to_plot: s_grid_0d_l[i].append(self.l_calc.soll0)
                if '0r' in to_plot: s_grid_0r_u[i].append(self.l_calc.solu0_r)
                #if '0r' in to_plot: s_grid_0r_l[i].append(self.l_calc.soll0_r)
                if '0f' in to_plot: s_grid_0f_u[i].append(self.l_calc.solu0_f)
                #if '0f' in to_plot: s_grid_0f_l[i].append(self.l_calc.soll0_f)
                if '0m' in to_plot: s_grid_0m_u[i].append(self.l_calc.solu0_m)
                #if '0m' in to_plot: s_grid_0m_l[i].append(self.l_calc.soll0_m)

                if '0x' in to_plot:
                    T_arr = self.l_calc.find_period_shift_exact() # fast, slow, pos, neg
                    #stm1_ex, stm2_ex = self.l_calc.find_start_mid_exact()

                    # I need start times for the real amplitude following cases
                    # they are from s_grid_0x_u_tc[np.argmax(abs(s_grid_0x_u_yc))]
                    # but then need to caclulate backwards
                    #xTmin, xTmax = self.l_calc.find_slow_oscillations_start()
                    xT = self.l_calc.find_real_env_start()

                    s_grid_0x_u_tc[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stu_ex, T_arr[0]))
                    s_grid_0x_l_tc[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stl_ex, T_arr[0]))
                    s_grid_0x_u_tcm[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, xT, T_arr[2]))
                    s_grid_0x_l_tcm[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, xT, T_arr[3]))
                    #s_grid_0x_u_tcs[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stu_ex, T_arr[2]))
                    #s_grid_0x_l_tcs[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stl_ex, T_arr[2]))
                    #s_grid_0x_u_tcd[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stu_ex, T_arr[3]))
                    #s_grid_0x_l_tcd[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stl_ex, T_arr[3]))

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
                    s_grid_0x_u_ya[i].append(self.l_calc.psij(s_grid_0x_u_ta[i][j]))
                    s_grid_0x_l_ya[i].append(self.l_calc.psij(s_grid_0x_l_ta[i][j]))
                    s_grid_0x_u_yr[i].append(self.l_calc.psij(s_grid_0x_u_tr[i][j]))
                    s_grid_0x_l_yr[i].append(self.l_calc.psij(s_grid_0x_l_tr[i][j]))
                
        if 'er' in to_plot: self.s_grid_er_u = s_grid_er_u
        #if 'er' in to_plot: self.s_grid_er_l = s_grid_er_l
        if 'ed' in to_plot: self.s_grid_ed_u = s_grid_ed_u
        #if 'ed' in to_plot: self.s_grid_ed_l = s_grid_ed_l
        if 'ef' in to_plot: self.s_grid_ef_u = s_grid_ef_u
        #if 'ef' in to_plot: self.s_grid_ef_l = s_grid_ef_l
        if 'em' in to_plot: self.s_grid_em_u = s_grid_em_u
        #if 'em' in to_plot: self.s_grid_em_l = s_grid_em_l

        if '0d' in to_plot: self.s_grid_0d_u = s_grid_0d_u
        #if '0d' in to_plot: self.s_grid_0d_l = s_grid_0d_l
        if '0r' in to_plot: self.s_grid_0r_u = s_grid_0r_u
        #if '0r' in to_plot: self.s_grid_0r_l = s_grid_0r_l
        if '0f' in to_plot: self.s_grid_0f_u = s_grid_0f_u
        #if '0f' in to_plot: self.s_grid_0f_l = s_grid_0f_l
        if '0m' in to_plot: self.s_grid_0m_u = s_grid_0m_u
        #if '0m' in to_plot: self.s_grid_0m_l = s_grid_0m_l

        if '0x' in to_plot: self.s_grid_0x_u_tc = s_grid_0x_u_tc
        if '0x' in to_plot: self.s_grid_0x_l_tc = s_grid_0x_l_tc
        if '0x' in to_plot: self.s_grid_0x_u_tcm = s_grid_0x_u_tcm
        if '0x' in to_plot: self.s_grid_0x_l_tcm = s_grid_0x_l_tcm
        if '0x' in to_plot: self.s_grid_0x_u_ta = s_grid_0x_u_ta
        if '0x' in to_plot: self.s_grid_0x_l_ta = s_grid_0x_l_ta
        if '0x' in to_plot: self.s_grid_0x_u_tr = s_grid_0x_u_tr
        if '0x' in to_plot: self.s_grid_0x_l_tr = s_grid_0x_l_tr
        if '0x' in to_plot: self.s_grid_0x_u_yc = s_grid_0x_u_yc
        if '0x' in to_plot: self.s_grid_0x_l_yc = s_grid_0x_l_yc
        if '0x' in to_plot: self.s_grid_0x_u_ycm = s_grid_0x_u_ycm
        if '0x' in to_plot: self.s_grid_0x_l_ycm = s_grid_0x_l_ycm
        if '0x' in to_plot: self.s_grid_0x_u_ya = s_grid_0x_u_ya
        if '0x' in to_plot: self.s_grid_0x_l_ya = s_grid_0x_l_ya
        if '0x' in to_plot: self.s_grid_0x_u_yr = s_grid_0x_u_yr
        if '0x' in to_plot: self.s_grid_0x_l_yr = s_grid_0x_l_yr

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
    def calc_period(self, nn=40, filename = None):

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


        if filename is not None: 
            file = open(filename, 'a')

        for i1, ii in enumerate(hand_list):
            print(hand_string[i1])
            if filename is not None: file.write(hand_string[i1])
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
                    if filename is not None: file.write("For array:", jj, kk)


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
                    if filename is not None: file.write('Max:', np.max(arr[arr_minmax]), "Min:", np.min(arr[arr_minmax]))


                    print('Difference: ')
                    if filename is not None: file.write('Difference: ')
                    for i in range(len(arr_t[arr_minmax])-1):
                        print(arr_t[arr_minmax][i+1]-arr_t[arr_minmax][i])
                        if filename is not None: file.write(arr_t[arr_minmax][i+1]-arr_t[arr_minmax][i])
        if filename is not None: 
            file.close()