# loop class 
# 


# Error handling has not been added to this class
# I also need to play around with the state setting variables
# and clearing solutions on changing parameters

# finish certain commenting (separation)


#--LIBRARIES----------------
import numpy as np
import scipy

from eelib.consts import pi, kFAu, R_max, B_max, phi0inv, rtol, atol, DK_min
from eelib.deriv_functions import psi_deriv, psi_deriv_old
from eelib.events import deriv_amp, deriv_real
from eelib.k_M_models import pred_fast_t, pred_slow_t
#from eelib.ivp_2 import solve_ivp_mod

#--TABLE OF CONTENTS--------

# 1 ----- Defining Constants
#    __init__(R, B, dk, mu, k = kFAu, amp=1.)
#    update_params(R=-1., B=-1., dk=-1., mu=-1., k=-1., amp=-1.)
#    calcInit()
#    setDeriv(p_prime, n = 20)

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
#    psij0(x)
#    psi_prime_0()
#    psi_prime_0_max()

# 3 ----- ODE solver wrappers
#    solve_ivp(n=20, percent_range = 1.0, method = 'RK45', rtol = rtol, atol = atol, solve = 30)
#    ivp_solver_steps(t0, tf, y0, yp0, n=00, ee_int=True, m=4, method = 'RK45', rtol = rtol, atol = atol, fullSol = False)
#    call_ivp_solver(t0, tf, y0, yp0, n=0, fullSol = True, t0_start = False, ee_int= True, method = 'RK45', rtol = rtol, 
#                           atol=atol, first_step = None, max_step = np.inf, t_eval = None)

# 4 ----- Fast oscillation calculations
#    find_fast_oscillations(n=20)
#    find_start_exact()
#    find_start_fast_oscillations(n, sol)
#    find_period_fast_oscillations(n, sol)
#    find_amplitude_fast_oscillations(sol)

# 5 ----- Slow oscillation and coupled calculations
#    find_period_shift_exact()
#    find_slow_oscillations_start()
#    find_real_env_start()
#    find_t_points(n_points, t_max, t_start, T)

# 6 ----- BVP matching phases

# 7 ----- BVP shooting

# 8 ----- Finding current
#    current()
#    current_calc(psi, psi_pr)



#--VARIABLES-----

#   R (m)
#   k (hz), k + dk
#   B (T)
#   mu 
#   amp

#   lngt     = 2 * pi * self.R
#   period_k = 2 * self.R * self.k
#   T0       = 2. * pi / self.k     
#   M        = self.B*self.R*phi0inv
#   denrem   = self.rem(self.R*self.M)
#   aj
#   bj
#   aj0      -- for matched exact solution
#   bj0      -- for matched exact solution

#   psi0_deriv_0 -- psi' for time 0

#   T_fast
#   A_max
#   T_fast0
#   T_fast_ex

# ---- State indicators ----
#   deriv_set
#   ivp_solved
#   solve   -- int, Indicates which IVPs were solved.
#   bvp_solved

# ---- Solutions to full IVP ---- 
#   solu    -- recovered with ee interaction
#   solu_d  -- decreasing with ee interaction
#   solu0   -- decreasing without ee
#   solu0_r -- recovered without ee
#   solu_f  -- 
#   solu_m  --  

# ---- Starting Times ----
# with or without ee interaction, or exact solution
#   stl
#   sth
#   stl0
#   sth0
#   stl_ex
#   stu_ex


#--FUNCTIONS TO CALL--------

# 1 ----- Defining Constants
#    __init__(R, B, dk, mu, k = kFAu, amp=1.)
#    update_params(R=-1., B=-1., dk=-1., mu=-1., k=-1., amp=-1.)
#    setDeriv(p_prime, n = 20)

# 2 ----- Analytic calculations of the case without e-e interaction
#    psij(x)           -- returns exact solution of psi without e-e interaction, for a provided derivative
#    psij0(x)          -- returns exact solution of psi without e-e interaction, for the matched solution
#    psi_prime_0_max() -- returns the maximum abs of psi' without e-e interaction, for use in constructing derivative grids

# 3 ----- ODE solver wrappers
#    solve_ivp(n=20, percent_range = 1.0, method = 'RK45', rtol = rtol, atol = atol, solve = 30)
#           -- used for running the ivp for the given ...

# 4 ----- Fast oscillation calculations
# all calculations are automatic when defining parameters

# 5 ----- Slow oscillation and coupled calculations
# may require ivp_solved = True
#    find_period_shift_exact()
#    find_slow_oscillations_start() -- I think this was moved to the below function
#    find_real_env_start()
#    find_t_points(n_points, t_max, t_start, T)

# 6 ----- BVP matching phases

# 7 ----- BVP shooting

# 8 ----- Finding current
# all require ivp_solved = True; needs to be updated
#    current()
#    current_calc(psi, psi_pr)



#--LOOP CLASS---------------
class loop:

    # constructor
    # On creation, takes R, B, dk, mu, k, amp as parameters
    # R is radius as a percentage of R_max
    # B is the magnetic field as a percentage of B_max
    # the wavefunction of the electron is k_el = k + dk/self.R /2.0 
    # k defaults to kFAu (assuming gold as the conductor)
    # psi(0) = amp
    # psi'(0) initially defined based on the known solution for mu = 0
    def __init__(self, R, B, dk, mu, k = kFAu, amp=1.):
        self.R   = R * R_max #R in micrometers
        self.k   = k + dk/self.R /2.0 #dk adds to min k
        self.B   = B * B_max #scales with 130 G
        self.mu  = mu
        self.amp = complex(amp)
        #self.n   = 1e7 #note: +1

        self.deriv_set = False # False = default derivitive, True = chosen derivitive
        self.ivp_solved = False
        self.solve = 0 # Indicates which ivps were solved
        self.bvp_solved = False

        self.solve_mu_0 = True
        
        self.calcInit()

    # the default -1 means that nothing is changed here
    # otherwise, the parameter is updated to the given values
    # all further calculations are the same as above
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
            self.k = k + dk/self.R / 2.0
        elif dk > 0.:
            self.k = kFAu + dk/self.R / 2.0
        elif k > 0.:
            self.k = k + DK_min

        #self.clear_sol()
        self.deriv_set = False # False = default derivitive, True = chosen derivitive
        self.ivp_solved = False
        self.solve = 0 # Indicates which ivps were solved
        self.bvp_solved = False

        self.calcInit()
        
    # Sets a bunch of constants based on the data defining the loop
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

        # I need to be careful here, with the understanding of what is found -- t or k, also with factors of 2
        self.T_fast_mod = pred_fast_t(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_slow_mod = pred_slow_t(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        # I will also need the starting point of the oscillation
        
        self.find_fast_oscillations(20)
        self.T_fast = self.T_fast * 2 # This t is now half the real t
        

    # This function is for setting the initial psi prime value of the loop
    # It is meant to be called numerous times
    
    def setDeriv(self, p_prime, n = 20, method = 'RK45', rtol = rtol, atol = atol):
        self.psi0_deriv_0 = p_prime

        self.aj = self.ajN_calc()
        self.bj = self.bjN_calc()

        #self.clear_sol()
        self.deriv_set = True # False = default derivitive, True = chosen derivitive
        self.ivp_solved = False
        self.solve = 0 # Indicates which ivps were solved
        self.bvp_solved = False

        # I need to be careful here, with the understanding of what is found -- t or k, also with factors of 2
        self.T_fast_mod = pred_fast_t(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_slow_mod = pred_slow_t(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        # I will also need the starting point of the oscillation
        
        self.find_fast_oscillations(n, method = method, rtol = rtol, atol = atol)
        self.T_fast = self.T_fast * 2 # this t is now half the real t, was one fourth before, (corresponds to t of abs plot)

        #print(self.T_fast, self.T_fast_mod, 2 * pi / self.k)
        #print(self.T_slow_mod, 2 * pi / self.M)
        
        #if aj, bk = nan, ....
        

    # 2 ----- Analytic calculations of the case without e-e interaction

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
    
    # for when setting psi prime manually
    def ajN_calc(self):
        den = 1.0 / 2.0 / self.k
        reA = den * (np.imag(self.psi0_deriv_0) - self.M + self.k)
        imA = - den * np.real(self.psi0_deriv_0)

        ret = 0.5*((self.psi0_deriv_0 - 1j * self.M * self.amp)/(1j*self.k)+self.amp)

        return ret
        #return complex(reA,imA)
    def bjN_calc(self):
        den = 1.0 / 2.0 / self.k
        reB = den * (-np.imag(self.psi0_deriv_0) + self.M + self.k)
        imB = - den * np.real(self.psi0_deriv_0)

        ret = 0.5*(self.amp - (self.psi0_deriv_0 - 1j * self.M * self.amp)/(1j*self.k))

        return ret
        #return complex(reB,imB)
    
    #old psij function, estimation of derivative at x=0
    def psij(self, x):
        return self.aj*np.exp(1j*x*(self.k+self.M)) + self.bj*np.exp(1j*x*(-self.k+self.M))
    def psij0(self, x):
        return self.aj0*np.exp(1j*x*(self.k+self.M)) + self.bj0*np.exp(1j*x*(-self.k+self.M))
    def psi_prime_0(self):
        return 1j * (self.aj0*(self.k+self.M)+self.bj0*(-self.k+self.M))
    #this is the max abs of psi' without e-e interaction
    def psi_prime_0_max(self):
        #psi = e^(iMx)*[ae^(ikx)+be^(-ikx)]
        #psi' = iM*psi + ik* e^(iMx)*[ae^(ikx)-be^(-ikx)]
        #abs(psi') = abs[M *(ae^(ikx)+be^(-ikx)) + k* (ae^(ikx)-be^(-ikx)]
        # <= (M+k)*[abs(a)+abs(b)]
        d_max = (self.M+self.k)*(np.abs(self.aj0)+np.abs(self.bj0))
        return d_max
    
    # 3 ---- ODE routines
    def solve_ivp(self, n = 20, percent_range = 1.0, method = 'RK45', rtol = rtol, atol = atol, solve=30):
        #t_range
        #t_evalh = self.find_t_points(n, t_max = self.lngt * percent_range, t_start = self.t_high, T = self.T_fast)
        #t_evall = self.find_t_points(n, t_max = self.lngt * percent_range, t_start = self.t_low, T = self.T_fast)
 
        t0h = self.sth
        tfh = self.lngt * percent_range # t_evalh[-1]
        t0l = self.stl
        tfl = self.lngt * percent_range #t_evall[-1]

        #t_vals = self.find_oscil_old(n, self.lngt * percent_range)

        t0h0 = self.sth0
        tfh0 = self.lngt * percent_range #t_vals[1]
        t0l0 = self.stl0
        tfl0 = self.lngt * percent_range #t_vals[3]


        #calculate y0
        y0h = self.amp
        yp0h = self.psi0_deriv_0
        y0l = self.amp
        yp0l = self.psi0_deriv_0

        y0h0, yp0h0 = self.amp, self.psi0_deriv_0 #self.calc_yval_old(n, t0h0)
        y0l0, yp0l0 = self.amp, self.psi0_deriv_0 #self.calc_yvals_old(n, t0l0)

        # call the solver multiple times  
        # with ee-interaction -- divided into steps (+), original(2), 
        # with predicted k_fast (7)
        # without ee-interaction -- original(3), divided into steps(5)
        if solve > 0:      
            #print('1')
            self.solu = self.ivp_solver_steps(t0h, tfh, y0h, yp0h, n, fullSol = False, ee_int = True, 
                                              method = method, rtol = rtol, atol = atol)
            #self.soll = self.ivp_solver_steps(t0l, tfl, y0l, yp0l, n, fullSol = False, ee_int = True, method = method, rtol = rtol, atol = atol)
        if abs(solve)%2 == 0:
            #print('2')
            self.solu_d = self.call_ivp_solver(t0h, tfh, y0h, yp0h, n, fullSol = False, ee_int = True, 
                                               method = method, rtol = rtol, atol = atol)
            #self.soll_d = self.call_ivp_solver(t0l, tfl, y0l, yp0l, n, fullSol = False, ee_int = True, method = method, rtol = rtol, atol = atol)
        if abs(solve)%3 == 0:
            #print('3')
            self.solu0 = self.call_ivp_solver(t0h0, tfh0, y0h0, yp0h0, n, fullSol = False, ee_int = False, 
                                              method = method, rtol = rtol, atol = atol)
            #self.soll0 = self.call_ivp_solver(t0l0, tfl0, y0l0, yp0l0, n, fullSol = False, ee_int = False, method = method, rtol = rtol, atol = atol)
        if abs(solve)%5 == 0:
            #print('5')
            self.solu0_r = self.ivp_solver_steps(t0h0, tfh0, y0h0, yp0h0, n, fullSol = False, ee_int = False, 
                                                 method = method, rtol = rtol, atol = atol)
            #self.soll0_r = self.ivp_solver_steps(t0l0, tfl0, y0l0, yp0l0, n, fullSol = False, ee_int = False, method = method, rtol = rtol, atol = atol)
        if abs(solve)%7 == 0: 
            #print('7')     
            self.solu_m = self.ivp_solver_steps(t0h, tfh, y0h, yp0h, n, fullSol = False, ee_int = True, 
                                              method = method, rtol = rtol, atol = atol, estimate_k = True)
            #self.soll_f = self.call_ivp_solver(t0l, tfl, y0l, yp0l, n, fullSol = False, ee_int = True, method = method, rtol = rtol, atol = atol)
        #if abs(solve)%11 == 0:      
            #print('11')
        #    self.solu_m = self.call_ivp_solver(t0h, tfh, y0h, yp0h, n, fullSol = False, ee_int = True, 
        #                                       method = method, rtol = rtol, atol = atol)
            #self.soll_m = self.call_ivp_solver(t0l, tfl, y0l, yp0l, n, fullSol = False, ee_int = True, method = method, rtol = rtol, atol = atol)
        

    # This is designed to correct for the drop in power over continuous cycles by just adding it back in
    # calls the solver multiple times for shorter intervals
    def ivp_solver_steps(self, t0, tf, y0, yp0, n=200, ee_int=True, m = 4, method = 'RK45', rtol = rtol, atol = atol, fullSol = False, estimate_k = False):
        # find all of the points we wish to solve for
        if ee_int:
            if estimate_k:
                t_eval_full = self.find_t_points(n, t_max = tf, t_start = t0, T = self.T_fast_mod)
            else:
                t_eval_full = self.find_t_points(n, t_max = tf, t_start = t0, T = self.T_fast)
        else:
            #if estimate_k:
            #    t_eval_full = self.find_t_points(n,t_max = tf, t_start = t0, T = self.T_fast0_mod)
            #else:
            t_eval_full = self.find_t_points(n,t_max = tf, t_start = t0, T = self.T_fast0)

        # desired step size for our solver
        first_step = max_step = t_eval_full[1]-t_eval_full[0]

        #initialization
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

        # limit the growth of the function
        #y_max_ave = abs(y0)

        #print(t_eval_full)

        # solve until the end of the chain
        while tfs <= tf:
            
            # define points to find
            if i+m+1 > len(t_eval_full): 
                t_eval_s = t_eval_full[i:]
                tfs = tf
            else: 
                t_eval_s = t_eval_full[i:i+m] 
                tfs = t_eval_full[i+m]

            #print("Begin:", t0s, "End:", tfs )

            if len(t_eval_s) == 0:
                break 
                #t_eval_s = None
                #first_step=None

            #solve the next step
            sol.append(self.call_ivp_solver(t0s, tfs, y0s, yp0s, fullSol = False, t0_start=t0_start, ee_int = ee_int, method = method, 
                        rtol = rtol, atol = atol, first_step=first_step, max_step = max_step, t_eval = t_eval_s))
            
            #print(sol[-1]['y'])

            #Too short
            if len(sol[-1]['y']) == 0:
                sol.pop()
                i += m
                continue

            # event trigger problems 
            if np.shape(sol[-1]['y_events'][0])[0] < 2:
                sol.pop()
                i += m
                continue

            #find final solutions
            yfs = sol[-1]['y_events'][0][-1,0]
            ypfs = sol[-1]['y_events'][0][-1,1]
            tfs = sol[-1]['t_events'][0][-1]

            y0a = sol[-1]['y_events'][0][0,0]
            y0a2 = sol[-1]['y_events'][0][1,0]

            t0_start = True
            
            # choose the same abs event side
            
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

            # choose the same re event side
            y_event_first = np.real(sol[-1]['y_events'][1][0,0])
            #y_event_second = np.real(sol[-1]['y_events'][0][1,1])
            y_event_last = np.real(sol[-1]['y_events'][1][-1,0])
            #print(y_event_first, y_event_last)

            if y_event_last < 0:
                end_high[1].append(False)
            else:
                end_high[1].append(True)
            if y_event_first < 0:
                start_high[1].append(False)
            else:
                start_high[1].append(True)

            #print(np.shape())


            # rescale for the new run

            #print("Target:", abs(yfs), "First:", abs(y0a), "Second:", abs(y0a2), "Correction:", rto)
            i += m
            t0s = tfs
            y0s = yfs * rto
            yp0s = ypfs * rto

            #print('1:', abs(yfs),abs(y0s), tfs)

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
                #print(len(sol_tev_list[-1]), len(sol_yev_list[-1][:, 0]))
                #print(sol_yev_list)


            sol_out_tev.append(np.concatenate(tuple(sol_tev_list)))
            #print(np.shape(sol_yev_list[0]))
            sol_out_yev.append(np.concatenate(tuple(sol_yev_list), axis = 0))

        #print(sol_tev_list)
        #print(sol_yev_list)

        #print(sol_y_list)
        sol_out_t = np.concatenate(tuple(sol_t_list))
        sol_out_y = np.concatenate(tuple(sol_y_list), axis = 1)
        #print(sol_out_tev)
        #print(sol_out_yev)
        
        sol_out_status = []
        for ev in sol:
            sol_out_status.append(ev['status'])

        #print(sol_out_yev)
        
        return {
            't': sol_out_t,
            'y': sol_out_y,
            'status': sol_out_status,
            't_events': sol_out_tev,
            'y_events': sol_out_yev
        }


    # wrapper for the library call to the IVP solver
    def call_ivp_solver(self, t0, tf, y0, yp0, n = 0, fullSol = True, t0_start=False, ee_int = True, method = 'RK45', 
                        rtol = rtol, atol = atol, first_step=None, max_step = np.inf, t_eval = None):
        #choose solver
        if ee_int:
            y_hand = psi_deriv
        else:
            y_hand = psi_deriv_old
            
        #range, initial values, and parameters
        if t0_start:
            t_range = [t0,tf]
        else: 
            t_range = [0.0, tf] #sequence
        y_0 = np.array([y0, yp0]) #array_like, shape (n,) 
        arg = (self.k, self.B, self.R, self.mu, 1000.) #tuple
        
        #events
        #if fullSol:
        elist = [deriv_amp, deriv_real]
        #else:
        #    elist = None
        
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

    #finding the period and starting point of ...
    # sol2 can be removed to speed up calculations. It is here because T0 doesn't correctly estimate the period. 
    # I need to compare the known and calculated solutions here at some point to figure out the error. 
    # The error in period may be more problematic than it seems.
    # I wonder if rtol and atol can fix them, or choosing a better method ... 
    def find_fast_oscillations(self, n=20, method = 'RK45', rtol = rtol, atol = atol):
        #choose percent_range
        #n = 20        # how many ocillations we want to calculate 
        maxX = n*self.T0
        per_range = maxX / self.lngt

        sol1 = self.call_ivp_solver(0.0, self.lngt *per_range, self.amp, self.psi0_deriv_0, 
                                    method = method, rtol = rtol, atol = atol) # find the first n cycles
        if self.solve_mu_0: 
            sol2 = self.call_ivp_solver(0.0, self.lngt *per_range, self.amp, self.psi0_deriv_0, ee_int = False,
                                    method = method, rtol = rtol, atol = atol) # find the first n cycles

        #find the period of the events
        self.T_fast = self.find_period_fast_oscillations(n, sol1)
        self.A_max = self.find_amplitude_fast_oscillations(sol1)
        if self.solve_mu_0: self.T_fast_0_calc = self.find_period_fast_oscillations(n, sol2)
        self.T_fast0 = self.T0
        self.T_fast_ex = self.T0

        #print('sol1[])

        #find the starting point of the events
        start_ee = self.find_start_fast_oscillations(n, sol1)
        if self.solve_mu_0: 
            start_0 =  self.find_start_fast_oscillations(n, sol2)
        #self.find_target_start_abs_deriv(n)
        #self.sol['t_events']
        #self.sol['y_events']

        self.stl = start_ee[0]
        self.sth = start_ee[1]
        if self.solve_mu_0:
            self.stl0 = start_0[0]
            self.sth0 = start_0[1]

        self.stl_ex, self.stu_ex = self.find_start_exact()

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

    #The first time period to start at for envelope following 
    #if psi'(0) is as given
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
        
    def find_period_fast_oscillations(self,n,sol):
        t = sol['t_events'][0]
        t2 = t[1:n]
        t1 = t[0:n-1]
        td = t2 - t1
        dt = np.average(td)
        return dt
    
    def find_amplitude_fast_oscillations(self,sol):
        amp = np.abs(sol['y_events'][0][:, 0])
        #print(amp)
        #ave = np.average(t)
        max = np.max(amp)
        #dt = np.average(td)
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
        #print(x1)
        x1 = np.real(x1)
        dx = pi / den
        T = pi / den
        nT = np.floor(x1 / (2. * T))
        x2 = x1 - nT * 2. * T
        x3 = x2 + T
        if x3 > 2.* T:
            x3 = x3 - 2. * T

        # return min, max
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
        #y = self.psij(xt)
        #if y < 0: xT = xTmin
        #else: xT = xTmax

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


    #gives the target real(psi'(0)) and its corresponding t start values
    #for upper and lower envelope
    def find_t_points(self, n_points, t_max, t_start, T):
        p0 = t_start
        nT = np.floor((t_max - t_start) / T)  
        count = np.floor(nT / (n_points - 1)) 
        
        pf = t_start + count * (n_points -1) * T
        x1 = np.linspace(p0, pf, n_points)
        return x1

    
    # 5 -- BVP routines
    #This is what is called the shooting method for BVP
    #I will separate it into two versions, for positive and negative mu values

    
    # 6 -- I for exact solution and new solution
    def current(self):
        ajsq = np.square(np.real(self.aj)) + np.square(np.imag(self.aj))
        bjsq = np.square(np.real(self.bj)) + np.square(np.imag(self.bj))
        return ajsq - bjsq
    def current_calc(self, psi, psi_pr):
        t1 = 1/ self.k / 1j
        t2 = -2j * self.B * self.R * phi0inv
        return t1*(np.conj(psi)*psi_pr - psi*np.conj(psi_pr) + t2*np.conj(psi)*psi)