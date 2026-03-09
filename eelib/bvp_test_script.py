# bvp_test_script.py

# Author: Elizabeth Gould
# Date Last Edit: 08.03.2026

import numpy as np
from scipy import optimize
from eelib.consts import pi, kFAu, phi0inv, B_max, R_max
from eelib.k_M_models_ivp import pred_fast_k, pred_slow_k_v3, pred_fast_k_true
#from eelib.k_M_models_ivp import deriv_fast_k, deriv_slow_k_v3
from eelib.fitted_functions import find_amp, find_root_start
from eelib.bvp_rootfinder_functions import find_root_both
from eelib.loop import loop
import matplotlib.pyplot as plt
from eelib.bvp_rootfinder_functions import function_wrapper_bvp, k_calc_0, M_calc_0

# This is a script to match the nonlinear solution at the boundary using the model for k and M
# and output results on the screen. 
# The root of our matching equation found is effectively random (chaotic, not random). There are many. 
# The program starts looking for roots around the linear solution, but depending on the algorithm, the solution may 
# not be the closest one.
def test_match(dk, R2, B2, mu):
    R  = R2 * R_max
    B  = B2 * B_max

    # find the roots
    deriv_psi_o, xs_sol, deriv_psi, dpsi_sol = find_root_both(dk, R, B, mu, model_k=pred_fast_k, ratio = 10, check_solution = True)
    print("Root of linear solution:", deriv_psi_o, xs_sol)
    print("Root of nonlinear solution:", deriv_psi)
    print("Value of matching function at this root (should be 0):", dpsi_sol)

    # our outputs separate the real and imaginary parts of our psi_0 derivative 
    # as if they were separate variables. we need to reunite them as a single complex number.
    #deriv_psi = dpsi0[0] + 1j * dpsi0[1]
    #deriv_psi_o = xs[0]+1j*xs[1]

    # Define a loop object for use with this root
    l_calc = loop(R2, B2, dk, mu)

    print("") # newline

    # a, b for calculated and fit linear solutions
    # they are typically very close
    print("aj, bj", l_calc.aj, l_calc.bj)
    l_calc.setDeriv(deriv_psi_o)
    print("aj, bj", l_calc.aj, l_calc.bj)

    # now for the nonlinear bvp solution
    l_calc.setDeriv(deriv_psi)

    print("") # newline

    # check to see how good the match is, here using the model for k and M, not the ode solver
    print("Exact solution endpoint:", l_calc.psij0(2*pi*R2 * R_max))
    print("New solution endpoint:", l_calc.psij_pred(2*pi*R2 * R_max), np.abs(l_calc.psij_pred(2*pi*R2 * R_max)))

    # linear and nonlinear coefficients 
    print("aj, bj:", l_calc.aj, l_calc.bj)
    print("aj0, bj0:", l_calc.aj0, l_calc.bj0)

    print("") # newline

    # current calculations
    cur_new = l_calc.current_new()
    cur_old = l_calc.current_old()

    print("New and old current estimates:", cur_new, cur_old)

    # Now all outputs are graphical rather than text based

    # ode solver with our solutions, for visual analysis of our matching
    l_calc.solve_ivp(n = 1000, solve=1)
    solu = l_calc.solu

    T_arr = l_calc.find_period_shift_exact() # fast, slow, pos, neg
    xT = l_calc.find_real_env_start()
    trim = 16
    n = 1000

    sutc = l_calc.find_t_points(n, l_calc.lngt, l_calc.stu_ex, T_arr[0])
    sltc = l_calc.find_t_points(n, l_calc.lngt, l_calc.stl_ex, T_arr[0])
    #sutcm = l_calc.find_t_points(n, l_calc.lngt, xT, T_arr[2])
    #sltcm = l_calc.find_t_points(n, l_calc.lngt, xT, T_arr[3])
    suyc = l_calc.psij0(sutc)
    slyc = l_calc.psij0(sltc)
    #suycm = l_calc.psij0(sutcm)
    #slycm = l_calc.psij0(sltcm)
    M_pred = 2*pi / l_calc.T_slow_mod

    A_pred = find_amp(solu)
    sol_t = np.real(solu['t']) 
    ind_st = find_root_start(solu)
    t0_pred = sol_t[ind_st]
    t_pred = sol_t
    y_pred = A_pred * np.sin(M_pred * (t_pred-t0_pred))

    #position arrays
    #estimated, triggered buy abs, triggered by real (both) -- c, a, r
    tuerr = np.real(solu['t_events'][1][0::trim])
    tlerr = np.real(solu['t_events'][1][1::trim])
    tuera = np.real(solu['t_events'][0][0::trim])
    tlera = np.real(solu['t_events'][0][1::trim])
            
    #value arrays
    suerr = np.real(solu['y_events'][1][0::trim, 0])
    slerr = np.real(solu['y_events'][1][1::trim, 0])
    suera = np.real(solu['y_events'][0][0::trim, 0])
    slera = np.real(solu['y_events'][0][1::trim, 0])

    tu0xc = np.real(sutc)
    tl0xc = np.real(sltc)
    su0xc = np.real(suyc)
    sl0xc = np.real(slyc)

    fig, ax = plt.subplots()
    h_list = []

    if np.max(suera) > np.max(slera):
        line1, = ax.plot(tuera, suera, color = 'red', label = 'with e-e interaction')
    else:
        line1, = ax.plot(tlera, slera, color = 'red', label = 'with e-e interaction')
    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')
    h_list.append(line1)
    h_list.append(line13)
    if np.max(su0xc) > np.max(sl0xc):
        line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'exact solution')
    else:
        line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'exact solution')
    h_list.append(line9)

    ax.legend(handles=h_list)
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

    fig, ax = plt.subplots()

    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

    # real triggered envelope
    fig, ax = plt.subplots()


    h_list = []

    line1, = ax.plot(tuerr, suerr, color = 'red', label = 'with e-e interaction')
    line2, = ax.plot(tlerr, slerr, color = 'red', label = 'with e-e interaction')
    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')

    h_list.append(line1)
    h_list.append(line13)

    if np.max(su0xc) > np.max(sl0xc):
        line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'exact solution')
    else:
        line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'exact solution')
    h_list.append(line9)

    ax.legend(handles=h_list)
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

def test_match_old(dk, R2, B2, mu):
    # set parameters
    #dk = 0.5
    #R2 = 1.0
    R  = R2 * R_max
    #B2 = 0.5
    B  = B2 * B_max
    #mu = 1.0e-7


    yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=k_calc_0, M_calc_f=M_calc_0)
    #jj = lambda x: jac(x, mu, dk, B, R, k_calc_f=k_calc_0, M_calc_f=M_calc_0, dk_calc_f=dk_calc_0, dM_calc_f=dM_calc_0)

    sol = optimize.root(yy, [-1e12, 1e12], method='hybr')
    xs = sol.x
    print("Root of linear solution:", xs, yy(xs))

    yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R)
    #jj = lambda x: jac(x, mu, dk, B, R)

    sol = optimize.root(yy, xs/10, method='broyden2', tol=1e-20)
    dphi0c = sol.x
    print(dphi0c)
    print("Root of nonlinear solution:", yy(dphi0c))

    deriv_psi = dphi0c[0] + 1j * dphi0c[1]
    print("Root of nonlinear solution:", deriv_psi)

    l_calc = loop(R2, B2, dk, mu)

    deriv_psi_o = xs[0]+1j*xs[1]
    print("Root of linear solution:", deriv_psi_o)

    print("")

    print("aj, bj", l_calc.aj, l_calc.bj)
    l_calc.setDeriv(deriv_psi_o)
    print("aj, bj", l_calc.aj, l_calc.bj)

    l_calc.setDeriv(deriv_psi)

    print("Exact solution endpoint:", l_calc.psij0(2*pi*R2 * R_max))
    print("New solution endpoint:", l_calc.psij_pred(2*pi*R2 * R_max), np.abs(l_calc.psij_pred(2*pi*R2 * R_max)))

    print("aj, bj:", l_calc.aj, l_calc.bj)
    print("aj0, bj0:", l_calc.aj0, l_calc.bj0)

    aabs = np.abs(l_calc.aj)**2
    babs = np.abs(l_calc.bj)**2

    kn = pi / l_calc.T_fast_mod
    mn = 2* pi /l_calc.T_slow_mod
    ko = l_calc.k
    mo = l_calc.M

    cur_new = (aabs + babs)*(mn-mo)/ko + kn/ko * (aabs - babs)
    cur_old = np.abs(l_calc.aj0)**2 - np.abs(l_calc.bj0)**2

    print("")
    print("New and old current estimates:", cur_new, cur_old)


    l_calc.solve_ivp(n = 1000, solve=1)
    solu = l_calc.solu

    T_arr = l_calc.find_period_shift_exact() # fast, slow, pos, neg
    xT = l_calc.find_real_env_start()
    trim = 16
    n = 1000

    sutc = l_calc.find_t_points(n, l_calc.lngt, l_calc.stu_ex, T_arr[0])
    sltc = l_calc.find_t_points(n, l_calc.lngt, l_calc.stl_ex, T_arr[0])
    sutcm = l_calc.find_t_points(n, l_calc.lngt, xT, T_arr[2])
    sltcm = l_calc.find_t_points(n, l_calc.lngt, xT, T_arr[3])
    suyc = l_calc.psij0(sutc)
    slyc = l_calc.psij0(sltc)
    suycm = l_calc.psij0(sutcm)
    slycm = l_calc.psij0(sltcm)
    M_pred = 2*pi / l_calc.T_slow_mod

    A_pred = find_amp(solu)
    sol_t = np.real(solu['t']) 
    sol_y = np.real(solu['y'][0])
    ind_st = find_root_start(solu)
    t0_pred = sol_t[ind_st]
    t_pred = sol_t
    y_pred = A_pred * np.sin(M_pred * (t_pred-t0_pred))

    #position arrays
    #estimated, triggered buy abs, triggered by real (both) -- c, a, r
    tuerr = np.real(solu['t_events'][1][0::trim])
    tlerr = np.real(solu['t_events'][1][1::trim])
    tuera = np.real(solu['t_events'][0][0::trim])
    tlera = np.real(solu['t_events'][0][1::trim])
            
    #value arrays
    suerr = np.real(solu['y_events'][1][0::trim, 0])
    slerr = np.real(solu['y_events'][1][1::trim, 0])
    suera = np.real(solu['y_events'][0][0::trim, 0])
    slera = np.real(solu['y_events'][0][1::trim, 0])

    tu0xc = np.real(sutc)
    tl0xc = np.real(sltc)
    su0xc = np.real(suyc)
    sl0xc = np.real(slyc)

    fig, ax = plt.subplots()
    h_list = []

    if np.max(suera) > np.max(slera):
        line1, = ax.plot(tuera, suera, color = 'red', label = 'with e-e interaction')
    else:
        line1, = ax.plot(tlera, slera, color = 'red', label = 'with e-e interaction')
    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')
    h_list.append(line1)
    h_list.append(line13)
    if np.max(su0xc) > np.max(sl0xc):
        line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'exact solution')
    else:
        line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'exact solution')
    h_list.append(line9)

    ax.legend(handles=h_list)
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

    fig, ax = plt.subplots()

    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

    # real triggered envelope
    fig, ax = plt.subplots()


    h_list = []

    line1, = ax.plot(tuerr, suerr, color = 'red', label = 'with e-e interaction')
    line2, = ax.plot(tlerr, slerr, color = 'red', label = 'with e-e interaction')
    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')

    h_list.append(line1)
    h_list.append(line13)

    if np.max(su0xc) > np.max(sl0xc):
        line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'exact solution')
    else:
        line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'exact solution')
    h_list.append(line9)

    ax.legend(handles=h_list)
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

