# bvp_test_script.py

# Author: Elizabeth Gould
# Date Last Edit: 09.03.2026

# test_match(dk, R2, B2, mu)
#    -- dk, R2, B2, and mu are our loop parameters, as we pass to create a loop.
#    -- R2 is our R parameter and B2 is our B parameter.
#    -- Note that R2 and B2 here are percents, as is effectively dk, like those used to create a loop object.

# This is my full script for creating graphs of the solution to the BVP. The shown graphs will be for
# the real part of the wave function, where it follows maxima of the absolute value of the fast oscillations.
# Therefore, the plotted function will be for the slow oscillation graph. There are two possible graphical 
# solutions to the linear case and four for the nonlinear case. For both cases, 
#  - psi_slow(0) = psi_slow(2*pi*R) or
#  - psi_slow(0) = - psi_slow(2*pi*R).
# The second case occurs if the matching of the fast oscillation switches sign, since this matching only 
# requires equal amplitude. 
# For the linear case, the derivatives of the absolute value at the beginning and end must have opposite signs.
# For the nonlinear case, the derivative may also match for the final solution, since dk is not selected to 
# prevent this case, as it doesn't create as severe an ambiguity in the matching solution. 

# Outputs will be displayed on the screen.

import numpy as np
from scipy import optimize
from eelib.consts import pi, B_max, R_max #, kFAu, phi0inv
from eelib.k_M_models_ivp import pred_fast_k, pred_slow_k_v3, pred_fast_k_true
from eelib.fitted_functions import fit_sin
from eelib.bvp_rootfinder_functions import find_root_both
from eelib.loop import loop
import matplotlib.pyplot as plt
#from eelib.bvp_rootfinder_functions import function_wrapper_bvp, k_calc_0, M_calc_0

# This is a script to match the nonlinear solution at the boundary using the model for k and M
# and output results on the screen. 
# The root of our matching equation found is effectively random (chaotic, not random). There are many. 
# The program starts looking for roots around the linear solution, but depending on the algorithm, the solution may 
# not be the closest one.
def test_match(dk, R2, B2, mu):
    R  = R2 * R_max
    B  = B2 * B_max

    # Find the roots
    deriv_psi_o, xs_sol, deriv_psi, dpsi_sol = find_root_both(dk, R, B, mu, model_k=pred_fast_k, ratio = 10, check_solution = True)
    print("Root of linear solution:", deriv_psi_o) 
    print("Value of matching function at this root (should be 0):", xs_sol)
    print("Root of nonlinear solution:", deriv_psi)
    print("Value of matching function at this root (should be 0):", dpsi_sol)

    # Our outputs separate the real and imaginary parts of our psi_0 derivative 
    # as if they were separate variables. We need to reunite them as a single complex number.
    #deriv_psi = dpsi0[0] + 1j * dpsi0[1]
    #deriv_psi_o = xs[0]+1j*xs[1]

    # Define a loop object for use with this root
    l_calc = loop(R2, B2, dk, mu)

    print("") # newline

    # a, b for calculated and fit linear solutions
    # They are typically very close.
    print("aj, bj for the calculated linear solution: ", l_calc.aj, l_calc.bj)
    l_calc.setDeriv(deriv_psi_o)
    print("aj, bj for the fit linear solution: ", l_calc.aj, l_calc.bj)

    # Now for the nonlinear bvp solution.
    l_calc.setDeriv(deriv_psi)

    print("") # newline

    # Check to see how good the match is, here using the model for k and M, not the ode solver.
    print("Exact solution endpoint:", l_calc.psij0(2*pi*R2 * R_max))
    print("New solution endpoint:", l_calc.psij_pred(2*pi*R2 * R_max), "Absolute value =", np.abs(l_calc.psij_pred(2*pi*R2 * R_max)))

    # linear and nonlinear coefficients 
    print("aj, bj:", l_calc.aj, l_calc.bj)
    print("aj0, bj0 (linear solution):", l_calc.aj0, l_calc.bj0)

    print("") # newline

    # current calculations
    cur_new = l_calc.current_alt()
    cur_old = l_calc.current_old()

    print("Nonlinear current estimate:", cur_new)
    print("Linear current estimate:", cur_old)


    # Now all outputs are graphical rather than text based.

    # ODE solver with our solutions, for visual analysis of our matching
    l_calc.solve_ivp(n = 1000, solve=1)
    solu = l_calc.solu

    # All this code below is a copy of that from our derivative grid plotting script.
    # We are retrieving our x and psi data from our loop for everything we wish to plot.

    # Initial parameters
    T_arr = l_calc.find_period_shift_exact() # fast, slow, pos, neg
    trim = 16
    n = 1000

    # This is our exact solution, following the fast oscillations.
    sutc = l_calc.find_t_points(n, l_calc.lngt, l_calc.stu_ex, T_arr[0])
    sltc = l_calc.find_t_points(n, l_calc.lngt, l_calc.stl_ex, T_arr[0])
    suyc = l_calc.psij0(sutc)
    slyc = l_calc.psij0(sltc)

    # Find our modelled wave number.
    M_pred = 2*pi / l_calc.T_slow_mod

    # Fit all parameters except the wave number to a sine.
    sol_t = np.real(solu['t']) 
    t_pred = sol_t
    fit_func = fit_sin(solu)
    A_pred = fit_func[0]
    theta = fit_func[2]
    y_pred = A_pred * np.sin(M_pred * (sol_t) + theta)

    # position arrays
    # estimated, triggered buy abs, triggered by real (both) -- c, a, r
    tuerr = np.real(solu['t_events'][1][0::trim])
    tlerr = np.real(solu['t_events'][1][1::trim])
    tuera = np.real(solu['t_events'][0][0::trim])
    tlera = np.real(solu['t_events'][0][1::trim])
            
    # value arrays
    suerr = np.real(solu['y_events'][1][0::trim, 0])
    slerr = np.real(solu['y_events'][1][1::trim, 0])
    suera = np.real(solu['y_events'][0][0::trim, 0])
    slera = np.real(solu['y_events'][0][1::trim, 0])

    # Clear imaginary part
    tu0xc = np.real(sutc)
    tl0xc = np.real(sltc)
    su0xc = np.real(suyc)
    sl0xc = np.real(slyc)

    # First plot
    fig, ax = plt.subplots()
    plt.title(f"Re(ψ(x)), for maxima of |ψ(x)|, dk={dk}, R={R2}, B={B2}, \u03BC={mu}")
    ax.set_ylabel('Re(ψ) (ratio of ψ(0))')
    ax.set_xlabel('x (m)')


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

    # Second plot of only the predicted value.
    fig, ax = plt.subplots()
    plt.title(f"Plot of modelled slow oscillations of Re(ψ(x)), dk={dk}, R={R2}, B={B2}, \u03BC={mu}")
    ax.set_ylabel('Re(ψ) (ratio of ψ(0))')
    ax.set_xlabel('x (m)')


    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'predicted e-e interaction')
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

    # Third plot
    # real triggered envelope
    fig, ax = plt.subplots()
    plt.title(f"Re(ψ(x)) envelope, dk={dk}, R={R2}, B={B2}, \u03BC={mu}")
    ax.set_ylabel('Re(ψ) (ratio of \u03A8(0))')
    ax.set_xlabel('x (m)')


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


# ---------------------------------------------------------------

# This is an old, unused version of the above script.
'''
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

'''