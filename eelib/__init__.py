# Builds the library. Note that all functions, constants, and classes are added twice.
# First they are imported, then assigned.

# Currently used library files (6) -- consts, deriv_functions, events, loop, deriv_grid, grid_fast_osc

from eelib.consts import h_pl, hbar, m_e, e_e, c_l, pi, rtol, atol, MU_min, MU_max, DK_min, DK_max
from eelib.consts import kFAu, R_max, R_min, B_min, B_max, I_e, phi0inv, pppterm
from eelib.deriv_functions import psi_deriv, psi_deriv_old, psi_deriv_0, psi_deriv_full, sin_fit
from eelib.events import event1, event2, deriv_amp, deriv_phase, deriv_real
from eelib.loop import loop
from eelib.deriv_grid import deriv_grid
from eelib.grid_fast_osc import grid_fast_osc
from eelib.grid_slow_osc import grid_slow_osc
from eelib.k_M_models import pred_fast_t, pred_slow_t, pred_slow_k_v2, pred_slow_k_v3, pred_fast_k_true
from eelib.grid_bvp import grid_BVP
#from eelib.loop_system import 
#from eelib.param_grid import
#from eelib.plotting_scripts import

__all__ = ['h_pl', 'hbar', 'm_e', 'e_e', 'c_l', 'pi', 'kFAu', 'R_max', 'R_min', 
           'B_min', 'B_max', 'I_e', 'phi0inv', 'pppterm', 'rtol', 'atol', 'MU_min', 
           'MU_max', 'DK_min', 'DK_max', 'psi_deriv', 'psi_deriv_old', 
           'psi_deriv_0', 'psi_deriv_full', 'event1', 'event2', 'deriv_amp', 
           'deriv_phase', 'deriv_real', 'loop', 'deriv_grid', 'grid_fast_osc', 
           'pred_fast_t', 'grid_slow_osc', 'pred_slow_t',  'sin_fit', 'pred_slow_k_v2', 
           'pred_slow_k_v3', 'pred_fast_k_true', 'grid_BVP']