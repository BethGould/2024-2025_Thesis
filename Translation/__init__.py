# __init__.py

# Автор: Элизабет Гоулд
# Дата последнего изменения: 19.03.2026

'''
Создает библиотеку. Все функции, константы и классы добавляются дважды. 
Сначала они импортируются, а затем назначаются.

Вспомогательные функции и функции, реализованные не полностью, здесь не 
отображаются, поскольку они не предназначены для использования из библиотеки.
Если они необходимы, они должны вызываться с помощью
eelib.filename.function_name вместо eelib.function_name. 
Формат eelib.filename.function_name всегда работает.
'''

from eelib.consts import h_pl, hbar, m_e, e_e, c_l, pi, rtol, atol, MU_min, MU_max, DK_min, DK_max
from eelib.consts import kFAu, R_max, R_min, B_min, B_max, I_e, phi0inv, pppterm
from eelib.deriv_functions import psi_deriv, psi_deriv_old, psi_deriv_0, psi_deriv_full
from eelib.events import event1, event2, deriv_amp, deriv_phase, deriv_real

from eelib.fitted_functions import sin_fit
from eelib.fitted_functions import fit_sin

from eelib.k_M_models_ivp import pred_fast_t, pred_fast_k, pred_slow_k, pred_slow_k_v3, pred_fast_k_true, pred_slow_k_v2
from eelib.bvp_rootfinder_functions import function_wrapper_bvp, k_calc_0, M_calc_0
from eelib.bvp_test_script import test_match

from eelib.loop import loop
from eelib.deriv_grid import deriv_grid
from eelib.grid_fast_osc import grid_fast_osc
from eelib.grid_slow_osc import grid_slow_osc
from eelib.grid_bvp import grid_BVP

from eelib.table_scripts import clean_table, find_ind_var, clean_table_MC

__all__ = ['h_pl', 'hbar', 'm_e', 'e_e', 'c_l', 'pi', 'kFAu', 'R_max', 'R_min', 'B_min', 'B_max', 'I_e', 
           'phi0inv', 'pppterm', 'rtol', 'atol', 'MU_min', 'MU_max', 'DK_min', 'DK_max', 
           'psi_deriv', 'psi_deriv_old', 'psi_deriv_0', 'psi_deriv_full', 
           'event1', 'event2', 'deriv_amp', 'deriv_phase', 'deriv_real', 
           'sin_fit', 'fit_sin', "clean_table", 'find_ind_var', "clean_table_MC",
           'pred_fast_t', 'pred_fast_k', 'pred_slow_k', 'pred_slow_k_v3', 'pred_fast_k_true',
           "pred_slow_k_v2", "function_wrapper_bvp", "test_match", "k_calc_0", "M_calc_0",
           'loop', 'deriv_grid', 'grid_fast_osc', 'grid_slow_osc', 'grid_BVP']