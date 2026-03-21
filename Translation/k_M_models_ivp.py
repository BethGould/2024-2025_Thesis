# k_M_models_ivp.py
#
# Автор: Элизабет Гоулд
# Дата последнего изменения: 19.03.2026

# Здесь приведены наши модели для изменения частот обоих типов колебаний. Они
# рассчитаны как функции mu, dk, B, R и dpsi0. Зависимость от dpsi0 означает, 
# что они решают начальную задачу, а не краевую задачу.

# Наиболее важными функциями, содержащимися здесь, являются:
# pred_fast_k(dpsi0, mu, dk, B, R, A = 1., k0=kFAu) 
#   - Модель для k для e-e взаимодействия, высчитанная с помощью кода для
#     решения ОДУ.
# pred_fast_k_true(dpsi0, mu, dk, B, R, A = 1., k0=kFAu)
#   - Модель для k для e-e взаимодействия, после вычитания ошибки.
# pred_slow_k(dpsi0, mu, dk, B, R, A = 1., k0=kFAu)
#   - Модель M для e-e взаимодействия.

# Также не скрыто от компилятора: 
# pred_fast_t(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001) 
#   Это наша исходная модель для быстрых колебаний. Она такая же, как и 
#   pred_fast_k, но возвращает значение t/2 вместо k. Я предпочитаю она
#   вместо pred_fast_k, поскольку она была первым.
# pred_slow_k_v3(dpsi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_004)
#   Это то же самое, что и pred_slow_k. Наша текущая модель была третьей по
#   счету в этом документе, отсюда и название. Остальные были удалены, но я
#   по-прежнему использую это название по этой причине.
# pred_slow_k_v2(dpsi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_004)
#   Это менее точная модель для медленных колебаний, которая остается для
#   демонстрации нашего кода извлечения модели.

# ----- БИБЛИОТЕКИ --------

import numpy as np
from eelib import pi, kFAu, phi0inv, pppterm, B_max, R_max

# ------ ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ --------
# Эти словари являются жестко запрограммированными параметрами, которые мы уже
# нашли в числовом выражении. Они не используются за пределами приведенных
# здесь функций.

# Первая модель, период быстрых колебаний
const_dic_001 ={"mA2": 3.675494115574291e-08, 
                "mDIM": -6.179621225636901e-11, 
                "mDI2": 3.674200444430406e-08, 
                "mDR2": 3.674365054354659e-08, 
                "m2": 8.522223583430349e-06, 
                "m2DI2": 2.0556114758548036e-05, 
                "m2DR2": 2.9358597545134476e-05, 
                "m2DR4": 1.598207117609667e-05, 
                "m2DI4": 1.5910577914323505e-05, 
                "m2DI2R2": 3.182358695140475e-05, 
                "intercept": -5.697364648383741e-16, 
                "DI2": -2.7618261397447087e-16, 
                "DR2": 8.0960114314636e-17, 
                "DI4": 9.19298181818588e-17}

# Вторая модель, период медленных колебаний
const_dic_002 = {"mdI": 104537.83229936952
                 }

# Третья модель, вольное число медленных колебаний (v2)
const_dic_003 ={"mdI": 1116125114862.058
                 }

# Четвёртая модель, вольное число медленных колебаний (v3)
const_dic_004 = {"mdI": 1114503328463.9722,
                  "mdR2I": 1487350163.9047058}

# ----- DELTA k МОДЕЛИ -----

# Первая модель
# Вычисляет t/2 для быстрых колебаний. t/2 - это то, что я получаю в результате
# усреднения расстояния между максимумами функции |psi|**2.
def pred_fast_t(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001):
    t_pred = 0

    #k_full = k0+dk/R/1e-6/2.0
    k_full = k0+dk/R_max/2.0
    t0 = pi / k_full
    dphik = dphi0/k_full

    # постоянные члены - t_0 и постоянная ошибка
    t_pred = t0 + consts['intercept']

    # члены не зависимые от mu (ошибка)                                
    t_pred += (np.real(dphik)**2 * consts['DR2'] 
                + np.imag(dphik)**2 * consts['DI2'] 
                + np.imag(dphik)**4 * consts['DI4'])

    # члены, линейные в mu
    t_pred += mu*(consts['mDR2'] * np.real(dphik)**2
                + consts['mDI2'] * np.imag(dphik)**2
                + consts['mA2']
                + consts['mDIM'] * np.imag(dphik) * B * R)

    # члены, квадратичные по mu
    t_pred += mu**2*(consts['m2']
                    + consts['m2DI2'] * np.imag(dphik)**2
                    + consts['m2DR2'] * np.real(dphik)**2 
                    + consts['m2DR4'] * np.real(dphik)**4 
                    + consts['m2DI4'] * np.imag(dphik)**4
                    + consts['m2DI2R2'] * np.real(dphik)**2 * np.imag(dphik)**2)
    
    return t_pred
    
# Здесь функция возвращает k вместо t / 2, что, как правило, более полезно. 
# Она изменяет масштаб B и R обратно на процентный, поэтому нужно передать R и B
# с единицами а не в процентах от максимума.
def pred_fast_k(dpsi0, mu, dk, B, R, A = 1., k0=kFAu):
    B = B / B_max
    R = R / R_max
    return pi / pred_fast_t(dpsi0, mu, dk, B, R, A, k0)

# Изменение масштаба A эквивалентно изменению масштаба mu и dphi0.
# Однако изменение масштаба dphi0 не является простым без знания psi(x).

# Окончательное решение чувствительно к R, но не к A, в то время как модель для
# k чувствительна к A, но не к R. Это связано с различием между начальными и
# краевыми задачами. Это также означает, что можно исключить отклонения A и R.

# Модель содержит:
# 0-й порядок:
# - Ожидаемое значение k без e-e взаимодействия.
# 1-й порядок: 
# - Основной член (-ы) дельта-k для нелинейного e-e взаимодействия.
#   По сути, это один член.
# 2-й порядок: 
# - Поправки второго порядка для модификации нелинейного e-e взаимодействия.
# - Поправки первого порядка для численных решений без e-e взаимодействия.
#   Насколько я могу судить, ошибка идентична случаю без e-e взаимодействия.
#   Скорее всего, ошибка от исправлений будет видна только в членах четвертого
#   порядка, хотя необходимы не более чем члены третьего порядка. Это гипотеза,
#   и она не доказана.


# Это производная от k, разделяющая действительную и мнимую производные psi
# как отдельные переменные. Она была удалена, так как якобиан не используется.
'''
def deriv_fast_k(x, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001):
    t_pred_r = 0
    t_pred_i = 0

    dphi0 = x[0] + 1j * x[1]

    #k_full = k0+dk/R/1e-6/2.0
    k_full = k0+dk/R_max/2.0
    dphik = dphi0/k_full

    t_new = pred_fast_t(dphi0, mu, dk, B, R, A , k0, consts)

    # постоянные члены имеют нуль производную

    # члены не зависимые от mu (ошибка)                                
    t_pred_r += (2 * np.real(dphik) * consts['DR2'] / k_full)
    t_pred_i += (2 * np.imag(dphik) * consts['DI2'] / k_full
                + 4 * np.imag(dphik)**3 * consts['DI4'] / k_full)

    # члены, линейные в mu
    t_pred_r += mu*(consts['mDR2'] * np.real(dphik)* 2 / k_full)
    t_pred_i += mu*(consts['mDI2'] * np.imag(dphik)* 2 / k_full
                + consts['mDIM'] * B * R / k_full)

    # члены, квадратичные по mu
    t_pred_r += mu**2*(consts['m2DR2'] * np.real(dphik)*2/k_full 
                    + consts['m2DR4'] * np.real(dphik)**3 * 4 / k_full
                    + consts['m2DI2R2'] * np.real(dphik)*2 * np.imag(dphik)**2/k_full)
    t_pred_i += mu**2*(consts['m2DI2'] * np.imag(dphik)*2/k_full
                    + consts['m2DI4'] * np.imag(dphik)**3*4/k_full
                    + consts['m2DI2R2'] * np.real(dphik)**2 * np.imag(dphik)*2/k_full)
    
    dk_pred_r = - pi * t_pred_r / (t_new**2)
    dk_pred_i = -pi * t_pred_i / (t_new**2)
    
    return [dk_pred_r, dk_pred_i]
'''

# Это наши старые, неудачные модели M, и поэтому они были удалены.
'''
def pred_slow_t(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_002):
    M = B*R*phi0inv
    #k_full = k0+dk/R/R_max/2.0
    k_full = k0+dk/R_max/2.0
    BB = B * R / R_max

    #print(M)
    #print(B, R, M, BB)

    # постоянные члены
    #M_pred = M 

    # члены, линейные в mu
    #M_pred += mu * consts['mdI'] * np.imag(dphi0) * R * phi0inv / k_full / 4

    #print(mu * consts['mdI'] * np.imag(dphi0) / k_full)

    # Обратите внимание, что масштабирование может быть неправильным из-за 
    # увеличения в 4 раза
    M_pred = 2 * (pi / (M)) / (mu/ BB * B_max * np.imag(dphi0) * consts['mdI'] / k_full + 1)
    
    return M_pred


# Якобиан; Обратите внимание, что Re(psi') = x_0 и Im(psi') = x_1.
def deriv_slow_k(x, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_002):
    M = B*R*phi0inv
    #k_full = k0+dk/R/2.0
    k_full = k0+dk/R_max/2.0
    BB = B * R / R_max

    # Обратите внимание, что масштабирование может быть неправильным из-за 
    # увеличения в 4 раза
    M_pred_i = - 2 * (pi / (2*M)) / (mu/ BB * B_max * x[1] * consts['mdI'] / k_full + 1)**2 * mu/ BB * B_max * consts['mdI'] / k_full
    
    return [0., M_pred_i]

'''
# Эта модель остается здесь для демонстрации проверок моделей.
def pred_slow_k_v2(dpsi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_003):
    M = B*R*phi0inv
    k_full = k0+dk/R_max/2.0
    dpsi0_s = dpsi0 / k_full

    #M_pred = (B*R*R_max*phi0inv*B_max) + consts['mdI'] * mu * np.real(dphi0) / k_full
    M_pred = M + consts['mdI'] * mu * np.imag(dpsi0_s)
    #   2 * (pi / (M)) / (mu/ BB * B_max * np.imag(dphi0) * consts['mdI'] / k_full + 1)
    
    return M_pred

# Представленная здесь модель для M - это просто взывание нашей самой успешной 
# модели, представленной ниже.
def pred_slow_k(dpsi0, mu, dk, B, R, A = 1., k0=kFAu):
    return pred_slow_k_v3(dpsi0, mu, dk, B, R, A, k0)

# Наша успешная модель для M
def pred_slow_k_v3(dpsi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_004):
    M = B*R*phi0inv
    k_full = k0+dk/R_max/2.0
    dpsi0_s = dpsi0 / k_full
    
    M_pred = M + consts['mdI'] * mu * np.imag(dpsi0_s) + consts['mdR2I'] * mu * np.imag(dpsi0_s) * np.real(dpsi0_s)**2
    
    return M_pred

# Якобиан не используется, поэтому он был удален.
'''
def deriv_slow_k_v3(x, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_004):
    #M = B*R*phi0inv
    k_full = k0+dk/R_max/2.0
    dpsi0_sr = x[0] / k_full
    dpsi0_si = x[1] / k_full
    
    #M_pred = M + consts['mdI'] * mu * np.imag(dpsi0_s) + consts['mdR2I'] * mu * np.imag(dpsi0_s) * np.real(dpsi0_s)**2
    M_pred_r = 2 * consts['mdR2I'] * mu * dpsi0_si * dpsi0_sr / k_full
    M_pred_i = consts['mdI'] * mu / k_full + consts['mdR2I'] * mu * dpsi0_sr**2 / k_full
    
    return [M_pred_r, M_pred_i]
'''

# Модель для k (нашего волнового числа быстрых колебаний) с удаленным ошибки. 
def pred_fast_k_true(dphi0, mu, dk, B, R, A = 1., k0=kFAu, consts=const_dic_001):
    t_pred = 0

    if A != 1.:
        mu = mu * np.abs(A)**2

    #k_full = k0+dk/R/1e-6/2.0
    k_full = k0+dk/R_max/2.0
    t0 = pi / k_full
    dphik = dphi0/k_full

    B = B / B_max
    R = R / R_max

    # постоянные члены - t_0 и постоянная ошибка
    t_pred = t0 #+ consts['intercept']

    # члены не зависимые от mu (ошибка)                                
    #t_pred += (np.real(dphik)**2 * consts['DR2'] 
    #            + np.imag(dphik)**2 * consts['DI2'] 
    #            + np.imag(dphik)**4 * consts['DI4'])

    # члены, линейные в mu
    t_pred += mu*(consts['mDR2'] * np.real(dphik)**2
                + consts['mDI2'] * np.imag(dphik)**2
                + consts['mA2']
                + consts['mDIM'] * np.imag(dphik) * B * R)

    # члены, квадратичные по mu
    t_pred += mu**2*(consts['m2']
                    + consts['m2DI2'] * np.imag(dphik)**2
                    + consts['m2DR2'] * np.real(dphik)**2 
                    + consts['m2DR4'] * np.real(dphik)**4 
                    + consts['m2DI4'] * np.imag(dphik)**4
                    + consts['m2DI2R2'] * np.real(dphik)**2 * np.imag(dphik)**2)
    
    return pi / t_pred
    
