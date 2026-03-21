# deriv_functions.py
#
# Автор: Элизабет Гоулд
# Дата последнего изменения: 18.03.2026
#
# Здесь содержатся функции, используемые функциями для решения ОДУ по ссылке.
# Они не включены в структуру класса и существуют в унифицированном виде, так
# что их можно легко заменить.

#--БИБЛИОТЕКИ--------
import numpy as np
from scipy.special import ellipj

from eelib.consts import pppterm, phi0inv

#--ПРОИЗВОДНЫЕ ФУНКЦИИ ДЛЯ ФУНКЦИЙ РЕШЕНИЯ ОДУ--
# x
# y  = [psi, dpsi/dx]
# k  -- волновое число электрона
# B  -- напряженность магнитного поля
# R  -- радиус -- обратите внимание, что здесь имеет значение только B * R, 
#       а не B и R отдельно
# mu -- сила e-e связи
# mA -- максимальная амплитуда - используется событиями, а не функциями для 
#       решения производных, но должна быть включена

# В кольце с e-e связи
def psi_deriv(x, y, k, B, R, mu, mA):
    knl = mu * pppterm
    k0  = complex(np.square(k))
    k1  = 2j * complex(B * R * phi0inv)
    k2  = complex(np.square(B * R * phi0inv))
    abssqpsi = (np.square(np.real(y[0]))+np.square(np.imag(y[0])))

    dpdxx = complex(knl * abssqpsi) * y[0] - k0 * y[0] + k1 * y[1] + k2 * y[0]
    dpdx  = y[1]
    return [dpdx, dpdxx]

# Внутри кольца, без e-e связи
def psi_deriv_old(x, y, k, B, R, mu, mA):
    k0 = complex(np.square(k))
    k1 = 2j * complex(B * R * phi0inv)
    k2 = complex(np.square(B * R * phi0inv))
    
    dpdxx = k1 * y[1] + k2 * y[0] - k0 * y[0]
    dpdx = y[1]
    return [dpdx, dpdxx]

# Для ввода в кольца, без магнитного поля или связи
def psi_deriv_0(x, y, k, B, R, mu, mA):
    k0 = complex(np.square(k))
    
    dpdxx = - k0 * y[0]
    dpdx = y[1]
    return [dpdx, dpdxx]

# Оба варианта вместе содержатся для компактности.
def psi_deriv_full(x, y, k, B, R, mu, mA):
    knl = mu * pppterm
    k0  = complex(np.square(k))
    k1  = 2j * complex(B * R * phi0inv)
    k2  = complex(np.square(B * R * phi0inv))
    abssqpsi = (np.square(np.real(y[2]))+np.square(np.imag(y[2])))

    dpdxx2 = complex(knl * abssqpsi) * y[2] - k0 * y[2] + k1 * y[3] + k2 * y[2]
    dpdx2  = y[3]
    dpdxx = k1 * y[1] + k2 * y[0] - k0 * y[0]
    dpdx = y[1]
    return [dpdx, dpdxx, dpdx2, dpdxx2]  
