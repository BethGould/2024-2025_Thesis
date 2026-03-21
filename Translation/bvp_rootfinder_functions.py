# bvp_rootfinder_functions.py
#
# Автор: Элизабет Гоулд
# Дата последнего изменения: 18.03.2026

# Здесь содержится моя дескриптор функции для решения краевой задачи с помощью поиска нулей функции, 
# линейные модели, которые будут использоваться с этим дескриптором функции, 
# и пример кода для выполнения поиска нулей функции.

# k_calc_0(dpsi0, mu, dk, B, R, A = 1., k0=kFAu) -- возвращает k = k0+dk/R_max/2.0, (линейную модель)
# M_calc_0(dpsi0, mu, dk, B, R, A = 1., k0=kFAu) -- возвращает M = B*R*phi0inv, (линейную модель)

# function_wrapper_bvp(x, mu, dk, B, R, A = 1., k0=kFAu, k_calc_f = pred_fast_k, M_calc_f = pred_slow_k_v3)
#   -- Принимает параметры кольца mu, dk, B, R, A и k0 и вычисляет разницу между заданной начальной 
#      производной psi dpsi0 = x[0] + 1.j*x[1] и требуемой производной для соединения краевой задачи. 
#      Поиск нулей этой функции решает краевую задачу.
#
#   -- Чтобы передать эту функцию в программу поиска нулей, используйте код:
#      >  yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f, M_calc_f)
#      и передайте yy в качестве дескриптора функции, для которой нужно найти нулей.
#
#   -- Параметры кольца B и R указаны в реальных единицах, а не в процентах от их максимального значения. 
#
#   -- Для этой функции требуется модель для k и M в зависимости от наших параметров.
#   -- k_calc_f(dpsi0, mu, dk, B, R, A, k0) and M_calc_f(dpsi0, mu, dk, B, R, A, k0)
#   -- Здесь значение dpsi0 представлено в виде единственного комплексного числа, а значения B и R 
#      указаны в реальных единицах измерения, как они хранятся в классе loop, а не в процентах от их 
#      максимального значения, как они передаются в класс loop.


# find_root_both(dk, R, B, mu, method = 'broyden2', tol = 1e-20, ratio = 1.0,
#                  check_solution = False, model_k=pred_fast_k_true, model_M = pred_slow_k_v3)
#   -- Это функция, которая найдет одно из решений нелинейной краевой задачи, а также решение линейной краевой 
#      задачи. Если значение check_solution равно false, то оно вернет только решение нелинейной краевой задачи 
#      в виде массива из двух действительных чисел с действительной и мнимой составляющими начальной производной 
#      от psi.  Если значение check_solution равно true, то оно вернет 4 значения:
#          - Комплексное решение для начальной производной от psi для линейного случая.
#          - Значение функции, для которого нужно выяснить нуль, для этой (линейной) производной.
#          - Комплексное решение для начальной производной от psi для нелинейного случая.
#          - Значение функции, для которого нужно выяснить нуль, для этой (нелинейной) производной.
#   -- Нашими входными данными являются:
#       - dk, R, B, mu, в реальных единицах, как получено из нашего цикла.
#       - method - Алгоритм поиска нуля из списка доступных алгоритмов SciPy.
#       - tol - Допуск для этого алгоритма.
#       - ratio - Первоначальное предположение соответствует линейному решению в виде dphi0 / ratio.
#       - model_k and model_M - это модели k and M для функции, для которой нужно найти нули.

# У меня есть код для якобиана, который сейчас не используется, так как он не был проверен и отлажен. 
# Раньше у меня были проблемы с соединение, поэтому мне нужно было переключить модель, и я не обновлял 
# свой якобиан. Весь код для якобиана был скрыт от компьютера в виде комментариев.


import numpy as np
from scipy import optimize
from eelib.consts import pi, kFAu, phi0inv, R_max
from eelib.k_M_models_ivp import pred_fast_k, pred_slow_k_v3, pred_fast_k_true
#from eelib.k_M_models_ivp import deriv_fast_k, deriv_slow_k_v3

# Первые функции предоставляют модели для k и M для линейного случая.
# Они используют неиспользуемые параметры, поскольку их параметры должны соответствовать параметрам других моделей.
def k_calc_0(dpsi0, mu, dk, B, R, A = 1., k0=kFAu):
    return k0+dk/R_max/2.0

def M_calc_0(dpsi0, mu, dk, B, R, A = 1., k0=kFAu):
    return B*R*phi0inv

#def dk_calc_0(x, mu, dk, B, R, A = 1., k0=kFAu):
#    return [0.0, 0.0]

#def dM_calc_0(x, mu, dk, B, R, A = 1., k0=kFAu):
#    return [0.0, 0.0]

# Вот функция, для которой мы находим нули.
# Обратите внимание, что это то же самое, что и в линейном случае, но со смещенными волновыми числами.
# Это приближение решения, а не фактическая ожидаемая форма, для которой быстрые колебания не являются 
# полностью синусоидальными. Однако графики показывают, что это приемлемое приближение.
def function_wrapper_bvp(x, mu, dk, B, R, A = 1., k0=kFAu, k_calc_f = pred_fast_k, M_calc_f = pred_slow_k_v3):
    # Расшифровать переменную x.
    dpsi = x[0] + 1j * x[1]

    k_calc = k_calc_f(dpsi, mu, dk, B, R, A, k0)  #k0+dk/R_max/2.0 #pi / pred_fast_t(dpsi, mu, dk, B, R)
    M_calc = M_calc_f(dpsi, mu, dk, B, R, A, k0)  #B*R*phi0inv #pred_slow_t(dpsi, mu, dk, B, R)

    # Обратите внимание, что x[0] - действительная часть, а x[1] - мнимая часть.

    F_re = (np.cos( M_calc *2 * pi * R) - np.cos(2*pi*k_calc*R)) / np.sin(2*pi*k_calc*R)
    F_im =  np.sin( M_calc *2 * pi * R) / np.sin(2*pi*k_calc*R)

    # Это дает правильные значения для psi_0 без e-e взаимодействия.
    root_real = x[0] - k_calc * F_re
    root_im = x[1] - M_calc + k_calc * F_im

    return [root_real, root_im]

'''
# Якобиан здесь не используется, поскольку он не был отлажен.
def jacobian_wrapper_bvp(x, mu, dk, B, R, A = 1., 
                         k0=kFAu, k_calc_f = pred_fast_k, M_calc_f = pred_slow_k_v3,
                         dk_calc_f = deriv_fast_k, dM_calc_f = deriv_slow_k_v3):
    #[[df0 / dx0,
    #df0/dx1],
    #[df1/dx0,
    #df1/dx1]])

    dpsi = x[0] + 1j * x[1]

    k_calc = k_calc_f(dpsi, mu, dk, B, R, A, k0)  #k0+dk/R_max/2.0 #pi / pred_fast_t(dpsi, mu, dk, B, R)
    M_calc = M_calc_f(dpsi, mu, dk, B, R, A, k0)  #B*R*phi0inv #pred_slow_t(dpsi, mu, dk, B, R)

    dkdx = dk_calc_f(x, mu, dk, B, R, A, k0)  #0#deriv_fast_k(x, mu, dk, B, R)
    dmdx = dM_calc_f(x, mu, dk, B, R, A, k0)  #0#deriv_slow_k(x, mu, dk, B, R)

    F_re = (np.cos( M_calc *2 * pi * R) - np.cos(2*pi*k_calc*R)) / np.sin(2*pi*k_calc*R)

    j00 = 1. + dkdx[0] * np.sin(2*pi*R*M_calc)/np.sin(2*pi*R*k_calc) 
    j00 = j00 + k_calc * dmdx[0] * np.cos(2*pi*R*M_calc)/np.sin(2*pi*R*k_calc) 
    j00 = j00 - k_calc * dkdx[0] * np.sin(2*pi*R*M_calc)*np.cos(2*pi*R*k_calc)/(np.sin(2*pi*R*k_calc)**2)

    j01 = 0.
    j01 = dkdx[1] * np.sin(2*pi*R*M_calc)/np.sin(2*pi*R*k_calc) 
    j01 = j01 + k_calc * dmdx[1] * np.cos(2*pi*R*M_calc)/np.sin(2*pi*R*k_calc) 
    j01 = j01 - k_calc * dkdx[1] * np.sin(2*pi*R*M_calc)*np.cos(2*pi*R*k_calc)/(np.sin(2*pi*R*k_calc)**2)

    j10 = 0.
    j10 = -dmdx[0] + dkdx[0] * F_re
    j10 = j10 + k_calc * (dkdx[0] * np.sin(2*pi*R*k_calc) - dmdx[0] * np.sin(2*pi*R*M_calc)) / np.sin(2*pi*R*k_calc) 
    j10 = j10 + k_calc * dkdx[0] * (np.cos(2*pi*R*M_calc)- np.cos(2*pi*R*k_calc)) *np.cos(2*pi*R*k_calc)/(np.sin(2*pi*R*k_calc)**2)

    j11 = 1. -dmdx[1] + dkdx[1] * F_re
    j11 = j11 + k_calc * (dkdx[1] * np.sin(2*pi*R*k_calc) - dmdx[1] * np.sin(2*pi*R*M_calc)) / np.sin(2*pi*R*k_calc) 
    j11 = j11 + k_calc * dkdx[1] * (np.cos(2*pi*R*M_calc)- np.cos(2*pi*R*k_calc)) *np.cos(2*pi*R*k_calc)/(np.sin(2*pi*R*k_calc)**2)

    return np.array([[j00,j01],[j10,j11]])
'''

#   -- Это функция, которая найдет одно из решений нелинейной краевой задачи, а также решение линейной краевой 
#      задачи. Если значение check_solution равно false, то оно вернет только решение нелинейной краевой задачи 
#      в виде массива из двух действительных чисел с действительной и мнимой составляющими начальной производной 
#      от psi.  Если значение check_solution равно true, то оно вернет 4 значения:
#          - Комплексное решение для начальной производной от psi для линейного случая.
#          - Значение функции, для которого нужно выяснить нуль, для этой (линейной) производной.
#          - Комплексное решение для начальной производной от psi для нелинейного случая.
#          - Значение функции, для которого нужно выяснить нуль, для этой (нелинейной) производной.
#   -- Нашими входными данными являются:
#       - dk, R, B, mu, в реальных единицах, как получено из нашего цикла.
#       - method - Алгоритм поиска нуля из списка доступных алгоритмов SciPy.
#       - tol - Допуск для этого алгоритма.
#       - ratio - Первоначальное предположение соответствует линейному решению в виде dphi0 / ratio.
#       - model_k and model_M - это модели k and M для функции, для которой нужно найти нули.
def find_root_both(dk, R, B, mu, method = 'broyden2', tol = 1e-20, ratio = 1.0,
                   check_solution = False, model_k=pred_fast_k_true, model_M = pred_slow_k_v3):

    # Найти нуль линейного уравнения. Существует только одно, поэтому метод и первоначальное предположение не важные.
    yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=k_calc_0, M_calc_f=M_calc_0)
    #jj = lambda x: jac(x, mu, dk, B, R, k_calc_f=k_calc_0, M_calc_f=M_calc_0, dk_calc_f=dk_calc_0, dM_calc_f=dM_calc_0)

    sol = optimize.root(yy, [-1e12, 1e12], method='hybr')
    xs = sol.x

    out_0 = yy(xs)

    # Теперь найти нуль для нелинейного решения.
    yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=model_k, M_calc_f=model_M)
    #jj = lambda x: jac(x, mu, dk, B, R)

    sol = optimize.root(yy, xs/ratio, method=method, tol=tol)
    dphi0c = sol.x

    deriv_psi = dphi0c[0] + 1j * dphi0c[1]
    deriv_psi_o = xs[0]+1j*xs[1]

    # И верните ожидаемые решения.
    if check_solution:
        return deriv_psi_o, out_0, deriv_psi, yy(dphi0c)
    else:
        return dphi0c
    