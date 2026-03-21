# bvp_test_script.py

# Автор: Элизабет Гоулд
# Дата последнего изменения: 18.03.2026

# test_match(dk, R2, B2, mu)
#    -- dk, R2, B2, и mu -- это параметры кольца, которые мы передаем для 
#       создания loop.
#    -- R2 - это параметр R, а B2 - это параметр B.
#    -- Обратите внимание, что R2 и B2 здесь - это проценты, подобно тем,
#       которые используются для создания объекта loop.

# Это моя полная подпрограмма для создания графиков решения краевых задач.
# Показанные графики будут для действительной части волновой функции, где она
# соответствует за максимумами модуля быстрых колебаний. Следовательно,
# построенная функция будет отображаться на графике медленных колебаний.
# Существует два возможных графических решения для линейного случая и четыре
# для нелинейного. В обоих случаях,
#  - psi_slow(0) = psi_slow(2*pi*R) или
#  - psi_slow(0) = - psi_slow(2*pi*R).
# Второй случай возникает, если согласование быстрых колебаний меняет знак,
# поскольку для этого согласования требуется только равная амплитуда. 
# Для линейного случая производные от модуля в начале и в конце должны иметь
# противоположные знаки.
# Для нелинейного случая производная также может совпадать с конечным решением,
# поскольку dk не выбран для предотвращения этого случая, поскольку это не
# создает столь серьезной двусмысленности в соответствующем решении.

# Выходные данные будут отображены на экране.

import numpy as np
from scipy import optimize
from eelib.consts import pi, B_max, R_max #, kFAu, phi0inv
from eelib.k_M_models_ivp import pred_fast_k, pred_slow_k_v3, pred_fast_k_true
from eelib.fitted_functions import fit_sin
from eelib.bvp_rootfinder_functions import find_root_both
from eelib.loop import loop
import matplotlib.pyplot as plt
#from eelib.bvp_rootfinder_functions import function_wrapper_bvp, k_calc_0, M_calc_0

# Это подпрограмма для соединения нелинейного решения на границе с
# использованием модели для k и M и вывода результатов на экран. Найденный нуль
# соединения уравнения является фактически случайным (хаотичным, а не
# случайным). Есть много решений. 
# Программа начнет поиск корней вокруг линейного решения, но, в зависимости от
# алгоритма, решение может оказаться не самым близким.
def test_match(dk, R2, B2, mu):
    R  = R2 * R_max
    B  = B2 * B_max

    # Найти нули
    deriv_psi_o, xs_sol, deriv_psi, dpsi_sol = find_root_both(dk, R, B, mu, model_k=pred_fast_k, ratio = 10, check_solution = True)
    print("Нуль линейного решения:", deriv_psi_o) 
    print("Значение соединяющей функции в этом нуле (должно быть равно 0):", xs_sol)
    print("Нуль нелинейного решения:", deriv_psi)
    print("Значение соединяющей функции в этом нуле (должно быть равно 0):", dpsi_sol)

    # Выходные данные разделяют действительную и мнимую части нашей производной
    # psi_0, как если бы они были отдельными переменными. Нам нужно объединить
    # их в одно комплексное число.
    #deriv_psi = dpsi0[0] + 1j * dpsi0[1]
    #deriv_psi_o = xs[0]+1j*xs[1]

    # Определить объект цикла для использования с этим нулем
    l_calc = loop(R2, B2, dk, mu)

    print("") # новая строка

    # a, b для аналитичных и численных линейных решений
    # Обычно они очень близки.
    print("aj, bj для точного линейного решения: ", l_calc.aj, l_calc.bj)
    l_calc.setDeriv(deriv_psi_o)
    print("aj, bj для найденного алгоритмом линейного решения: ", l_calc.aj, l_calc.bj)

    # Теперь перейдем к нелинейному решению краевых задач
    l_calc.setDeriv(deriv_psi)

    print("") # новая строка

    # Проверить, насколько хорошо согласуется результат, используя модель для
    # k и M, а не алгоритм для решения ОДУ.
    print("Конечная точка точного решения:", l_calc.psij0(2*pi*R2 * R_max))
    print("Конечная точка нового решения:", l_calc.psij_pred(2*pi*R2 * R_max), "модуль =", np.abs(l_calc.psij_pred(2*pi*R2 * R_max)))

    # линейные и нелинейные коэффициенты 
    print("aj, bj:", l_calc.aj, l_calc.bj)
    print("aj0, bj0 (линейное решение):", l_calc.aj0, l_calc.bj0)

    print("") # новая строка

    # вычисления тока
    cur_new = l_calc.current_alt()
    cur_old = l_calc.current_old()

    print("Нелинейная оценка тока:", cur_new)
    print("Линейная оценка тока:", cur_old)

    # Теперь все выходные данные являются графическими, а не текстовыми.

    # Метод для решения ОДУ с нашим решением краевой задаче для визуального
    # анализа согласия
    l_calc.solve_ivp(n = 1000, solve=1)
    solu = l_calc.solu

    # Весь приведенный ниже код является копией кода из кода для построения
    # производной сетки. Код извлекает x и psi из нашего кольца (loop) для
    # всего, что мы хотим построить.

    # Начальные параметры
    T_arr = l_calc.find_period_shift_exact() # быстрое, медленное, 
                                             # положительное, отрицательное
    trim = 16
    n = 1000

    # Это наше точное решение, соответствующее огибающей быстрых колебаний.
    sutc = l_calc.find_t_points(n, l_calc.lngt, l_calc.stu_ex, T_arr[0])
    sltc = l_calc.find_t_points(n, l_calc.lngt, l_calc.stl_ex, T_arr[0])
    suyc = l_calc.psij0(sutc)
    slyc = l_calc.psij0(sltc)

    # Найти смоделированное волновое число.
    M_pred = 2*pi / l_calc.T_slow_mod

    # Привести все параметры, кроме волнового числа, к синусу.
    sol_t = np.real(solu['t']) 
    t_pred = sol_t
    fit_func = fit_sin(solu)
    A_pred = fit_func[0]
    theta = fit_func[2]
    y_pred = A_pred * np.sin(M_pred * (sol_t) + theta)

    # массивы положений
    # оценено, вызвано модулем psi, вызвано действительней частью psi -- c, a, r
    tuerr = np.real(solu['t_events'][1][0::trim])
    tlerr = np.real(solu['t_events'][1][1::trim])
    tuera = np.real(solu['t_events'][0][0::trim])
    tlera = np.real(solu['t_events'][0][1::trim])
            
    # массивы значений
    suerr = np.real(solu['y_events'][1][0::trim, 0])
    slerr = np.real(solu['y_events'][1][1::trim, 0])
    suera = np.real(solu['y_events'][0][0::trim, 0])
    slera = np.real(solu['y_events'][0][1::trim, 0])

    # Удалить мнимую часть
    tu0xc = np.real(sutc)
    tl0xc = np.real(sltc)
    su0xc = np.real(suyc)
    sl0xc = np.real(slyc)

    # Первый график
    fig, ax = plt.subplots()
    plt.title(f"Re(\u03C8(x)), для максимумов |\u03C8(x)|, dk={dk}, R={R2}, B={B2}, \u03BC={mu}")
    ax.set_ylabel('Re(\u03C8) (\u03C8(0) = 1)')
    ax.set_xlabel('x (м)')

    h_list = []

    if np.max(suera) > np.max(slera):
        line1, = ax.plot(tuera, suera, color = 'red', label = 'с e-e взаимодействием')
    else:
        line1, = ax.plot(tlera, slera, color = 'red', label = 'с e-e взаимодействием')
    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'предсказанное, с e-e взаимодействие')
    h_list.append(line1)
    h_list.append(line13)
    if np.max(su0xc) > np.max(sl0xc):
        line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'точное решение')
    else:
        line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'точное решение')
    h_list.append(line9)

    ax.legend(handles=h_list)
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

    # Второй график показывает только прогнозируемое значение.
    fig, ax = plt.subplots()
    plt.title(f"График смоделированных медленных колебаний Re(\u03C8(x)), dk={dk}, R={R2}, B={B2}, \u03BC={mu}")
    ax.set_ylabel('Re(\u03C8) (\u03C8(0) = 1)')
    ax.set_xlabel('x (м)')

    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'предсказанное, с e-e взаимодействие')
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()

    # Третий график
    # огибающая, с точками, вызванными действительней частью psi
    fig, ax = plt.subplots()
    plt.title(f"Re(\u03C8(x)) огибающая, dk={dk}, R={R2}, B={B2}, \u03BC={mu}")
    ax.set_ylabel('Re(\u03C8) (\u03C8(0) = 1)')
    ax.set_xlabel('x (м)')


    h_list = []

    line1, = ax.plot(tuerr, suerr, color = 'red', label = 'с e-e взаимодействием')
    line2, = ax.plot(tlerr, slerr, color = 'red', label = 'с e-e взаимодействием')
    line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'предсказанное, с e-e взаимодействие')

    h_list.append(line1)
    h_list.append(line13)

    if np.max(su0xc) > np.max(sl0xc):
        line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'точное решение')
    else:
        line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'точное решение')
    h_list.append(line9)

    ax.legend(handles=h_list)
    ax.set_box_aspect(2.0/3.5)
            
    plt.show()
    