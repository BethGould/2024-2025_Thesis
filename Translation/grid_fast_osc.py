# eelib/grid_fast_osc.py
#
# класс grid_fast_osc 
# Автор: Элизабет Гоулд
# Дата последнего изменения: 20.03.2026

"""
Этот код создает сетку решений начальной задачи, чтобы вычислить 
t_fast (= 2*pi /k), период быстрых колебаний. Здесь можно варьировать значения
R, B, k, dk, mu, A и psi'(0) и построить либо сетку значений, либо сгенерировать
набор случайных значений. 

Сетка psi'(0) расположена синусоидально. Сетка mu разнесена логарифмически. 
Остальные сетки разнесены линейно. Одним из результатов этого является то, что
mu берется как показатель степени 10 для сетки, в то время как для прогонов по
методу Монте-Карло оно берется как исходное число. 
"""

# Код дла запуски на сетке:
# > gridl = eelib.grid_fast_osc(R, B, dk, mu)
# > gridl.makeGridPoints(mu=mu_r, B=b_r, num = [n])
# > gridl.gridFastOsc()

# Код дла запуска по методу Монте-Карло:
# > gridl = eelib.grid_fast_osc(R, B, dk, mu)
# > gridl.makeMCPoints(mu=mu_r, B=b_r, num = n)
# > gridl.mcFastOsc()

# __init__(self, R=1.0, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.) --
#     Создает объект loop с заданными параметрами. Переданные значения 
#     параметров актуальны только для тех параметров, которые не подлежат
#     варьированию, поскольку эти параметры будут заменены во время вычислений.

# __repr__(self)
# __str__(self)
#       Печатают описание объекта. 

# clear_calcs(self) -- Удаляет данные. 

# setIntegratorParameters(self, solve_mu_0 = None, n_sm = None, method = None,
#                         rtol = None, atol = None)
# Значения параметров по умолчанию следующие: solve_mu_0 = True, n_sm = 20, 
#                                    method = 'RK45', rtol = rtol, atol = atol.
# n_sm подсчитывает, сколько колебаний |Psi|^2 необходимо подсчитать, чтобы
# оценить t / 2 (что равно pi / k).
# Если solve_mu_0 = True, код также оценит быстрые колебания для случая без
# взаимодействия с ee.
# Не заданные параметры изменены не будут.

# makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0),
#                R = (-1.0, -1.0), A = (-1.0, -1.0), k0 = (-1.0, -1.0),
#                ang_lim = 0.1, num = [10])
# makeMCPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0),
#          R = (-1.0, -1.0), A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = 1000)

# Эти функции определяют различные значения параметров, для анализа начальной
# задачи. Если набор точек не включен, он не будет изменен, вместо этого
# будет использоваться значение параметра, переданного при инициализации.
# В противном случае значения будут изменяться от заданного минимального
# значения до заданного максимального. 
#   !!! Примечание - значение mu изменяется на основе показателей, равных 10
#       для точек сетки, но как чистое число для анализа методом Монте-Карло.
#       Я знаю, что это проблема, но так все записано.
# num - положительное целое число, указывающее количество точек для метода
# Монте-Карло.
# Для сетки num должно быть списком натуральных чисел. При размере списка,
# равном единице, все измерения сетки будут равны этому числу для num^n точек.
# Вы можете отправить два числа, чтобы указать отдельные длины для производных
# и других чисел, или несколько, чтобы указать значение для каждого параметра
# отдельно. (Размер для действительной и мнимой производной должны быть равны,
# и они никогда не будут принимать два параметра, только один.)

# gridFastOsc(self) выполняет вычисление, если задана сетка.
# mcFastOsc(self) выполняет вычисление, если задан случайный набор точек для 
#                 запуска.
# runCalc(self) выполняет вычисление в любом случае.

# Результаты сохраняются в следующих переменных:
# self.fast_osc_t = период быстрых колебаний
# self.fast_osc_a = амплитуда быстрых колебаний
# self.fast_osc_t_0 = период быстрых колебаний, вычислен для линейного случай,
#                     чтобы было возможно сравнивать с точным решением и
#                     узнать ошибку

# Для варианта запуска на сетке сохраненные массивы отображаются 
# следующим образом: [n_mu, n_dk, n_b, n_r, n_a, n_k0, n_dr, n_di]

# Параметры сохраняются в следующих переменных.
# - Таблица Монте-Карло:
# self.val_table    -- Таблица случайных значений параметров для
#                      варианта Монте-Карло.
# - Таблицы сетки:
# напряженность магнитного поля = self.mfs
# волновое число электрона      = self.ew
# дифференциал волнового числа электрона = self.ewd
# сила нелинейности             = self.nls
# начальная амплитуда           = self.amp
# радиус кольца                 = self.rr
# сетка начальных производных волновой функции = self.d0_grid[idr,idi]
# действительная часть начальной производной волновой функции = self.dr
# мнимой часть начальной производной волновой функции         = self.di

import numpy as np
from eelib.consts import pi, kFAu, rtol, atol
from eelib.loop import loop
import time
import gc

class grid_fast_osc:

    # ------ ИНИЦИАЛИЗАЦИЯ --------------

    # R, B, dk, mu, k и amp должны быть указаны, если они фиксированные.
    # k по умолчанию равно тому у золота, amp по умолчанию равно 1.
    def __init__(self, R=0.5, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.):
        self.l_calc = loop(R, B, dk, mu, k, amp)

        self.R = R
        self.B = B
        self.dk = dk
        self.k = k
        self.mu = mu
        self.a = amp

        self.is_grid = False
        self.is_mc = False
        self.calculated = False

        self.setIntegratorParameters(solve_mu_0 = True, n_sm = 20, method = 'RK45', rtol = rtol, atol = atol)

    # Существует множество параметров для интегратора, которые я обычно просто 
    # устанавливаю по умолчанию и игнорирую. Здесь их можно задать один раз и 
    # использовать каждый раз.
    def setIntegratorParameters(self, solve_mu_0 = None, n_sm = None, method = None, rtol = None, atol = None):
        if solve_mu_0 is not None:
            self.l_calc.solve_mu_0 = solve_mu_0
        if n_sm is not None:
            self.n_sm = n_sm
        if method is not None:
            self.method = method
        if rtol is not None:
            self.rtol = rtol
        if atol is not None:
            self.atol = atol

    # ------------ ВЫВОД СТРОКИ ---------------
    # Выходные данные для инструкции print.
    def __repr__(self):
        if self.is_grid:
            if self.calculated:
                str = "Сеточный объект для измерения быстрых колебаний:\n"
            else:
                str = "Нерасчетный сеточный объект для измерения быстрых колебаний:\n"

            # Показать параметры сетки.
            if self.num_mu > 1:
                str = str + f"mu имеет {self.num_mu} точки от {self.nls[0]} до {self.nls[-1]}.\n"
            else:
                str = str + f"mu: {self.mu}\n"
            if self.num_dk > 1:
                str = str + f"dk имеет {self.num_dk} точки от {self.ewd[0]} до {self.ewd[-1]}.\n"
            else:
                str = str + f"dk: {self.dk}\n"
            if self.num_b > 1:
                str = str + f"B имеет {self.num_b} точки от {self.mfs[0]} до {self.mfs[-1]}.\n"
            else:
                str = str + f"B: {self.B}\n"
            if self.num_r > 1:
                str = str + f"R имеет {self.num_r} точки от {self.rr[0]} до {self.rr[-1]}.\n"
            else:
                str = str + f"R: {self.R}\n"
            if self.num_a > 1:
                str = str + f"A имеет {self.num_a} точки от {self.amp[0]} до {self.amp[-1]}.\n"
            else:
                str = str + f"A: {self.amp}\n"
            if self.num_k0 > 1:
                str = str + f"k0 имеет {self.num_k0} точки от {self.ew[0]} до {self.ew[-1]}.\n"
            else:
                str = str + f"k0: {self.k}\n"
            str = str + f"Размер сетки производный: {self.grid_size}\n"
            str = str + f"Общее количество точек: {self.num_mu*self.num_dk*self.num_k0*self.num_b*self.num_a*self.num_r*self.grid_size*self.grid_size}"
            return str
        elif self.is_mc:
            if self.calculated:
                str = f"Объект Монте-Карло для измерения быстрых колебаний:\n"
            else:
                str = f"Нерасчетный объект Монте-Карло для измерения быстрых колебаний:\n"
            # Следующий выход не является элегантным.
            str = str + f"mu имеет точки от {np.min(self.val_table[:,0])} до {np.max(self.val_table[:,0])}.\n"
            str = str + f"dk имеет точки от {np.min(self.val_table[:,1])} до {np.max(self.val_table[:,1])}.\n"
            str = str + f"B имеет точки от {np.min(self.val_table[:,2])} до {np.max(self.val_table[:,2])}.\n"
            str = str + f"R имеет точки от {np.min(self.val_table[:,3])} до {np.max(self.val_table[:,3])}.\n"
            str = str + f"A имеет точки от {np.min(self.val_table[:,4])} до {np.max(self.val_table[:,4])}.\n"
            str = str + f"k0 имеет точки от {np.min(self.val_table[:,5])} до {np.max(self.val_table[:,5])}.\n"
            str = str + f"Количество точек: {self.val_table.shape[0]}"
            return str
        else:
            return "Пустой объект для измерения быстрых колебаний:\nR - %s, B - %s, dk - %s, k - %s, mu - %s, A - %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # Выходные данные для инструкции print.
    def __str__(self):
        if self.is_grid:
            if self.calculated:
                str = "Сеточный объект для измерения быстрых колебаний:\n"
            else:
                str = "Нерасчетный сеточный объект для измерения быстрых колебаний:\n"

            #  Показать параметры сетки.
            if self.num_mu > 1:
                str = str + f"mu имеет {self.num_mu} точки от {self.nls[0]} до {self.nls[-1]}.\n"
            else:
                str = str + f"mu: {self.mu}\n"
            if self.num_dk > 1:
                str = str + f"dk имеет {self.num_dk} точки от {self.ewd[0]} до {self.ewd[-1]}.\n"
            else:
                str = str + f"dk: {self.dk}\n"
            if self.num_b > 1:
                str = str + f"B имеет {self.num_b} точки от {self.mfs[0]} до {self.mfs[-1]}.\n"
            else:
                str = str + f"B: {self.B}\n"
            if self.num_r > 1:
                str = str + f"R имеет {self.num_r} точки от {self.rr[0]} до {self.rr[-1]}.\n"
            else:
                str = str + f"R: {self.R}\n"
            if self.num_a > 1:
                str = str + f"A имеет {self.num_a} точки от {self.amp[0]} до {self.amp[-1]}.\n"
            else:
                str = str + f"A: {self.amp}\n"
            if self.num_k0 > 1:
                str = str + f"k0 имеет {self.num_k0} точки от {self.ew[0]} до {self.ew[-1]}.\n"
            else:
                str = str + f"k0: {self.k}\n"
            str = str + f"Размер сетки производный: {self.grid_size}\n"
            str = str + f"Общее количество точек: {self.num_mu*self.num_dk*self.num_k0*self.num_b*self.num_a*self.num_r*self.grid_size*self.grid_size}"
            return str
        elif self.is_mc:
            if self.calculated:
                str = f"Объект Монте-Карло для измерения быстрых колебаний:\n"
            else:
                str = f"Нерасчетный объект Монте-Карло для измерения быстрых колебаний:\n"
            # Следующий выход не является элегантным.
            str = str + f"mu имеет точки от {np.min(self.val_table[:,0])} до {np.max(self.val_table[:,0])}.\n"
            str = str + f"dk имеет точки от {np.min(self.val_table[:,1])} до {np.max(self.val_table[:,1])}.\n"
            str = str + f"B имеет точки от {np.min(self.val_table[:,2])} до {np.max(self.val_table[:,2])}.\n"
            str = str + f"R имеет точки от {np.min(self.val_table[:,3])} до {np.max(self.val_table[:,3])}.\n"
            str = str + f"A имеет точки от {np.min(self.val_table[:,4])} до {np.max(self.val_table[:,4])}.\n"
            str = str + f"k0 имеет точки от {np.min(self.val_table[:,5])} до {np.max(self.val_table[:,5])}.\n"
            str = str + f"Количество точек: {self.val_table.shape[0]}"
            return str
        else:
            return "Пустой объект для измерения быстрых колебаний:\nR - %s, B - %s, dk - %s, k - %s, mu - %s, A - %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # ------------ ВЫБОР ТОЧЕК (СЕТКА) -----------------

    # def makeGridPoints(self, m_min, m_max, k_min, k_max, a_min, a_max, mu_min,
    #                    mu_max, num = 10):
    # В порядке -- mu, dk, B, R, M, A, k0
    # ang_lim используется для того, чтобы избежать нулевых значений там, где
    # это может быть проблемой.
    # размер num -- 1, 2 (длина для большинства перемен и для производных),
    #               или n = число определенных переменных + 1 для производных
    def makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0),
                       R = (-1.0, -1.0), A = (-1.0, -1.0), k0 = (-1.0, -1.0),
                       ang_lim = 0.1, num = [10]):
        
        # Определяет, с какими параметрами строить сетку.
        # Для dk мне нужно что-то еще, чтобы указать, что это не нужно, 
        # поскольку теоретически могут быть полезны отрицательные значения.
        # Однако я предполагаю, что dk принимает значение от 0 до 1.
        # На этом шаге можно выполнить более широкий спектр проверок на ошибки,
        # что ограничивает наши диапазоны.
        # Я не знаю, почему я просто не использовала None.
        var_set = [False,False,False,False,False,False]
        if mu[0] <  0.0: var_set[0] = True
        if dk[0] > -0.1: var_set[1] = True
        if B[0]  > -0.1: var_set[2] = True
        if R[0]  > -0.1: var_set[3] = True
        if A[0]  > -0.1: var_set[4] = True
        if k0[0] > -0.1: var_set[5] = True

        self.var_set = var_set # чтобы можно пользоваться позже

        # Указать количество точек сетки, которые будут использоваться, 
        # в нашем объекте.
        self.set_nums(var_set, num)

        # сила нелинейность / mu
        if var_set[0]:
            self.nls = np.logspace(mu[0], mu[1], num=self.num_mu)
        else: 
            self.nls = np.array([self.mu])

        # волновое число электрона / k
        if var_set[1]:
            self.ewd = np.linspace(dk[0], dk[1], num=self.num_dk)
        else:
            self.ewd = np.array([self.dk])
        if var_set[5]:
            self.ew = np.linspace(k0[0], k0[1], num=self.num_k0)
        else:
            self.ew = np.array([self.k])

        # колебания, обусловленные магнитным полем, напряженность магнитного поля 
        # и радиус кольца
        if var_set[2]:
            self.mfs = np.linspace(B[0], B[1], num=self.num_b)
        else:
            self.mfs = np.array([self.B])
        if var_set[3]:
            self.rr = np.linspace(R[0], R[1], num=self.num_r)
        else:
            self.rr = np.array([self.R])

        # начальные значения psi, psi'
        if var_set[4]:
            self.amp = np.linspace(A[0], A[1], num=self.num_a)
        else:
            self.amp = np.array([self.a])

        self.dlim = self.l_calc.psi_prime_0_max()
        self.ang_lim = ang_lim
        self.makeDerivPoints()

        if self.calculated: self.clear_calcs()

        self.is_grid = True
        self.is_mc = False
        self.calculated = False

        gc.collect()

    # Это разделено для наглядности предыдущего кода, поскольку он длинный и 
    # повторяющийся. Его нельзя запускать независимо.
    def set_nums(self, var_set, num):

        self.num_mu = 1
        self.num_a = 1
        self.num_b = 1
        self.num_r = 1
        self.num_dk = 1
        self.num_k0 = 1

        if len(num) < 1: 
            raise ValueError("Количество точек не указано.")
        elif len(num) == 1:
            self.grid_size = num[0]
            self.num = num[0]
            if var_set[0]: self.num_mu = num[0]
            if var_set[4]: self.num_a  = num[0]
            if var_set[2]: self.num_b  = num[0]
            if var_set[3]: self.num_r  = num[0]
            if var_set[1]: self.num_dk = num[0]
            if var_set[5]: self.num_k0 = num[0]
        # отделение производной от других параметров
        elif len(num) == 2:
            if max(var_set) == False: 
                raise ValueError("Ожидается только одно значение для num, так как будет построена только сетка производных.")
            self.grid_size = num[1]
            self.num = num[0]
            if var_set[0]: self.num_mu = num[0]
            if var_set[4]: self.num_a  = num[0]
            if var_set[2]: self.num_b  = num[0]
            if var_set[3]: self.num_r  = num[0]
            if var_set[1]: self.num_dk = num[0]
            if var_set[5]: self.num_k0 = num[0]
        # полный набор num
        else:
            if max(var_set) == False: 
                raise ValueError("Ожидается только одно значение для num, так как будет построена только сетка производных.")
            i = 0
            j = 0
            ln = len(num)
            if var_set[i]: 
                self.num_mu = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                self.num_dk = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                self.num_b = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                if j == ln: 
                    raise ValueError(f"Передано недостаточно значений для num. Остановились на параметре index {i} (R) и массиве чисел {j}.")
                self.num_r = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                if j == ln: 
                    raise ValueError(f"Передано недостаточно значений для num. Остановились на параметре index {i} (A0) и массиве чисел {j}.")
                self.num_a = num[j]
                j += 1
            i += 1
            if var_set[i]: 
                if j == ln: 
                    raise ValueError(f"Передано недостаточно значений для num. Остановились на параметре index {i} (k0) и массиве чисел {j}.")
                self.num_k0 = num[j]
                j += 1
            i += 1

            if j == ln: 
                raise ValueError(f"Передано недостаточно значений для num. Остановились на параметре index {i} (производные) и массиве чисел {j}.")
            self.grid_size = num[j]
            j += 1
            if j != ln:
                raise ValueError(f"Передано слишком много значений для num. j = {j}, i = {i}, ln = {ln}")

    # формирование сетки производных
    # Входы:
    #       self.ang_lim   -- Должна оставаться постоянным, так как она не 
    #                         допускает нулевых значений. 
    #       self.dlim      -- Должна изменяться и записываться в массив. 
    #                         Масштаб производной.
    #       self.grid_size -- Должна оставаться постоянным, устанавливается
    #                         при запуске.
    def makeDerivPoints(self):
        # Из сетки производных для t = 0.
        # Обратите внимание, что при этом используются заданные значения для
        # выбора идеальных точек. Вместо этого точки должны быть основаны на
        # других значениях.    
        x = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        x = np.cos(x) * self.dlim 
        y = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        y = np.cos(y) * self.dlim
        xx, yy = np.meshgrid(x, y)
        d0_grid = xx + 1j * yy

        d0_grid = d0_grid.T

        self.d0_grid = d0_grid

    # Очистить память
    def clear_calcs(self):
        self.fast_osc_t   = None
        self.fast_osc_t_0 = None
        self.fast_osc_a   = None
        gc.collect()


    # ------------ ВЫБОР ТОЧЕК (МОНТЕ КАРЛО) -----------------

    # Точки по методу Монте-Карло принимают диапазоны параметров, как в сетке,
    # но поскольку они случайны, вместо построения сетки будет выведен массив
    # случайных точек. num - это полное количество точек, а не для каждого
    # параметра, как в случае с сеткой.
    def makeMCPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), R = (-1.0, -1.0), 
                       A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = 1000):
        
        # Определяет, с какими параметрами строить массив.
        # Для dk мне нужно что-то еще, чтобы указать, что это не нужно, 
        # поскольку теоретически могут быть полезны отрицательные значения.
        # Однако я предполагаю, что dk принимает значение от 0 до 1.
        # На этом шаге можно выполнить более широкий спектр проверок на ошибки,
        # что ограничивает наши диапазоны.
        var_set = [False,False,False,False,False,False]
        if mu[0] > -0.1: var_set[0] = True
        if dk[0] > -0.1: var_set[1] = True
        if B[0]  > -0.1: var_set[2] = True
        if R[0]  > -0.1: var_set[3] = True
        if A[0]  > -0.1: var_set[4] = True
        if k0[0] > -0.1: var_set[5] = True

        self.var_set = var_set # чтобы можно пользоваться позже

        # Создать таблицу значений. 
        self.num = num
        self.val_table = np.zeros((num, 8))

        rng = np.random.default_rng()

        # сила нелинейность / mu
        if var_set[0]:
            for ii in range(num):
                self.val_table[ii,0] = mu[0] + rng.random()*(mu[1]-mu[0])
        else: 
            self.val_table[:,0] = self.mu

        # волновое число электрона / k
        if var_set[1]:
            for ii in range(num):
                self.val_table[ii,1] = dk[0] + rng.random()*(dk[1]-dk[0])
        else: 
            self.val_table[:,1] = self.dk
        if var_set[5]:
            for ii in range(num):
                self.val_table[ii,5] = k0[0] + rng.random()*(k0[1]-k0[0])
        else:
            self.val_table[:,5] = self.k

        # колебания, обусловленные магнитным полем, напряженность магнитного поля 
        # и радиус кольца
        if var_set[2]:
            for ii in range(num):
                self.val_table[ii,2] = B[0] + rng.random()*(B[1]-B[0])
        else:
            self.val_table[:,2] = self.B
        if var_set[3]:
            for ii in range(num):
                self.val_table[ii,3] = R[0] + rng.random()*(R[1]-R[0])
        else:
            self.val_table[:,3] = self.R

        # начальные значения psi, psi'
        if var_set[4]:
            for ii in range(num):
                self.val_table[ii,4] = A[0] + rng.random()*(A[1]-A[0])
        else:
            self.val_table[:,4] = self.a

        self.dlim = self.l_calc.psi_prime_0_max()

        for ii in range(num):
            self.val_table[ii,6] = -self.dlim + rng.random()*2*self.dlim
            self.val_table[ii,7] = -self.dlim + rng.random()*2*self.dlim

        # Освободите память после очистки старых данных, так как они больше 
        # недействительны.
        if self.calculated: self.clear_calcs()

        self.is_grid = False
        self.is_mc = True
        self.calculated = False

        gc.collect()


    # ------------- ПОИСК ЗНАЧЕНИЙ НА СЕТКЕ ---------------

    # Дескриптор для следующих двух метода для запуска кода.
    def runCalc(self):
        if self.is_grid: 
            self.gridFastOsc()
        elif self.is_mc: 
            self.mcFastOsc()
        else: 
            print("Точки для вычисления еще не заданы.")

    # Выполнить вычисление по сетке.
    # Это требует времени, но больше не требует параметров. Из-за этого 
    # требования оно не связано с параметрами.
    def gridFastOsc(self):

        # Старое вычисление уже существует.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # Сетка не задана.
        if not self.is_grid:
            raise Exception("Сначала для gridFastOsc требуется сформировать сетку. Попробуйте вызвать makeGridPoints.") 

        # Синхронизация
        start_time = time.time()
        print('Начать построение сетки: ', time.time() - start_time)

        # Инициализация сетки
        num_dk = self.num_dk
        num_k0 = self.num_k0
        num_b = self.num_b
        num_r = self.num_r
        num_mu = self.num_mu
        num_a = self.num_a
        num_d = self.grid_size

        # напряженность магнитного поля = self.mfs
        # волновое число электрона      = self.ew
        # дифференциал волнового числа электрона = self.ewd
        # сила нелинейности             = self.nls
        # начальная амплитуда           = self.amp
        # радиус кольца                 = self.rr
        # сетка начальных производных волновой функции = self.d0_grid[idr,idi]
        # действительная часть начальной производной волновой функции = self.dr
        # мнимой часть начальной производной волновой функции         = self.di

        # сохраненные данные
        fast_oscillation_period = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        fast_oscillation_amplitude = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        fast_oscillation_period_0 = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))

        # Индексы для каждого параметра, который меняется.
        ib = 0
        ir = 0
        im = 0
        ik = 0
        ik0 = 0
        ia = 0
        idr = 0
        idi = 0

        # Решить для каждой точки на сетке.
        # 8 уровней являются проблемой только в том случае, если все они
        # используются.
        print("Количество точек для вычисления:", num_dk*num_k0*num_b*num_r*num_mu*num_a*num_d*num_d)
        for im in range(num_mu):
            for ik in range(num_dk):
                print(f"mu: {im}, dk: {ik}, время: {time.time() - start_time}")
                for ib in range(num_b):
                    for ir in range(num_r):    
                        for ia in range(num_a):
                            for ik0 in range(num_k0):
                                self.l_calc.update_params(R=self.rr[ir], B=self.mfs[ib], dk=self.ewd[ik], 
                                                  mu=self.nls[im], k = self.ew[ik0], amp=self.amp[ia])
                                # И для производной.
                                for idr in range(num_d):
                                    for idi in range(num_d):
                                        # Вычислить производную от параметров, если она
                                        # не соответствует заданным.
                                        self.l_calc.setDeriv(self.d0_grid[idr,idi])
                                        # Оценить k. 
                                        self.l_calc.find_fast_oscillations(self.n_sm, method = self.method, 
                                                                           rtol = self.rtol, atol = self.atol)
                                        #print(self.l_calc.T_fast * 2, self.l_calc.A_max)
                                        fast_oscillation_period[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.T_fast * 2
                                        fast_oscillation_amplitude[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.A_max  
                                        fast_oscillation_period_0[im,ik,ib,ir,ia,ik0,idr,idi] = self.l_calc.T_fast_0_calc * 2

        # Преобразовать локальные переменные в объектные, сохраняя данные.
        self.fast_osc_t = fast_oscillation_period
        self.fast_osc_a = fast_oscillation_amplitude
        self.fast_osc_t_0 = fast_oscillation_period_0 # Чтобы узнать ошибку.

        # Указывать, что этот метод был запущен.
        self.calculated = True

        # И завершать синхронизацию.
        print("Построение сетки завершено: ", time.time() - start_time)

    # Запуск для случайно выбранных точек.
    def mcFastOsc(self):

        # Старое вычисление уже существует.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # Таблица не задана.
        if not self.is_mc:
            raise Exception("mcFastOsc требует сначала выбрать точки. Попробуйте вызвать makeMCPoints.") 

        # Синхронизация
        start_time = time.time()
        print('Начать построение Монте-Сарло: ', time.time() - start_time)

        # Узнать количество точек.
        num = self.num

        # напряженность магнитного поля = self.mfs
        # волновое число электрона      = self.ew
        # дифференциал волнового числа электрона = self.ewd
        # сила нелинейности             = self.nls
        # начальная амплитуда           = self.amp
        # радиус кольца                 = self.rr
        # сетка начальных производных волновой функции = self.d0_grid[idr,idi]
        # действительная часть начальной производной волновой функции = self.dr
        # мнимой часть начальной производной волновой функции         = self.di

        # self.val_table

        # результаты
        fast_oscillation_period = np.zeros((num))
        fast_oscillation_amplitude = np.zeros((num))
        fast_oscillation_period_0 = np.zeros((num))

        # Решить для каждой случайно определенной точки.
        print("Количество точек для вычисления:", num)
        for ii in range(num):
            self.l_calc.update_params(R=self.val_table[ii,3], B=self.val_table[ii,2], dk=self.val_table[ii,1], 
                                      mu=self.val_table[ii,0], k = self.val_table[ii,5], amp=self.val_table[ii,4])
            # Вычислить производную от параметров, если она не соответствует 
            # заданным
            self.l_calc.setDeriv(self.val_table[ii,6]+ 1.j *self.val_table[ii,7])
            # Оценить k
            self.l_calc.find_fast_oscillations(self.n_sm, method = self.method, rtol = self.rtol, atol = self.atol)

            # Сохранить результаты.
            fast_oscillation_period[ii] = self.l_calc.T_fast * 2
            fast_oscillation_amplitude[ii] = self.l_calc.A_max  
            fast_oscillation_period_0[ii] = self.l_calc.T_fast_0_calc * 2

        # Преобразовать локальные переменные в объектные, сохраняя данные.
        self.fast_osc_t = fast_oscillation_period
        self.fast_osc_a = fast_oscillation_amplitude
        self.fast_osc_t_0 = fast_oscillation_period_0 # Чтобы узнать ошибку.

        # Указывать, что этот метод был запущен.
        self.calculated = True

        # И завершать синхронизацию.
        print("Построение таблицы завершено: ", time.time() - start_time)