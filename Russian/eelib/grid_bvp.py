# eelib/grid_BVP.py
#
# класс grid_BVP
# Автор: Элизабет Гоулд
# Дата последнего изменения: 23.03.2026

"""
Этот код создает сетку решений краевой задачи, чтобы вычислить psi'(0),
первоначальная производная от psi. Здесь можно варьировать значения
R, B, k, dk, mu, и A и построить либо сетку значений, либо сгенерировать
набор случайных значений. 

Сетка mu разнесена логарифмически. Остальные сетки разнесены линейно. Одним из
результатов этого является то, что mu берется как показатель степени 10 для
сетки, в то время как для прогонов по методу Монте-Карло оно берется как
исходное число.
"""

# Код дла запуски на сетке:
# > gridl = eelib.grid_BVP(R, B, dk, mu)
# > gridl.makeGridPoints(mu=mu_r, B=b_r, num = [n])
# > gridl.gridBVP()

# Код дла запуска по методу Монте-Карло:
# > gridl = eelib.grid_BVP(R, B, dk, mu)
# > gridl.makeMCPoints(mu=mu_r, B=b_r, num = n)
# > gridl.mcBVP()

# __init__(self, R=1.0, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.) --
#     Создает объект loop с заданными параметрами. Переданные значения 
#     параметров актуальны только для тех параметров, которые не подлежат
#     варьированию, поскольку эти параметры будут заменены во время вычислений.

# __repr__(self)
# __str__(self)
#       Печатают описание объекта. 

# clear_calcs(self) -- Удаляет данные. 

# makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), 
#            R = (-1.0, -1.0), A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = [10])
# makeMCPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0),
#            R = (-1.0, -1.0), A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = 1000)

# Эти функции определяют различные значения параметров, для анализа краевой
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
# Можно отправить несколько, чтобы указать значение для каждого параметра
# отдельно.


# gridBVP(self) выполняет вычисление, если задана сетка.
# mcBVP(self) выполняет вычисление, если задан случайный набор точек для 
#             запуска.
# runCalc(self) выполняет вычисление в любом случае.

# Результаты сохраняются в следующих переменных:
# self.derivs выводит список словарей с вычисленными значениями. Данные
# могут быть получены с помощью pandas:
# > tbl = pd.DataFrame(gridl.derivs)

# Для варианта запуска на сетке сохраненные массивы отображаются 
# следующим образом: [n_mu, n_dk, n_b, n_r, n_a, n_k0]

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

import numpy as np
from eelib.consts import pi, kFAu, rtol, atol
from eelib.loop import loop
import time
import gc
import warnings

class grid_BVP:

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

    # ------------ ВЫВОД СТРОКИ ---------------
    # Выходные данные для инструкции print.
    def __repr__(self):
        if self.is_grid:
            if self.calculated:
                str = "Сеточный объект для измерения тока:\n"
            else:
                str = "Нерасчетный сеточный объект для измерения тока:\n"

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
            str = str + f"Общее количество точек: {self.num_mu*self.num_dk*self.num_k0*self.num_b*self.num_a*self.num_r}"
            return str
        elif self.is_mc:
            if self.calculated:
                str = f"Объект Монте-Карло для измерения тока:\n"
            else:
                str = f"Нерасчетный объект Монте-Карло для измерения тока:\n"
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
            return "Пустой объект для измерения тока:\nR - %s, B - %s, dk - %s, k - %s, mu - %s, A - %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # Выходные данные для инструкции print.
    def __str__(self):
        if self.is_grid:
            if self.calculated:
                str = "Сеточный объект для измерения тока:\n"
            else:
                str = "Нерасчетный сеточный объект для измерения тока:\n"

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
            str = str + f"Общее количество точек: {self.num_mu*self.num_dk*self.num_k0*self.num_b*self.num_a*self.num_r}"
            return str
        elif self.is_mc:
            if self.calculated:
                str = f"Объект Монте-Карло для измерения тока:\n"
            else:
                str = f"Нерасчетный объект Монте-Карло для измерения тока:\n"
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
            return "Пустой объект для измерения тока:\nR - %s, B - %s, dk - %s, k - %s, mu - %s, A - %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # ------------ ВЫБОР ТОЧЕК (СЕТКА) -----------------

    # def makeGridPoints(self, m_min, m_max, k_min, k_max, a_min, a_max, mu_min,
    #                    mu_max, num = 10):
    # В порядке -- mu, dk, B, R, M, A, k0
    # размер num -- 1 или n = определенные переменные
    def makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), 
                       B = (-1.0, -1.0), R = (-1.0, -1.0), 
                       A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = [10]):
        
        # Определяет, с какими параметрами строить сетку.
        # Для dk мне нужно что-то еще, чтобы указать, что это не нужно, 
        # поскольку теоретически могут быть полезны отрицательные значения.
        # Однако я предполагаю, что dk принимает значение от 0 до 1.
        # На этом шаге можно выполнить более широкий спектр проверок на ошибки,
        # что ограничивает диапазоны.
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
        # в объекте.
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

            if j != ln:
                raise ValueError(f"Передано слишком много значений для num. j = {j}, i = {i}, ln = {ln}")

    # Очистить память
    def clear_calcs(self):
        self.derivs   = None
        gc.collect()


    # ------------ ВЫБОР ТОЧЕК (МОНТЕ КАРЛО) -----------------

    # Точки по методу Монте-Карло принимают диапазоны параметров, как в сетке,
    # но поскольку они случайны, вместо построения сетки будет выведен массив
    # случайных точек. num - это полное количество точек, а не для каждого
    # параметра, как в случае с сеткой.
    def makeMCPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0),
            R = (-1.0, -1.0), A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = 1000):
        
        # Определяет, с какими параметрами строить массив.
        # Для dk мне нужно что-то еще, чтобы указать, что это не нужно, 
        # поскольку теоретически могут быть полезны отрицательные значения.
        # Однако я предполагаю, что dk принимает значение от 0 до 1.
        # На этом шаге можно выполнить более широкий спектр проверок на ошибки,
        # что ограничивает диапазоны.
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
        self.val_table = np.zeros((num, 6))

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

        # Освободить память после очистки старых данных, так как они больше 
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
            self.gridBVP()
        elif self.is_mc: 
            self.mcBVP()
        else: 
            print("Точки для вычисления еще не заданы.")

    # Выполнить вычисление по сетке.
    # Это требует времени, но больше не требует параметров. Из-за этого 
    # требования оно не связано с параметрами.
    def gridBVP(self):

        # Старое вычисление уже существует.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # Сетка не задана.
        if not self.is_grid:
            raise Exception("Сначала для gridBVP требуется сформировать сетку. Попробуйте вызвать makeGridPoints.") 

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

        # напряженность магнитного поля = self.mfs
        # волновое число электрона      = self.ew
        # дифференциал волнового числа электрона = self.ewd
        # сила нелинейности             = self.nls
        # начальная амплитуда           = self.amp
        # радиус кольца                 = self.rr

        #fast_oscillation_period = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0))
        #fast_oscillation_amplitude = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0))
        #fast_oscillation_period_0 = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0))

        # сохраненные данные
        found_derivs_array = []

        # Индексы для каждого параметра, который меняется.
        ib = 0
        ir = 0
        im = 0
        ik = 0
        ik0 = 0
        ia = 0

        # Решить для каждой точки на сетке.
        # 6 уровней являются проблемой только в том случае, если все они
        # используются.
        print("Количество точек для вычисления:", num_dk*num_k0*num_b*num_r*num_mu*num_a)
        for im in range(num_mu):
            for ik in range(num_dk):
                print(f"mu: {im}, dk: {ik}, время: {time.time() - start_time}")
                for ib in range(num_b):
                    for ir in range(num_r):    
                        for ia in range(num_a):
                            for ik0 in range(num_k0):
                                # Определить кольцо.
                                self.l_calc.update_params(R=self.rr[ir], 
                                        B=self.mfs[ib], dk=self.ewd[ik], 
                                        mu=self.nls[im], k = self.ew[ik0], 
                                        amp=self.amp[ia])
                                # Найти нули, которые решают краевую задачу;
                                # это та часть, которая требует времени.
                                with warnings.catch_warnings(action="ignore"):
                                    deriv_found = self.l_calc.find_root_many()
                                # Цифры здесь указаны для решения без e-e 
                                # взаимодействия.
                                psiprime0 = self.l_calc.psi_prime_0()
                                exactend = self.l_calc.psij0(2*pi*self.l_calc.R)
                                i0 = self.l_calc.current_old()
                                a0 = self.l_calc.aj0
                                b0 = self.l_calc.bj0
                                A_max_0 = self.l_calc.amp_max_0()
                                # Сохранить все данные в словарном объекте для
                                # использования с pandas.
                                for deriv in deriv_found:
                                    self.l_calc.setDeriv(deriv)
                                    new_element = {"R": self.rr[ir],
                                                   "B": self.mfs[ib],
                                                   "dk": self.ewd[ik],
                                                   "mu": self.nls[im],
                                                   "k": self.ew[ik0],
                                                   "A": self.amp[ia],
                                                   "dpsi0": psiprime0,
                                                   "Exact End": exactend,
                                                   "a0": a0,
                                                   "b0": b0,
                                                   "A max 0": A_max_0,
                                                   "I0": i0,
                                                   "dpsi": deriv,
                                                   "New End": self.l_calc.psij_pred_true(2*pi*self.l_calc.R),
                                                   "a": self.l_calc.aj,
                                                   "b": self.l_calc.bj,
                                                   "A max new": self.l_calc.amp_max(),
                                                   "I v1": self.l_calc.current_new(),
                                                   "I v2": self.l_calc.current_alt(),
                                                   "I v3": self.l_calc.current_calc(),
                                                   "effective mu": self.l_calc.mu * self.l_calc.amp_max() ** 2
                                    }
                                    found_derivs_array.append(new_element)

        # Преобразовать локальные переменные в объектные, сохраняя данные.
        self.derivs = found_derivs_array

        # Указывать, что этот метод был запущен.
        self.calculated = True

        # И завершать синхронизацию.
        print("Построение сетки завершено: ", time.time() - start_time)

    #Запуск для случайно выбранных точек.
    def mcBVP(self):

        # Старое вычисление уже существует.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # Таблица не задана.
        if not self.is_mc:
            raise Exception("mcBVP требует сначала выбрать точки. Попробуйте вызвать makeMCPoints.") 

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

        # self.val_table

        # результаты
        found_derivs_array = []
        
        # Решить для каждой случайно определенной точки.
        print("Количество точек для вычисления:", num)
        for ii in range(num):
            self.l_calc.update_params(R=self.val_table[ii,3], B=self.val_table[ii,2], dk=self.val_table[ii,1], 
                                      mu=self.val_table[ii,0], k = self.val_table[ii,5], amp=self.val_table[ii,4])
            # Вычислить производную от параметров.

            # Найти нули, которые решают краевую задачу; это та часть, 
            # которая требует времени.
            with warnings.catch_warnings(action="ignore"):
                deriv_found = self.l_calc.find_root_many()
            # Цифры здесь указаны для решения без e-e взаимодействия.
            psiprime0 = self.l_calc.psi_prime_0()
            exactend = self.l_calc.psij0(2*pi*self.l_calc.R)
            i0 = self.l_calc.current_old()
            a0 = self.l_calc.aj0
            b0 = self.l_calc.bj0
            A_max_0 = self.l_calc.amp_max_0()
            # Сохранить все данные в словарном объекте для использования 
            # с pandas.
            for deriv in deriv_found:
                self.l_calc.setDeriv(deriv)
                new_element = {"R": self.val_table[ii,3],
                               "B": self.val_table[ii,2],
                               "dk": self.val_table[ii,1],
                               "mu": self.val_table[ii,0],
                               "k": self.val_table[ii,5],
                               "A": self.val_table[ii,4],
                               "dpsi0": psiprime0,
                               "Exact End": exactend,
                               "a0": a0,
                               "b0": b0,
                               "A max 0": A_max_0,
                               "I0": i0,
                               "dpsi": deriv,
                               "New End": self.l_calc.psij_pred_true(2*pi*self.l_calc.R),
                               "a": self.l_calc.aj,
                               "b": self.l_calc.bj,
                               "A max new": self.l_calc.amp_max(),
                               "I v1": self.l_calc.current_new(),
                               "I v2": self.l_calc.current_alt(),
                               "I v3": self.l_calc.current_calc(),
                               "effective mu": self.l_calc.mu * self.l_calc.amp_max() ** 2
                              }
                found_derivs_array.append(new_element)

        # Преобразовать локальные переменные в объектные, сохраняя данные.
        self.derivs = found_derivs_array

        # Указывать, что этот метод был запущен.
        self.calculated = True

        # И завершать синхронизацию.
        print("Построение таблицы завершено: ", time.time() - start_time)