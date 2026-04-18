# eelib/grid_slow_osc.py
#
# класс grid_slow_osc 
# Автор: Элизабет Гоулд
# Дата последнего изменения: 23.03.2026

"""
Этот код создает сетку решений начальной задачи, чтобы вычислить 
t_slow (= 2*pi /M), период медленных колебаний. Здесь можно варьировать значения
R, B, k, dk, mu, A и psi'(0) и построить либо сетку значений, либо сгенерировать
набор случайных значений. 

Сетка psi'(0) расположена синусоидально. Сетка mu разнесена логарифмически. 
Остальные сетки разнесены линейно. Одним из результатов этого является то, что
mu берется как показатель степени 10 для сетки, в то время как для прогонов по
методу Монте-Карло оно берется как исходное число.
 
grid_slow_osc построен на основе grid_fast_osc, наследуя методы формирования
сетки, поскольку структура идентична.
"""

# Код дла запуски на сетке:
# > gridl = eelib.grid_slow_osc(R, B, dk, mu)
# > gridl.makeGridPoints(mu=mu_r, B=b_r, num = [n])
# > gridl.gridSlowOsc()

# Код дла запуска по методу Монте-Карло:
# > gridl = eelib.grid_slow_osc(R, B, dk, mu)
# > gridl.makeMCPoints(mu=mu_r, B=b_r, num = n)
# > gridl.mcSlowOsc()

# __init__(self, R=1.0, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.) --
#     Создает объект loop с заданными параметрами. Переданные значения 
#     параметров актуальны только для тех параметров, которые не подлежат
#     варьированию, поскольку эти параметры будут заменены во время вычислений.

# __repr__(self)
# __str__(self)
#       Печатают описание объекта. 

# save_solution -- Логическое значение по умолчанию равно False. Если значение
#                  равно True, будут сохранены полные решения начальных задачей.
#                  Переменная используется, поскольку решения могут быть
#                  большими и редко требуются

# clear_calcs(self) -- Удаляет данные. 
# setIntegratorParameters(self, solve_mu_0 = None, n_sm = None, method = None, 
#                          rtol = None, atol = None, R_max = None, n_lg = None)
# Значения параметров по умолчанию следующие: solve_mu_0 = False, n_sm = 20, 
#        method = 'RK45', rtol = rtol, atol = atol, R_max = 1.0, n_lg = 1000.
# n_sm подсчитывает, сколько колебаний |Psi|^2 необходимо подсчитать, чтобы
# оценить t / 2 (что равно pi / k).
# n_lg подсчитывает, сколько точек будет выбрано для измерения psi, когда
# быстрые колебания достигнут своего максимального значения. Это позволит
# рассчитать кривую для медленных колебаний.
# Если solve_mu_0 = True, код также оценит быстрые колебания для случая без
# взаимодействия с ee.
# Не заданные параметры изменены не будут.

# makeGridPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0), 
#                R = (-1.0, -1.0), A = (-1.0, -1.0), k0 = (-1.0, -1.0),
#                ang_lim = 0.1, num = [10])
# makeMCPoints(self, mu = (1.0, 1.0), dk = (-1.0, -1.0), B = (-1.0, -1.0),
#            R = (-1.0, -1.0), A = (-1.0, -1.0), k0 = (-1.0, -1.0), num = 1000)

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

# gridSlowOsc(self) выполняет вычисление, если задана сетка.
# mcSlowOsc(self) выполняет вычисление, если задан случайный набор точек для
#                 запуска.
# runCalc(self) выполняет вычисление в любом случае.

# Результаты сохраняются в следующих переменных:
# self.slow_osc_k   -- Оценка волнового числа для медленных колебаний. 
#                      (Найдено путем подгонки к синусоидальной функции.)
# self.slow_osc_a   -- Оценка амплитуды медленных колебаний.
# self.slow_osc_th  -- Оценка смещения угла медленного колебания по 
#                      синусоидальной функции.
# self.slow_osc_sol -- Сохраняет полное решение. Оно сохраняется в виде списка,
#                      а не массива, как другие решения.
# self.slow_osc_i   -- i-й элемент этого списка соответствует полному решению
#         для индексов массива других оценок, как указано в содержащемся списке.
#         Для i[j] = [n_mu, n_dk, n_b, n_r, n_a, n_k0, n_d, n_d] (где j и все n
#         являются целыми числами) sol[j] является полное решение для расчетных
#         значений k[i], a[i] и th[i].
#         Эта последняя переменная используется только для решения на сетке, 
#         поскольку все переменные имеют один индекс для случая Монте-Карло. 

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
from eelib.consts import kFAu, rtol, atol
from eelib.loop import loop
from eelib.grid_fast_osc import grid_fast_osc
from eelib.fitted_functions import fit_sin
import time
import gc

class grid_slow_osc(grid_fast_osc):

    # ------ ИНИЦИАЛИЗАЦИЯ --------------

    # Здесь установлены параметры для удаления вызова интегратора, что сэкономит некоторое время.
    def __init__(self, R=1.0, B=0.8, dk=0.5, mu=1.e-6, k = kFAu, amp=1.):
        super().__init__(R,B,dk,mu,k,amp)

        self.setIntegratorParameters(solve_mu_0=False, R_max = 1.0, n_lg = 1000)
        self.save_solution = False 
                # Тесты с сохраненным решением показывают, что оно слишком велико.

    # Выходные данные для инструкции print.
    def __repr__(self):
        if self.is_grid:
            if self.calculated:
                str = "Сеточный объект для измерения медленных колебаний:\n"
            else:
                str = "Нерасчетный сеточный объект для измерения медленных колебаний:\n"

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
                str = f"Объект Монте-Карло для измерения медленных колебаний:\n"
            else:
                str = f"Нерасчетный объект Монте-Карло для измерения медленных колебаний:\n"
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
            return "Пустой object объект для измерения медленных колебаний:\nR - %s, B - %s, dk - %s, k - %s, mu - %s, A - %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # Выходные данные для инструкции print.
    def __str__(self):
        if self.is_grid:
            if self.calculated:
                str = "Сеточный объект для измерения медленных колебаний:\n"
            else:
                str = "Нерасчетный сеточный объект для измерения медленных колебаний:\n"

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
                str = f"Объект Монте-Карло для измерения медленных колебаний:\n"
            else:
                str = f"Нерасчетный объект Монте-Карло для измерения медленных колебаний:\n"
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
            return "Пустой объект для измерения медленных колебаний:\nR - %s, B - %s, dk - %s, k - %s, mu - %s, A - %s" % (self.R, self.B, self.dk, self.k, self.mu, self.amp)

    # ------ ИЗМЕНЕННЫЕ МЕТОДЫ -------

    # Это изменено, поскольку данные сохраняются другим способом.
    def clear_calcs(self):
        self.slow_osc_k   = None
        self.slow_osc_a   = None
        self.slow_osc_th  = None
        self.slow_osc_i   = None
        self.slow_osc_sol = None
        gc.collect()

    # R_max не имеет актуальность к быстрым колебаниям, поскольку для этой оценки
    # вычисляются только первые n быстрых колебаний. Таким образом, это работает
    # как в предыдущем случае, но теперь с параметром R_max, позволяющим
    # выполнить перерасчет для получения более точных M оценок.
    def setIntegratorParameters(self, solve_mu_0 = None, n_sm = None, method = None, rtol = None, atol = None, R_max = None, n_lg = None):
        super().setIntegratorParameters(solve_mu_0, n_sm, method, rtol, atol)
        if n_lg is not None:
            self.n_lg = n_lg
        if R_max is not None:
            self.R_max = R_max

    # ------------- ПОИСК ЗНАЧЕНИЙ НА СЕТКЕ ---------------

    # Дескриптор для следующих двух метода для запуска кода.
    def runCalc(self):
        if self.is_grid: 
            self.gridSlowOsc()
        elif self.is_mc: 
            self.mcSlowOsc()
        else: 
            print("Точки для вычисления еще не заданы.")

    # Удалить унаследованный метод.
    def gridFastOsc(self):
        print("Неверный тип объекта. Попробуйте вместо этого gridSlowOsc.")

    # Выполнить вычисление по сетке.
    # Это требует времени, но больше не требует параметров. Из-за этого 
    # требования оно не связано с параметрами.
    def gridSlowOsc(self):

        # Старое вычисление уже существует.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # Сетка не задана.
        if not self.is_grid:
            raise Exception("Сначала для gridSlowOsc требуется сформировать сетку. Попробуйте вызвать makeGridPoints") 

        # Параметры интеграции.
        pr = self.R_max # может быть изменен, если требуется больше циклов
        plot_code = 1   # только построить er, экономить времени

        # Синхронизация
        start_time = time.time()
        print('Начать построение сетки: ', time.time() - start_time)

        # Инициализация сетки
        num_dk = self.num_dk
        num_k0 = self.num_k0
        num_b  = self.num_b
        num_r  = self.num_r
        num_mu = self.num_mu
        num_a  = self.num_a
        num_d  = self.grid_size

        # сохранять решение для анализа
        sol_er_u = None

        # сохраненные данные
        slow_oscillation_wavenumber = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        slow_oscillation_amplitude  = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))
        slow_oscillation_theta      = np.zeros((num_mu, num_dk, num_b, num_r, num_a, num_k0, num_d, num_d))

        slow_osc_sv_ind  = [] # индексы для сопоставления полного решения 
                              # с другими сохраненными данными
        slow_osc_sv_data = [] # полное решение

        # Индексы для каждого параметра, который меняется.
        ib  = 0
        ir  = 0
        im  = 0
        ik  = 0
        ik0 = 0
        ia  = 0
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
                                        # Вычислить производную от параметров,
                                        # если она не соответствует заданным.
                                        self.l_calc.setDeriv(self.d0_grid[idr,idi])
                                        # Оценить k.
                                        self.l_calc.find_fast_oscillations(self.n_sm, method = self.method, 
                                                                           rtol = self.rtol, atol = self.atol)
                                        self.l_calc.solve_ivp(n = self.n_lg, percent_range=pr, method = self.method,
                                                             rtol = self.rtol, atol = self.atol, solve = plot_code)

                                        # Найти решение
                                        sol_er_u = self.l_calc.solu
                                        # подогнать синусоиду к данным
                                        fit = fit_sin(sol_er_u)
                                        if fit is not None:
                                            slow_oscillation_wavenumber[im,ik,ib,ir,ia,ik0,idr,idi] = fit[1]
                                            slow_oscillation_amplitude[im,ik,ib,ir,ia,ik0,idr,idi] = fit[0]
                                            slow_oscillation_theta[im,ik,ib,ir,ia,ik0,idr,idi] = fit[2]

                                        if self.save_solution:
                                        # А также сохранить полное решение.
                                            slow_osc_sv_ind.append([im,ik,ib,ir,ia,ik0,idr,idi])
                                            slow_osc_sv_data.append(sol_er_u.copy())

        # Преобразовать локальные переменные в объектные, сохраняя данные.
        self.slow_osc_k  = slow_oscillation_wavenumber  
                # Оценка волнового числа для медленных колебаний 
                # (найдена путем подгонки sin).
        self.slow_osc_a  = slow_oscillation_amplitude   # максимум модуля
        self.slow_osc_th = slow_oscillation_theta
        if self.save_solution:
            self.slow_osc_i  = slow_osc_sv_ind
                # Индексы для разделителей, 
                # из-за альтернативного механизму хранения
            self.slow_osc_sol  = slow_osc_sv_data
                # Сохранить полное решение.

        # Указывать, что этот метод был запущен.
        self.calculated = True

        # И завершать синхронизацию.
        print("Построение сетки завершено: ", time.time() - start_time)

    # Удалить унаследованный метод.
    def mcFastOsc(self):
        print("Неверный тип объекта. Попробуйте mcSlowOsc.")

    # Запуск для случайно выбранных точек.
    def mcSlowOsc(self):

        # Старое вычисление уже существует.
        if self.calculated:
            self.clear_calcs()
            gc.collect()

        # Таблица не задана.
        if not self.is_mc:
            raise Exception("mcSlowOsc требует сначала выбрать точки. Попробуйте вызвать makeMCPoints.") 

        # Параметры интеграции.
        pr = self.R_max # может быть изменен, если потребуется больше циклов
        plot_code = 1   # только построить er, экономить времени

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

        # Сохранить решение для анализа.
        sol_er_u = None
        sol_er_u_save = []

        slow_oscillation_wavenumber = np.zeros((num))
        slow_oscillation_theta = np.zeros((num))
        slow_oscillation_amplitude = np.zeros((num))

        # Решить для каждой случайно определенной точки.
        print("Количество точек для вычисления:", num)
        for ii in range(num):
            self.l_calc.update_params(R=self.val_table[ii,3], B=self.val_table[ii,2], dk=self.val_table[ii,1], 
                                      mu=self.val_table[ii,0], k = self.val_table[ii,5], amp=self.val_table[ii,4])
            # Вычислить производную от параметров, если она не соответствует заданным.
            self.l_calc.setDeriv(self.val_table[ii,6]+ 1.j *self.val_table[ii,7])
            # Оценить k
            self.l_calc.find_fast_oscillations(self.n_sm, method = self.method, rtol = self.rtol, atol = self.atol)
            self.l_calc.solve_ivp(n = self.n_lg, percent_range=pr, method = self.method,
                                                             rtol = self.rtol, atol = self.atol, solve = plot_code)

            # Найти решение.
            sol_er_u = self.l_calc.solu

            # подогнать синусоиду к данным.
            fit = fit_sin(sol_er_u)
            if fit is not None:
                slow_oscillation_wavenumber[ii] = fit[1]
                slow_oscillation_amplitude[ii] = fit[0]
                slow_oscillation_theta[ii] = fit[2]

            # А также сохранить полное решение.
            if self.save_solution: sol_er_u_save.append(sol_er_u.copy())

        # Преобразовать локальные переменные в объектные, сохраняя данные.
        self.slow_osc_k  = slow_oscillation_wavenumber
                # Оценка волнового числа для медленных колебаний 
                # (найдена путем подгонки sin).
        self.slow_osc_a  = slow_oscillation_amplitude   # максимум модуля.
        self.slow_osc_th = slow_oscillation_theta
        if self.save_solution: self.slow_osc_sol  = sol_er_u_save
                # Сохранить полное решение.


        # Указывать, что этот метод был запущен.
        self.calculated = True

        # И завершать синхронизацию.
        print("Построение таблицы завершено: ", time.time() - start_time)
