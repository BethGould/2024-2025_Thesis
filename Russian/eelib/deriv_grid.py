# eelib/geriv_grid.py
#
# класс deriv_grid
# Автор: Элизабет Гоулд
# Дата последнего изменения: 23.03.2026
#  
# Принимает фиксированные значения R, B, k, dk, mu и amp и создает сетку с 
# синусоидальным интервалом, состоящую из grid_size действительных и мнимых
# значений. loop вызвано grid_size^2 раз. Результаты сохраняются и их графики
# могут быть построены позже с использованием методов того же объекта. Графики
# позволяют построить ожидаемую кривую в соответствии с моделью. Этот код
# предназначен только для визуального анализа результатов начальных задач.
# 
# Пример код построения графика:
#
# * сначала установите R (% R_max), B (% R_max), 
#       k (dk, 0-1 - это один период для R = R_max)
#       mu (необработанный, диапазоны от 10^-10 до 10^-6 кажутся подходящими,
#           предпочтительно от 10^-8 до 10^-7)
#       n_g (количество точек в одном направлении), no (номер набор графиков для
#            обозначения графиков)
# 
# > loopl = eelib.deriv_grid(R, B, k, mu, grid_size=n_g) # создать объект
# > loopl.derivGrid()                                    # запустить объект

# ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ СЕТОК (большое количество сохраненных pdf-файлов).
# > for i in range(n_g):
# >     for j in range(n_g):
# >         loopl.plot_abs(i,j,k,R,B,mu,no)
# >         loopl.plot_real(i,j,k,R,B,mu,no)

# ----- Методы ---- 
# __init__(self, R, B, dk, mu, k = kFAu, amp=1., grid_size = 9, ang_lim = 0.1)
# derivGrid(self, n = 200, pr=1.0, method = 'RK45', rtol = rtol, atol = atol, 
#           trim = 16, to_plot = ['er','0r','0x','em'])
# plot_real(self, i, j, k, R, B, mu, no)
# plot_abs(self, i, j, k, R, B, mu, no)
#
# __repr__(self)
# __str__(self)
# change_parameters(self, R=None, B=None, dk=None, mu=None, k=None, amp = None, 
#                   grid_size = None, ang_lim = None)
# clear_run(self)
#
# find_root_points(self,y_points)
# find_root_start(self,y_points)
# find_root_dif(self,y_points)
# find_amp(self, sol)

# change_parameters очищает все данные и восстанавливает объект, как если бы он
# был создан с заданными параметрами. В этом случае все параметры, которые не
# были указаны, останутся неизменными. clear_run просто очищает данные из
# процесса интеграции derivGrid. Больше ничего не меняется.

# to_plot может принимать следующие коды (при условии, что код не был взломан,
#         что вполне возможно).
# "er" - Решение с восстановленной амплитудой (с удаленной ошибкой) при 
#        электрон-электронном взаимодействии.
# "ed" - решение с убывающей амплитудой (ошибочное) при электрон-электронном
#        взаимодействии.
# "0r" - Решение с восстановленной амплитудой (с удаленной ошибкой) без
#        электрон-электронного взаимодействия.
# "0d" - Решение с убывающей амплитудой (ошибочное) без электрон-электронного
#        взаимодействия.
# "0x" - Точное решение без электрон-электронного взаимодействия. 
# "em" - смоделированное решение с электрон-электронным взаимодействием.
# Нужные коды передаются в виде списка из двух символьных строк. Нераспознанные
# коды, скорее всего, игнорируются.

# derivGrid выполняет интеграцию, и это наиболее трудоемкую часть.
# n и trim уменьшают количество точек на графиках.
# pr - это процент от радиуса, для которого решение нужно быть найдено. Его
#      также можно задать больше, чем 1.0, чтобы получить больше данных.
# method, rtol, atol управляют интегратором.
# to_plot предоставляет варианты интеграции для вычисления, которые сохраняются
# и наносятся на графики.

# plot_real, plot_abs построят и сохранят указанные решение. 
# i, j - это обозначения в сетке нужного решения.
# k, R, B, mu предназначены для обозначения графика.
# no - это индекс, используемый для обозначения графиков, позволяет создавать
#      уникальные обозначения.

# Обработка ошибок не была добавлена в этот класс, поэтому при неправильном
# использовании он может работать некорректно.

# --- БИБЛИОТЕКИ ---
import numpy as np
import matplotlib.pyplot as plt
from eelib.consts import pi, kFAu, rtol, atol, R_max, B_max
from eelib.loop import loop
from eelib.fitted_functions import fit_sin
import time

class deriv_grid:

    # Инициализировать сетку, создав объект loop и выбрав точки производной
    # на сетке.
    def __init__(self, R, B, dk, mu, k = kFAu, amp=1., grid_size = 9, ang_lim = 0.1):
        self.l_calc = loop(R, B, dk, mu, k, amp)  # класс loop для вычислений
        self.grid_size = grid_size
        self.dlim = self.l_calc.psi_prime_0_max() # max(abs(psi'(0)))

        self.ang_lim = ang_lim
        self.R = R
        self.B = B
        self.dk = dk
        self.mu = mu
        self.k = k
        self.amp = amp
        
        # формирование сетки производных для t = 0
        # ang_limit помогает избежать проблем, в то время как максимальный
        # модуль производных определяется их максимальным модулем точного
        # решения. 
        x = np.linspace(pi-ang_lim, ang_lim, num=self.grid_size, dtype = complex)
        x = np.cos(x) * self.dlim 
        y = np.linspace(pi-ang_lim, ang_lim, num=self.grid_size, dtype = complex)
        y = np.cos(y) * self.dlim
        xx, yy = np.meshgrid(x, y)
        d0_grid = xx + 1j * yy
        self.d0_grid = d0_grid

        self.run = False

    # Выходные данные для инструкции print.
    def __repr__(self):
        if self.run:
            str = "Объект с сеткой решений ОДУ для различных производных:\n"
        else:
            str = "Объект для построения сетки решений ОДУ для различных производных:\n"

        # И показать параметры.
        str = str + f"mu: {self.mu}\n"
        str = str + f"dk: {self.dk}\n"
        str = str + f"B:  {self.B}\n"
        str = str + f"R:  {self.R}\n"
        str = str + f"A:  {self.amp}\n"
        str = str + f"k0: {self.k}\n"
        str = str + f"Размер сетки (в каждом направлении - действительном и мнимом): {self.grid_size}"
        return str

    def __str__(self):
        if self.run:
            str = "Объект с сеткой решений ОДУ для различных производных:\n"
        else:
            str = "Объект для построения сетки решений ОДУ для различных производных:\n"
            
        # И показать параметры.
        str = str + f"mu: {self.mu}\n"
        str = str + f"dk: {self.dk}\n"
        str = str + f"B:  {self.B}\n"
        str = str + f"R:  {self.R}\n"
        str = str + f"A:  {self.amp}\n"
        str = str + f"k0: {self.k}\n"
        str = str + f"Размер сетки (в каждом направлении - действительном и мнимом): {self.grid_size}"
        return str

    # Код для сброса объекта сетки и изменения параметров.
    def change_parameters(self, R=None, B=None, dk=None, mu=None, k=None, amp = None, grid_size = None, ang_lim = None):
        # Очистить данные для старых параметров.
        self.clear_run()

        # Изменить параметры, заданные в коде.
        if R is not None: 
            self.R = R
        if B is not None:
            self.B = B
        if dk is not None:
            self.dk = dk
        if mu is not None: 
            self.mu = mu
        if amp is not None: 
            self.amp = amp
        if k is not None:
            self.k = k
        if grid_size is not None:
            self.grid_size = grid_size
        if ang_lim is not None:
            self.ang_lim = ang_lim

        # Применить это изменение к параметрам:
        self.l_calc.update_params(self.R, self.B, self.dk, self.mu, self.k, self.amp)
        self.dlim = self.l_calc.psi_prime_0_max()
        self.d0_grid = None

        # формирование сетки производных для t = 0
        # ang_limit помогает избежать проблем, в то время как максимальный
        # модуль производных определяется их максимальным модулем точного
        # решения. 
        x = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        x = np.cos(x) * self.dlim 
        y = np.linspace(pi-self.ang_lim, self.ang_lim, num=self.grid_size, dtype = complex)
        y = np.cos(y) * self.dlim
        xx, yy = np.meshgrid(x, y)
        d0_grid = xx + 1j * yy
        self.d0_grid = d0_grid

    # Удалить все данные. 
    def clear_run(self):
        self.run = False

        self.s_grid_er_u = None
        self.s_grid_ed_u = None
        self.s_grid_em_u = None
        self.s_grid_M_pred = None
        self.s_grid_0d_u = None
        self.s_grid_0r_u = None
        self.s_grid_0x_u_tc = None
        self.s_grid_0x_l_tc = None
        #self.s_grid_0x_u_tcm = None
        #self.s_grid_0x_l_tcm = None
        self.s_grid_0x_u_ta = None
        self.s_grid_0x_l_ta = None
        self.s_grid_0x_u_tr = None
        self.s_grid_0x_l_tr = None
        self.s_grid_0x_u_yc = None
        self.s_grid_0x_l_yc = None
        self.s_grid_0x_u_ycm = None
        self.s_grid_0x_l_ycm = None
        self.s_grid_0x_u_ya = None
        self.s_grid_0x_l_ya = None
        self.s_grid_0x_u_yr = None
        self.s_grid_0x_l_yr = None
    
    # Это просто создает сетку значений для различных значений psi'_0. 
    # Из-за многократного запуска интегратора это может быть довольно медленным 
    # процессом. Поэтому программа использует библиотеку time для регулярного
    # расчета времени запуска, что позволяет отслеживать, сколько времени осталось. 
    # n - количество точек для построения графика, используется для не вызванных, 
    #     предопределенных точек
    # pr -- percent_range 
    # method, rtol (относительный допуск), atol (абсолютный допуск)
    # trim -- должно быть четным, каждая точка x используется для построения
    #         графиков, используется для вызванных точек
    # to_plot -- список строк с желательными вариантами для построения графика
    # n_start -- количество колебаний, используемое при усреднении для оценки k
    #            для быстрых колебаний
    def derivGrid(self, n = 200, pr=1.0, method = 'RK45', rtol = rtol, atol = atol, trim = 16, 
                  to_plot = ['er', '0r', '0x', 'em'], n_start=20):

        if self.run:
            print("Этот объект уже вычислил сетку.")
            return None

        # сохранить входные параметры
        self.trim = trim
        self.to_plot = to_plot
 
        # начинать отсчет времени
        start_time = time.time()

        # Преобразуем понятную систему кода построения графиков класса 
        # deriv_grid в сложную для понимания систему класса loop.
        plot_code = -1
        if 'er' in to_plot: plot_code *= -1
        if 'ed' in to_plot: plot_code *= 2
        if '0d' in to_plot: plot_code *= 3
        if '0r' in to_plot: plot_code *= 5
        if 'em' in to_plot: plot_code *= 7

        # Места для найденных решений (которые будут списками списков решений).
        '''
        u, l -- верхний, нижний
        e -- с e-e взаимодействием
        0 -- без e-e взаимодействия
        d -- полный вычисление
        r -- частичное восстановление мощности
        m -- частичное восстановление мощности, вызванное расчетным периодом
             быстрых колебаний
        x -- точное
        tc, ta, tr -- точки временя (положения), выбранные вычислением (tc), 
                      вызыванием |psi|' = 0 (ta), вызыванием Re(psi) = 0 (tr), 
                      чтобы следовать огибающей
        yc, ya, yr -- значение функции в точках tc (yc), ta (ya), tr (yr)
        '''

        if 'er' in to_plot: s_grid_er_u = []
        if 'er' in to_plot: s_grid_er_l = []
        if 'ed' in to_plot: s_grid_ed_u = []
        if 'ed' in to_plot: s_grid_ed_l = []
        if 'em' in to_plot: s_grid_em_u = []
        if 'em' in to_plot: s_grid_em_l = []

        if 'er' in to_plot: s_grid_M_pred = []

        if '0d' in to_plot: s_grid_0d_u = []
        if '0d' in to_plot: s_grid_0d_l = []
        if '0r' in to_plot: s_grid_0r_u = []
        if '0r' in to_plot: s_grid_0r_l = []

        if '0x' in to_plot: s_grid_0x_u_tc = []
        if '0x' in to_plot: s_grid_0x_l_tc = []
        #if '0x' in to_plot: s_grid_0x_u_tcm = []
        #if '0x' in to_plot: s_grid_0x_l_tcm = []
        if '0r' in to_plot:
            if '0x' in to_plot: s_grid_0x_u_ta = []
            if '0x' in to_plot: s_grid_0x_l_ta = []
            if '0x' in to_plot: s_grid_0x_u_tr = []
            if '0x' in to_plot: s_grid_0x_l_tr = []
        if '0x' in to_plot: s_grid_0x_u_yc = []
        if '0x' in to_plot: s_grid_0x_l_yc = []
        #if '0x' in to_plot: s_grid_0x_u_ycm = []
        #if '0x' in to_plot: s_grid_0x_l_ycm = []
        if '0r' in to_plot:
            if '0x' in to_plot: s_grid_0x_u_ya = []
            if '0x' in to_plot: s_grid_0x_l_ya = []
            if '0x' in to_plot: s_grid_0x_u_yr = []
            if '0x' in to_plot: s_grid_0x_l_yr = []

        # Начинаем трудоемкую часть кода
        print('Начать построение сетки: ', time.time() - start_time)
        print('Код графика:', to_plot, plot_code)
        
        # Вычислить для каждой точки сетки
        for i in range(self.grid_size):
            # сетка хранится в виде двумерного списка. Этот код создает второе 
            # измерение.
            if 'er' in to_plot: s_grid_er_u.append([])
            if 'er' in to_plot: s_grid_er_l.append([])
            if 'ed' in to_plot: s_grid_ed_u.append([])
            if 'ed' in to_plot: s_grid_ed_l.append([])
            if 'em' in to_plot: s_grid_em_u.append([])
            if 'em' in to_plot: s_grid_em_l.append([])

            if 'er' in to_plot: s_grid_M_pred.append([])

            if '0d' in to_plot: s_grid_0d_u.append([])
            if '0d' in to_plot: s_grid_0d_l.append([])
            if '0r' in to_plot: s_grid_0r_u.append([])
            if '0r' in to_plot: s_grid_0r_l.append([])

            if '0x' in to_plot: 
                s_grid_0x_u_tc.append([])
                s_grid_0x_l_tc.append([])
                #s_grid_0x_u_tcm.append([])
                #s_grid_0x_l_tcm.append([])
                if '0r' in to_plot:
                    s_grid_0x_u_ta.append([])
                    s_grid_0x_l_ta.append([])
                    s_grid_0x_u_tr.append([])
                    s_grid_0x_l_tr.append([])
                s_grid_0x_u_yc.append([])
                s_grid_0x_l_yc.append([])
                #s_grid_0x_u_ycm.append([])
                #s_grid_0x_l_ycm.append([])
                if '0r' in to_plot:
                    s_grid_0x_u_ya.append([])
                    s_grid_0x_l_ya.append([])
                    s_grid_0x_u_yr.append([])
                    s_grid_0x_l_yr.append([])
            
            # Вывести текущую стадию вычисления и время, затраченное на данный
            # момент. Переключить подсчет с 0 на 1, чтобы сделать его более
            # интуитивно понятным. 
            print("i = ", i + 1, "из ", self.grid_size, "время: ", time.time() - start_time)

            for j in range(self.grid_size):
                # Это единственное место, где код вызывает интегратор, который
                # работает медленно. Обратите внимание, что find_fast_oscillations
                # был перенесен сюда из setDeriv, поскольку это замедляло работу
                # алгоритма соединения краевых задач. 

                self.l_calc.setDeriv(self.d0_grid[j,i]) 
                self.l_calc.find_fast_oscillations(n=n_start, method = method,
                                                   rtol = rtol, atol = atol)
                self.l_calc.solve_ivp(n = n, percent_range = pr, method = method,
                                      rtol = rtol, atol = atol, solve=plot_code)

                # Вывести текущую стадию расчета и время, затраченное на данный
                # момент. Переключить подсчет с 0 на 1, чтобы сделать его более
                # интуитивно понятным. 
                print("Готово, i, j = ", i+1,j+1, "время: ", time.time() - start_time)

                # Добавлять все решения в сохраненные списки решений, выбранные
                # на основе plot_code.
                if 'er' in to_plot: s_grid_er_u[i].append(self.l_calc.solu)
                if 'ed' in to_plot: s_grid_ed_u[i].append(self.l_calc.solu_d)
                if 'em' in to_plot: s_grid_em_u[i].append(self.l_calc.solu_m)

                if '0d' in to_plot: s_grid_0d_u[i].append(self.l_calc.solu0)
                if '0r' in to_plot: s_grid_0r_u[i].append(self.l_calc.solu0_r)

                if '0x' in to_plot:
                    T_arr = self.l_calc.find_period_shift_exact()  
                            # быстрое, медленное, положительное, отрицательное

                    # Нужно время начала для случаев, которые соответствуют 
                    # реальной амплитуде. Они взяты из 
                    # s_grid_0x_u_tc[np.argmax(abs(s_grid_0x_u_yc))], но затем 
                    # нужно рассчитать в обратном порядке.
                    xT = self.l_calc.find_real_env_start()

                    s_grid_0x_u_tc[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stu_ex, T_arr[0]))
                    s_grid_0x_l_tc[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, self.l_calc.stl_ex, T_arr[0]))
                    #s_grid_0x_u_tcm[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, xT, T_arr[2]))
                    #s_grid_0x_l_tcm[i].append(self.l_calc.find_t_points(n, self.l_calc.lngt, xT, T_arr[3]))

                    if '0r' in to_plot:
                        # вызванные времени
                        s_grid_0x_u_ta[i].append(self.l_calc.solu0_r['t_events'][0][0::trim])
                        s_grid_0x_l_ta[i].append(self.l_calc.solu0_r['t_events'][0][1::trim])
                        s_grid_0x_u_tr[i].append(self.l_calc.solu0_r['t_events'][1][0::trim])
                        s_grid_0x_l_tr[i].append(self.l_calc.solu0_r['t_events'][1][1::trim])

                    # значения y
                    s_grid_0x_u_yc[i].append(self.l_calc.psij(s_grid_0x_u_tc[i][j]))
                    s_grid_0x_l_yc[i].append(self.l_calc.psij(s_grid_0x_l_tc[i][j]))
                    #s_grid_0x_u_ycm[i].append(self.l_calc.psij(s_grid_0x_u_tcm[i][j]))
                    #s_grid_0x_l_ycm[i].append(self.l_calc.psij(s_grid_0x_l_tcm[i][j]))
                    
                    if '0r' in to_plot:
                        s_grid_0x_u_ya[i].append(self.l_calc.psij(s_grid_0x_u_ta[i][j]))
                        s_grid_0x_l_ya[i].append(self.l_calc.psij(s_grid_0x_l_ta[i][j]))
                        s_grid_0x_u_yr[i].append(self.l_calc.psij(s_grid_0x_u_tr[i][j]))
                        s_grid_0x_l_yr[i].append(self.l_calc.psij(s_grid_0x_l_tr[i][j]))
                    
                if 'er' in to_plot: s_grid_M_pred[i].append(self.l_calc.T_slow_mod)

        # И теперь нужно сохранить решения (потому что я по какой-то причине
        # решила не сохранять их ранее).
        if 'er' in to_plot: self.s_grid_er_u = s_grid_er_u
        if 'ed' in to_plot: self.s_grid_ed_u = s_grid_ed_u
        if 'em' in to_plot: self.s_grid_em_u = s_grid_em_u

        if 'er' in to_plot: self.s_grid_M_pred = s_grid_M_pred

        if '0d' in to_plot: self.s_grid_0d_u = s_grid_0d_u
        if '0r' in to_plot: self.s_grid_0r_u = s_grid_0r_u

        if '0x' in to_plot: self.s_grid_0x_u_tc = s_grid_0x_u_tc
        if '0x' in to_plot: self.s_grid_0x_l_tc = s_grid_0x_l_tc
        #if '0x' in to_plot: self.s_grid_0x_u_tcm = s_grid_0x_u_tcm
        #if '0x' in to_plot: self.s_grid_0x_l_tcm = s_grid_0x_l_tcm
        if '0r' in to_plot:
            if '0x' in to_plot: self.s_grid_0x_u_ta = s_grid_0x_u_ta
            if '0x' in to_plot: self.s_grid_0x_l_ta = s_grid_0x_l_ta
            if '0x' in to_plot: self.s_grid_0x_u_tr = s_grid_0x_u_tr
            if '0x' in to_plot: self.s_grid_0x_l_tr = s_grid_0x_l_tr
        if '0x' in to_plot: self.s_grid_0x_u_yc = s_grid_0x_u_yc
        if '0x' in to_plot: self.s_grid_0x_l_yc = s_grid_0x_l_yc
        #if '0x' in to_plot: self.s_grid_0x_u_ycm = s_grid_0x_u_ycm
        #if '0x' in to_plot: self.s_grid_0x_l_ycm = s_grid_0x_l_ycm
        if '0r' in to_plot:
            if '0x' in to_plot: self.s_grid_0x_u_ya = s_grid_0x_u_ya
            if '0x' in to_plot: self.s_grid_0x_l_ya = s_grid_0x_l_ya
            if '0x' in to_plot: self.s_grid_0x_u_yr = s_grid_0x_u_yr
            if '0x' in to_plot: self.s_grid_0x_l_yr = s_grid_0x_l_yr

        # И теперь вывести полный тайминг.
        print("Готово построение сетки: ", time.time() - start_time)
        self.run = True


# ---------

    # построение графиков сетки
    def plot_real(self, i, j, k, R, B, mu, no):

        # Потому что я все еще не хочу писать self.* постоянно.
        to_plot = self.to_plot

        # Приведенные ниже параметры используются для обозначения графика и 
        # должны быть короче фактических параметров
        #k = self.l_calc.dk
        #R = self.l_calc.R
        #B = self.l_calc.B
        #mu = self.l_calc.mu

        # Это вычисляет модель кривой для медленных колебаний. Это
        # синусоидальная кривая, которая моделирует выборку в одной и той же
        # части каждого периода быстрых колебаний. Модель быстрых колебаний
        # представлена кривой "em", которая интегрирует решение, обеспечивая
        # выходные данные в каждый момент времени, заданный моделируемым
        # значением k, что позволяет наблюдать разницу между фактическими и
        # смоделированными значениями k. Чтобы получить хорошее угловое смещение
        # и амплитуду, я подгоняю свои данные к синусоидальной кривой, но
        # отбрасываю данные о волновом числе, так как хочу посмотреть, насколько
        # это отличается от ожидаемого значения.
        if 'er' in to_plot: 
            M_pred = self.s_grid_M_pred[i][j]
            #A_pred = find_amp(self.s_grid_er_u[i][j])
            sol_t = np.real(self.s_grid_er_u[i][j]['t']) 
            #sol_y = np.real(self.s_grid_er_u[i][j]['y'][0])
            fit_func = fit_sin(self.s_grid_er_u[i][j])
            #ind_st = find_root_start(sol_y)
            #t0_pred = sol_t[ind_st]
            theta = fit_func[2]
            A_pred = fit_func[0]
            t_pred = sol_t
            y_pred = A_pred * np.sin(2*pi / M_pred * (t_pred)+ theta)


        # массивы положений
        # оцененные, вызванные |psi|, вызванные Re(psi) -- c, a, r
        if 'er' in to_plot: tuerr = np.real(self.s_grid_er_u[i][j]['t_events'][1][0::self.trim])
        if '0r' in to_plot: tu0rr = np.real(self.s_grid_0r_u[i][j]['t_events'][1][0::self.trim])
        if 'ed' in to_plot: tuedr = np.real(self.s_grid_ed_u[i][j]['t_events'][1][0::self.trim])
        if '0d' in to_plot: tu0dr = np.real(self.s_grid_0d_u[i][j]['t_events'][1][0::self.trim])
        if 'er' in to_plot: tlerr = np.real(self.s_grid_er_u[i][j]['t_events'][1][1::self.trim])
        if '0r' in to_plot: tl0rr = np.real(self.s_grid_0r_u[i][j]['t_events'][1][1::self.trim])
        if 'ed' in to_plot: tledr = np.real(self.s_grid_ed_u[i][j]['t_events'][1][1::self.trim])
        if '0d' in to_plot: tl0dr = np.real(self.s_grid_0d_u[i][j]['t_events'][1][1::self.trim])

        # вызваны |psi|
        if 'er' in to_plot: tuera = np.real(self.s_grid_er_u[i][j]['t_events'][0][0::self.trim])
        if '0r' in to_plot: tu0ra = np.real(self.s_grid_0r_u[i][j]['t_events'][0][0::self.trim])
        if 'ed' in to_plot: tueda = np.real(self.s_grid_ed_u[i][j]['t_events'][0][0::self.trim])
        if '0d' in to_plot: tu0da = np.real(self.s_grid_0d_u[i][j]['t_events'][0][0::self.trim])
        if 'er' in to_plot: tlera = np.real(self.s_grid_er_u[i][j]['t_events'][0][1::self.trim])
        if '0r' in to_plot: tl0ra = np.real(self.s_grid_0r_u[i][j]['t_events'][0][1::self.trim])
        if 'ed' in to_plot: tleda = np.real(self.s_grid_ed_u[i][j]['t_events'][0][1::self.trim])
        if '0d' in to_plot: tl0da = np.real(self.s_grid_0d_u[i][j]['t_events'][0][1::self.trim])
        
        # массивы значений
        if 'er' in to_plot: suerr = np.real(self.s_grid_er_u[i][j]['y_events'][1][0::self.trim, 0])
        if 'ed' in to_plot: suedr = np.real(self.s_grid_ed_u[i][j]['y_events'][1][0::self.trim, 0])
        if 'er' in to_plot: slerr = np.real(self.s_grid_er_u[i][j]['y_events'][1][1::self.trim, 0])
        if 'ed' in to_plot: sledr = np.real(self.s_grid_ed_u[i][j]['y_events'][1][1::self.trim, 0])
        if '0r' in to_plot: su0rr = np.real(self.s_grid_0r_u[i][j]['y_events'][1][0::self.trim, 0])
        if '0d' in to_plot: su0dr = np.real(self.s_grid_0d_u[i][j]['y_events'][1][0::self.trim, 0])
        if '0r' in to_plot: sl0rr = np.real(self.s_grid_0r_u[i][j]['y_events'][1][1::self.trim, 0])
        if '0d' in to_plot: sl0dr = np.real(self.s_grid_0d_u[i][j]['y_events'][1][1::self.trim, 0])

        # вызваны |psi|
        if 'er' in to_plot: suera = np.real(self.s_grid_er_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'ed' in to_plot: sueda = np.real(self.s_grid_ed_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'er' in to_plot: slera = np.real(self.s_grid_er_u[i][j]['y_events'][0][1::self.trim, 0])
        if 'ed' in to_plot: sleda = np.real(self.s_grid_ed_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0r' in to_plot: su0ra = np.real(self.s_grid_0r_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0d' in to_plot: su0da = np.real(self.s_grid_0d_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0r' in to_plot: sl0ra = np.real(self.s_grid_0r_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0d' in to_plot: sl0da = np.real(self.s_grid_0d_u[i][j]['y_events'][0][1::self.trim, 0])

        # точные решения
        if '0x' in to_plot: 
            tu0xc = np.real(self.s_grid_0x_u_tc[i][j])
            tl0xc = np.real(self.s_grid_0x_l_tc[i][j])
            su0xc = np.real(self.s_grid_0x_u_yc[i][j])
            sl0xc = np.real(self.s_grid_0x_l_yc[i][j])
            if '0r' in to_plot:
                tu0xr = np.real(self.s_grid_0x_u_tr[i][j])
                tl0xr = np.real(self.s_grid_0x_l_tr[i][j])
                su0xr = np.real(self.s_grid_0x_u_yr[i][j])
                sl0xr = np.real(self.s_grid_0x_l_yr[i][j])
        
        #-- ПОСТРОЙКА ГРАФИКОВ --
        # Отображает значения точек, вызванные достижением максимумов модуля,
        # таким образом, отображая огибающую psi.
        fig, ax = plt.subplots()
        ax.set_ylabel('Действительная часть \u03C8')
        ax.set_xlabel('x (м)')
        plt.title(f"Огибающая \u03C8, вызванная |\u03C8|, dk={k}, R={R}, B={B}, \u03BC={mu}")

        # Я формирую свою метку на основе списка, который прилагается, так что
        # будут отображены только те графики, которые существуют.
        h_list = []

        if 'er' in to_plot: 
            if np.max(suera) > np.max(slera):
                line1, = ax.plot(tuera, suera, color = 'red', label = 'с e-e взаимодействием')
            else:
                line1, = ax.plot(tlera, slera, color = 'red', label = 'с e-e взаимодействием')
            line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'предсказанное, с e-e взаимодействие')
            h_list.append(line1)
            h_list.append(line13)
        if 'ed' in to_plot: 
            line3, = ax.plot(tueda, sueda, color = 'orange', label = 'с e-e взаимодействием')
            line4, = ax.plot(tleda, sleda, color = 'orange', label = 'с e-e взаимодействием')
            h_list.append(line3)
        if '0r' in to_plot: 
            if np.max(su0ra) > np.max(sl0ra):
                line5, = ax.plot(tu0ra, su0ra, color = 'green', label = 'без e-e взаимодействия')
            else:
                line5, = ax.plot(tl0ra, sl0ra, color = 'green', label = 'без e-e взаимодействия')
            h_list.append(line5)
        if '0d' in to_plot: 
            line7, = ax.plot(tu0da, su0da, color = 'blue', label = 'без e-e взаимодействия')
            line8, = ax.plot(tl0da, sl0da, color = 'blue', label = 'без e-e взаимодействия')
            h_list.append(line7)
        if '0x' in to_plot: 
            if np.max(su0xc) > np.max(sl0xc):
                line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'точное решение')
            else:
                line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'точное решение')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_real_a.pdf", format='pdf')
        plt.close()

        # Эта кривая определяется путем построения графиков случаев, когда 
        # производная от действительной части равна нулю.
        fig, ax = plt.subplots()
        ax.set_ylabel('Действительная часть \u03C8')
        ax.set_xlabel('x (м)')
        plt.title(f"Огибающая \u03C8, вызванная Re(\u03C8), dk={k}, R={R}, B={B}, \u03BC={mu}")

        h_list = []

        if 'er' in to_plot: 
            line1, = ax.plot(tuerr, suerr, color = 'red', label = 'с e-e взаимодействием')
            line2, = ax.plot(tlerr, slerr, color = 'red', label = 'с e-e взаимодействием')
            line13, = ax.plot(t_pred, y_pred, color = 'goldenrod', label = 'предсказанное, с e-e взаимодействие')
            h_list.append(line1)
            h_list.append(line13)
        if 'ed' in to_plot: 
            line3, = ax.plot(tuedr, suedr, color = 'orange', label = 'с e-e взаимодействием')
            line4, = ax.plot(tledr, sledr, color = 'orange', label = 'с e-e взаимодействием')
            h_list.append(line3)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0rr, su0rr, color = 'green', label = 'без e-e взаимодействия')
            line6, = ax.plot(tl0rr, sl0rr, color = 'green', label = 'без e-e взаимодействия')
            h_list.append(line5)
        if '0d' in to_plot: 
            line7, = ax.plot(tu0dr, su0dr, color = 'blue', label = 'без e-e взаимодействия')
            line8, = ax.plot(tl0dr, sl0dr, color = 'blue', label = 'без e-e взаимодействия')
            h_list.append(line7)
        if '0x' in to_plot: 
            if np.max(su0xc) > np.max(sl0xc):
                line9, = ax.plot(tu0xc, su0xc, color = 'purple', label = 'точное решение')
            else:
                line9, = ax.plot(tl0xc, sl0xc, color = 'purple', label = 'точное решение')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_real_r.pdf", format='pdf')
        plt.close()

        # Это также вызванное максимумами действительных значений и также
        # использует эти вычисленные точки в качестве точек, в которых
        # вычисляется точное решение. Это гарантирует, что точное решение также
        # будет иметь форму огибающей, а не синусоиды.
        fig, ax = plt.subplots()
        ax.set_ylabel('Действительная часть \u03C8')
        ax.set_xlabel('x (м)')
        plt.title(f"Огибающая \u03C8, вызванная Re(\u03C8), dk={k}, R={R}, B={B}, \u03BC={mu}")

        h_list = []

        if 'er' in to_plot: 
            line1, = ax.plot(tuerr, suerr, color = 'red', label = 'с e-e взаимодействием')
            line2, = ax.plot(tlerr, slerr, color = 'red', label = 'с e-e взаимодействием')
            h_list.append(line1)
        if 'ed' in to_plot: 
            line3, = ax.plot(tuedr, suedr, color = 'orange', label = 'с e-e взаимодействием')
            line4, = ax.plot(tledr, sledr, color = 'orange', label = 'с e-e взаимодействием')
            h_list.append(line3)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0rr, su0rr, color = 'green', label = 'без e-e взаимодействия')
            line6, = ax.plot(tl0rr, sl0rr, color = 'green', label = 'без e-e взаимодействия')
            h_list.append(line5)
        if '0d' in to_plot: 
            line7, = ax.plot(tu0dr, su0dr, color = 'blue', label = 'без e-e взаимодействия')
            line8, = ax.plot(tl0dr, sl0dr, color = 'blue', label = 'без e-e взаимодействия')
            h_list.append(line7)
        if '0x' in to_plot: 
            if '0r' in to_plot:
                line9, = ax.plot(tu0xr, su0xr, color = 'purple', label = 'точное решение')
                line10, = ax.plot(tl0xr, sl0xr, color = 'purple', label = 'точное решение')
                h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_real_rx.pdf", format='pdf')
        plt.close()


    # --- ГРАФИКИ МОДУЛЯ ---
    def plot_abs(self, i, j, k, R, B, mu, no):

        to_plot = self.to_plot

        # Приведенные ниже параметры используются для обозначения графика и
        # должны быть короче фактических параметров
        #k = self.l_calc.dk
        #R = self.l_calc.R
        #B = self.l_calc.B
        #mu = self.l_calc.mu

        # массивы положений
        # оцененные, вызванные |psi|, вызванные Re(psi) -- c, a, r
        # Все точки, указанные ниже, вызваны максимумами модуля.
        # Точки на графике должны выглядеть, как горизонтальные линии. 
        if 'er' in to_plot: tuera = np.abs(self.s_grid_er_u[i][j]['t_events'][0][0::self.trim])
        if '0r' in to_plot: tu0ra = np.abs(self.s_grid_0r_u[i][j]['t_events'][0][0::self.trim])
        if 'ed' in to_plot: tueda = np.abs(self.s_grid_ed_u[i][j]['t_events'][0][0::self.trim])
        if '0d' in to_plot: tu0da = np.abs(self.s_grid_0d_u[i][j]['t_events'][0][0::self.trim])
        if 'er' in to_plot: tlera = np.abs(self.s_grid_er_u[i][j]['t_events'][0][1::self.trim])
        if '0r' in to_plot: tl0ra = np.abs(self.s_grid_0r_u[i][j]['t_events'][0][1::self.trim])
        if 'ed' in to_plot: tleda = np.abs(self.s_grid_ed_u[i][j]['t_events'][0][1::self.trim])
        if '0d' in to_plot: tl0da = np.abs(self.s_grid_0d_u[i][j]['t_events'][0][1::self.trim])

        # Попробовать также построить графики вычисляемых версий. 
        # Точность расчета видна по убыванию того, что должно быть прямой линией.
        if 'er' in to_plot: tuerc = np.abs(self.s_grid_er_u[i][j]['t'])
        if '0r' in to_plot: tu0rc = np.abs(self.s_grid_0r_u[i][j]['t'])
        if 'em' in to_plot: tuemc = np.abs(self.s_grid_em_u[i][j]['t'])

        # массивы значений
        if 'er' in to_plot: suera = np.abs(self.s_grid_er_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'ed' in to_plot: sueda = np.abs(self.s_grid_ed_u[i][j]['y_events'][0][0::self.trim, 0])
        if 'er' in to_plot: slera = np.abs(self.s_grid_er_u[i][j]['y_events'][0][1::self.trim, 0])
        if 'ed' in to_plot: sleda = np.abs(self.s_grid_ed_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0r' in to_plot: su0ra = np.abs(self.s_grid_0r_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0d' in to_plot: su0da = np.abs(self.s_grid_0d_u[i][j]['y_events'][0][0::self.trim, 0])
        if '0r' in to_plot: sl0ra = np.abs(self.s_grid_0r_u[i][j]['y_events'][0][1::self.trim, 0])
        if '0d' in to_plot: sl0da = np.abs(self.s_grid_0d_u[i][j]['y_events'][0][1::self.trim, 0])

        # Попробовать также построить графики вычисляемых версий.
        if 'er' in to_plot: suerc = np.abs(self.s_grid_er_u[i][j]['y'][0])
        if '0r' in to_plot: su0rc = np.abs(self.s_grid_0r_u[i][j]['y'][0])
        if 'em' in to_plot: suemc = np.abs(self.s_grid_em_u[i][j]['y'][0])

        # точные решения
        if '0x' in to_plot: 
            tu0xa = np.abs(self.s_grid_0x_u_tc[i][j])
            tl0xa = np.abs(self.s_grid_0x_l_tc[i][j])
            su0xa = np.abs(self.s_grid_0x_u_yc[i][j])
            sl0xa = np.abs(self.s_grid_0x_l_yc[i][j])


        # ---- ПОСТРОЙКА ГРАФИКОВ ----
        # --- вычислено ---
        fig, ax = plt.subplots()
        ax.set_ylabel('Модуль \u03C8')
        ax.set_xlabel('x (м)')
        plt.title(f"Огибающая \u03C8, вычисленная, dk={k}, R={R}, B={B}, \u03BC={mu}")
        
        h_list = []

        if 'er' in to_plot: 
            line1, = ax.plot(tuerc, suerc, color = 'red', label = 'с e-e взаимодействием')
            h_list.append(line1)
        if 'em' in to_plot: 
            line2, = ax.plot(tuemc, suemc, color = 'goldenrod', label = 'с e-e взаимодействием')
            h_list.append(line2)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0rc, su0rc, color = 'green', label = 'без e-e взаимодействия')
            h_list.append(line5)
        if '0x' in to_plot: 
            line9, = ax.plot(tu0xa, su0xa, color = 'purple', label = 'точное решение')
            line10, = ax.plot(tl0xa, sl0xa, color = 'purple', label = 'точное решение')
            h_list.append(line9)

        ax.legend(handles=h_list)
        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_abs_c.pdf", format='pdf')
        plt.close()
        
        # --- огибающая, вызвана |psi| ---
        fig, ax = plt.subplots()
        ax.set_ylabel('Модуль \u03C8')
        ax.set_xlabel('x (м)')
        plt.title(f"Огибающая \u03C8, вызванная, dk={k}, R={R}, B={B}, \u03BC={mu}")

        h_list = []

        if 'er' in to_plot: 
            line1, = ax.plot(tuera, suera, color = 'red', label = 'с e-e взаимодействием')
            line2, = ax.plot(tlera, slera, color = 'red', label = 'с e-e взаимодействием')
            h_list.append(line1)
        if 'ed' in to_plot: 
            line3, = ax.plot(tueda, sueda, color = 'orange', label = 'с e-e взаимодействием')
            line4, = ax.plot(tleda, sleda, color = 'orange', label = 'с e-e взаимодействием')
            h_list.append(line3)
        if '0r' in to_plot: 
            line5, = ax.plot(tu0ra, su0ra, color = 'green', label = 'без e-e взаимодействия')
            line6, = ax.plot(tl0ra, sl0ra, color = 'green', label = 'без e-e взаимодействия')
            h_list.append(line5)
        if '0d' in to_plot: 
            line7, = ax.plot(tu0da, su0da, color = 'blue', label = 'без e-e взаимодействия')
            line8, = ax.plot(tl0da, sl0da, color = 'blue', label = 'без e-e взаимодействия')
            h_list.append(line7)
        if '0x' in to_plot: 
            line9, = ax.plot(tu0xa, su0xa, color = 'purple', label = 'точное решение')
            line10, = ax.plot(tl0xa, sl0xa, color = 'purple', label = 'точное решение')
            h_list.append(line9)

        ax.legend(handles=h_list)

        ax.set_box_aspect(2.0/3.5)
        
        plt.savefig(f"plot{no}_{i}_{j}_abs_a.pdf", format='pdf')
        plt.close()