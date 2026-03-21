# класс loop
#
# Автор: Элизабет Гоулд
# Дата последнего изменения: 19.03.2026
#
# Этот класс является основной частью этой библиотеки для решения начальных и
# граничных задач для нелинейного уравнения Шредингера данной модели. Класс
# может решать как начальную, так и краевую задачи в одном кольце для 
# заданного набора параметров.
#
# Он может решать уравнения как для линейного, так и для нелинейного случая.
# Для решения линейного случая нет необходимости устанавливать параметр
# нелинейности равным нулю. Просто нада вызывать методы, которые предоставляют
# решение для линейного случая. Эта функциональность была реализована для
# удобства сравнения обоих решений. Даже решение краевой задачи для линейного
# случая сохраняется и не удаляется при изменении начальной производной, 
# несмотря на то, что оно имеет другую начальную производную.
#
# Решения начальных и краевых задач являются отдельными функциональными 
# возможностями класса. Класс может решать или начальную или краевую задачу, но
# не обе вместе. При решении начальной задачи принимается производная в качестве
# входных данных и не требуется решения краевой задачи для запуска, в то время
# как решения краевой задачи не будут предоставлять соответствующего решения
# при запуске с использованием метода для решения начальной задачи. Волновая
# функция краевой задачи вычисляется с использованием смоделированных решений
# начальной задачи, которые были ранее найдены путем анализа решений программы
# для решения начальных задач. Нужно будет вызвать методы, которые дают 
# смоделированное решение при заданном значении x, чтобы получить решение
# краевой задачи. Методы для решения краевой задачи также сбрасывают производную,
# что приводит к удалению всех данных класса, полученных в результате решения
# начальной задачи. Однако данные о решении краевой задачи удаляются только при
# сбросе других параметров, но сохраняются при изменении только производной.
#
# Входными параметрами являются:
# - радиус кольца (R) в процентах от максимального ожидаемого радиуса кольца, 
# - напряженность магнитного поля (B) в процентах от максимальной ожидаемой 
#   напряженности поля,
# - волновое число электрона (k и dk), где k по умолчанию соответствует
#   волновому числу Ферми для золота, а dk - процент от периода синусоидальной
#   функции в кольце максимальной длины,
# - сила нелинейной связи (mu) как безразмерный коэффициент,
# - начальная волновая функция (A), которая по умолчанию равна 1. + 0.j,
# - и начальная производная волновой функции, которая должна быть найдена при
#   решении краевой задачи.
#
# Решения краевой задачи моделируются в предположении, что материалом колец 
# является золото, и эти решения не будут действительны для других материалов.
# Предполагается, что начальная волновая функция равна 1, и изменения в ней
# будут соответствовать изменению mu. (mu_eff = mu * |A|^2, где mu и A являются 
# входными данными, а mu_eff - это то, чему соответствуют наши решения.)
# Значения dk, равные 0, вызовут проблемы с граничным значением решение задачи
# для линейного случая, но предполагается, что оно не является физическим, а
# скорее это просто результат нашей неточности.

'''
Код и документация этого файла разделены на следующие разделы:
1. Библиотеки - Сюда импортируют внешние библиотеки или код из других источников
   этой библиотеки.
2. Значения по умолчанию - Здесь задаются значения по умолчанию, используемые
   в коде. Значения по умолчанию можно изменить либо непосредственно в коде,
   либо путем изменения переменной в вашем экземпляре класса loop, которая
   хранит это значение по умолчанию. Также можно просто вызвать соответствующие
   методы с выбранным значением и игнорировать значение по умолчанию. Есть
   несколько исключений, когда значения по умолчанию хранятся в файле consts.py,
   и эти значения нельзя изменить в рамках структуры класса, но их можно
   игнорировать, указав свое собственное значение.
3. Оглавление - Содержит список разделов класса loop и методов, содержащихся в
   нем.
4. Переменные - Содержит список переменных, содержащихся в классе loop, и их
   назначение.
5. Методы для вызова - Содержит краткое описание всех соответствующих методов
   класса.
6. class loop - наш код.
'''

# ---- БИБЛИОТЕКИ ------------
# Библиотеки численного анализа
import numpy as np
import scipy

# константы
from eelib.consts import pi, kFAu, R_max, B_max, phi0inv, rtol, atol
# дескрипторы функции
from eelib.deriv_functions import psi_deriv, psi_deriv_old
from eelib.events import deriv_amp, deriv_real
# модели
from eelib.k_M_models_ivp import pred_fast_k, pred_slow_k_v3, pred_fast_k_true
from eelib.bvp_rootfinder_functions import function_wrapper_bvp

# ---- ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ ------
# Допуски на погрешность
default_tol_root_finder    = 1e-20
default_tol_bvp_matching   = 0.01
# Выбор алгоритма SciPy
default_rootfinder         = 'broyden2'
default_integration_method = 'RK45'
# Выбранные из наших библиотек, основанные на моделировании решений начальных задач
default_k_model_bvp        = pred_fast_k_true
default_k_model_ivp        = pred_fast_k   
default_M_model            = pred_slow_k_v3

# Целые положительные числа по умолчанию (количество)
# Сколько быстрых циклов колебаний следует усреднить, чтобы оценить период
default_number_averaged_initial_oscillations        = 20
# Сколько точек должен вернуть интегратор
default_number_of_points_from_integration           = 1000 
# Как часто должно восстанавливать амплитуду (удаление ошибки)
default_number_fast_oscillations_between_recoveries = 4


# ---- ОГЛАВЛЕНИЕ --------

# 1 ----- Определение констант
#    __init__(R, B, dk, mu, k = kFAu, amp=1.)
#    ivp_solved(ivp_type="er")
#    clear_solution()
#    update_params(R=-1., B=-1., dk=-1., mu=-1., k=-1., amp=-1.)
#    calcInit()
#    setDeriv(p_prime)

# 2 ----- Аналитические расчеты для случая без e-e взаимодействия
#    rem(number)
#    adjdiv1r()
#    bjdiv1r()
#    normToAmp()
#    aj_calc()
#    bj_calc()
#    ajN_calc()
#    bjN_calc()
#    psij(x)
#    psij_pred(x)
#    psij_pred_true(x)
#    psij0(x)
#    psi_prime_0()
#    psi_prime_0_max()
#    amp_max()
#    amp_max_0()

# 3 ----- Оболочка функции для решение ОДУ
#    get_solve_code(solve_arr)
#    solve_ivp(n=None, percent_range = 1.0, method = None, rtol = rtol,
#          atol = atol, solve = 1)
#    ivp_solver_steps(t0, tf, y0, yp0, n=None, ee_int=True, m=None, method = None,
#          rtol = rtol, atol = atol, fullSol = False)
#    call_ivp_solver(t0, tf, y0, yp0, n=0, fullSol = True, t0_start = False,
#          ee_int= True, method = None, rtol = rtol, atol=atol, first_step = None,
#          max_step = np.inf, t_eval = None)

# 4 ----- Вычисление быстрых колебаний
#    find_fast_oscillations(n=None, method = None, rtol = rtol, atol = atol)
#    find_start_exact()
#    find_start_fast_oscillations(n, sol)
#    find_period_fast_oscillations(n, sol)
#    find_amplitude_fast_oscillations(sol)

# 5 ----- Медленные колебания и связанные вычисления
#    find_period_shift_exact()
#    find_slow_oscillations_start()
#    find_real_env_start()
#    find_t_points(n_points, t_max, t_start, T)

# 6 ----- Методы соединения краевых задач
#    find_root_rand(method = None, tol = None, ratio = 1.0)
#    find_root_many(tol_root_finder = None, tol_bvp_matching = None)
#    check_solution_for_boundary_matching(deriv_psi = None, tol = None)
#    find_root_min(tol_root_finder = None, tol_bvp_matching = None)

# 7 ----- Определение тока
#    current_old()
#    current_bvp()
#    current_new()
#    current_alt()
#    current_calc(psi=None, psi_pr=None)

# ---- ПЕРЕМЕННЫЕ -----

# Не изменяйте переменные вручную, за исключением указанных значений по
# умолчанию. Параметры кольца изменяются с помощью методов update_params и
# setDeriv, в то время как другие устанавливаются путем запуска кода, не
# указанного пользователем.

# -- Входные данные
# Параметры кольца
#   R (m)
#   k (hz), k0 + dk
#   B (T)
#   mu 
#   amp
# Сохраненные входные данные для тех случаев, когда требуются k0 и dk отдельно
#   k0 -- сохраняет входные данные
#   dk -- сохраняет входные данные
# Рассчитывается изначально, но может быть задан с помощью setDeriv
#   psi0_deriv_0 -- начальное значение psi'

# -- Производные величины
# Все эти переменные рассчитываются при установке наших параметров.
#   lngt     = 2 * pi * self.R
#   period_k = 2 * self.R * self.k
#   T0       = 2. * pi / self.k     
#   M        = self.B*self.R*phi0inv
#   denrem   = self.rem(self.R*self.M)
#   aj
#   bj
#   aj0      -- для точного соединенного решения
#   bj0      -- для точного соединенного решения
# Периоды из наших моделей. Также рассчитываются при установке параметров.
#   T_fast_mod      - полупериод быстрых колебаний с ошибкой
#   T_fast_mod_true - полупериод быстрых колебаний без ошибок
#   T_slow_mod      - период медленных колебаний
#
# Эти переменные рассчитываются с помощью find_fast_oscillations.
#   T_fast   -- расчетный полупериод быстрых колебаний
#   A_max    -- максимальное модуль волновой функции первых n быстрых колебаний
#   T_fast0  -- известен безошибочный период быстрых колебаний для линейного случая
#   T_fast_ex -- известен безошибочный период быстрых колебаний для линейного случая
#   T_fast_0_calc -- оценочный полупериод быстрых колебаний для линейного случая
#
# Исходные значения для нелинейных, линейных и точно-линейных решений для интегратора.
# Эти переменные должны быть первыми максимумами и минимумами модуля волновой функции.
#   stl
#   sth
#   stl0
#   sth0
#   stl_ex
#   stu_ex

# -- Индикаторы состояния
#   k_ee_estimated -- Оценили ли мы наше волновое число быстрых колебаний для
#         нелинейного случая на основе решения первых n колебаний?
#   k_0_estimated  -- Рассчитали ли мы наше волновое число быстрых колебаний для
#         линейного случая на основе решения первых n колебаний?
#   bvp_solved     -- Соответствует ли наша начальная производная psi той,
#         которая была определена путем решения краевой задачи?
#   deriv_set      -- Была ли изменена наша производная из решения линейной
#         краевой задачи?

# -- Решения полной начальной задачи  
#   solu    -- восстановленное нелинейное решение
#   solu_d  -- убывающее нелинейное решение
#   solu0   -- убывающее линейное решение
#   solu0_r -- восстановленное линейное решение
#   solu_m  -- восстановленное нелинейное решение с точками, выбранными на 
#              основе нашего смоделированного k
#
#   ivp     -- словарь следующих решений со следующими метками: 
#              ['er', 'ed', 'em', '0r', '0d']
#              'er' = восстановленная нелинейность
#              'ed' = убывающая нелинейность
#              'em' = восстановленное нелинейное, смоделированное
#              '0r' = восстановленное линейное решение
#              '0d' = убывающее линейное решение
#   percent_R_solved -- словарь процентов от радиуса R, который был получен для 
#                       данного решения

# -- Решение краевой задачи
# bvp_deriv  -- Производная, для которой параметры кольца решает краевую задачу.
#         Это значение устанавливается при решении краевой задачи, так что, если
#         мы позже изменим нашу производную, но не наши параметры, решение будет
#         сохранено.

# -- Значения по умолчанию
# При необходимости их можно изменить вручную. 
#   k_model_bvp -- дескриптор функции модели для определения волнового числа
#                  с быстрым колебанием без ошибок
#   k_model_ivp -- дескриптор функции модели для определения волнового числа
#                  с быстрым колебанием с ошибкой
#   M_model     -- дескриптор функции модели для волнового числа медленных колебаний
#   tol_root_finder
#   tol_bvp_matching
#   default_rootfinder         -- строка названия алгоритма поиска нуля функции SciPy
#   default_integration_method -- строка названия алгоритма поиска нуля функции SciPy
#   default_no_ave_init_osc
#   default_no_int_points
#   default_recovery_rate
#
#   solve_mu_0 -- Это логическое значение, которое определяет, должна ли оценка
#       для волнового числа быстрых колебаний включать в себя оценку для
#       линейного случая. По умолчанию оно равно True. Это необходимо для
#       точного определения огибающей линейного случая (mu = 0) при решении задачи
#       о начальном значении из-за ошибки в периоде. Если вы хотите решить
#       нелинейную начальную задачу, не решая линейную начальную задачу, установка
#       значения False сэкономит некоторое время.

# ---- МЕТОДЫ ДЛЯ ВЫЗОВА --------

# 1 ----- Определение констант
#    __init__(R, B, dk, mu, k = kFAu, amp=1.)
#    update_params(R=-1., B=-1., dk=-1., mu=-1., k=-1., amp=-1.) 
#        -- Изменят все переданные параметры кольца, сохраняя при этом старые
#           значения тех, которые не были переданы.
#    setDeriv(p_prime)         -- Устанавливает начальную производную от psi
#                                 на заданное значение.
#    ivp_solved(ivp_type="er") -- Указывает, существует ли запрошенное решение.

# 2 ----- Аналитические вычисления для случая без e-e взаимодействия
#    psij(x)           -- Возвращает точное решение psi без e-e взаимодействия 
#                         для заданной производной.
#    psij_pred(x)      -- Возвращает смоделированное решение psi с e-e
#                         взаимодействием для заданной производной.
#    psij_pred_true(x) -- Возвращает смоделированное решение psi без ошибок с
#                         e-e взаимодействием для заданной производной.
#    psij0(x)          -- Возвращает точное решение psi для решения краевой
#                         задачи без e-e взаимодействия.
#    psi_prime_0_max() -- Возвращает максимальное модуль psi' для решения
#                         краевой задачи без e-e взаимодействия. 
#                         Для использования при построении производных сеток.
#    amp_max()         -- Возвращает максимальное модуль psi для смоделированного
#                         решения без погрешности psi с e-e взаимодействия для
#                         заданной производной.
#    amp_max_0()       -- Возвращает максимальное модуль psi для точного решения
#                         psi без учета e-e взаимодействие, которое решает
#                         краевую задачу.

# 3 ----- Дескрипторы алгоритмов решения ОДУ
#    get_solve_code(solve_arr)  
#       -- Преобразует список кодов ('er', 'ed', '0r', '0d', 'em') в целое число
#          для параметра solve метода solve_ivp. 
#    solve_ivp(n=None, percent_range = 1.0, method = None, rtol = rtol,
#              atol = atol, solve = 30)
#           -- Используется для решения начальной задачи для заданных колец. 

# 4 ----- Вычисления быстрых колебаний
#    find_fast_oscillations(n=None, method = None, rtol = rtol, atol = atol)
#        -- Оценивает полупериод быстрого колебания по первым n колебаниям модуля
#           psi. Этот метод может быть запущена отдельно от метода решения
#           начальной задачи, но при необходимости метод решения начальной задачи
#           вызовет этот метод самостоятельно. Существует отдельный ввод
#           solve_mu_0, который указывает, следует ли оценивать k для линейного 
#           случая. По умолчанию для этого параметра установлено значение True,
#           и его необходимо вручную установить в значение False, чтобы избежать
#           дополнительных вычислений. Метод решения начальной задачи установит
#           значение True, если это необходимо, но проигнорирует его, если
#           линейные решения не будут вычислены.

# 5 ----- Медленные колебания и совместные вычисления
#    find_period_shift_exact() -- Я пыталась проследить и нарисовать огибающую
#         быстрых колебаний для конкретного случая. Но зто не работало. В конце
#         концов, я объявила это ненужным.
#    find_real_env_start() -- Указывает время для начала прохождения огибающей,
#         создаваемой быстрыми колебаниями.
#    find_t_points(n_points, t_max, t_start, T) -- Создает массив точек numpy,
#         для которых требуется интегрировать или построить график, интервал
#         между которыми должен быть кратен T. Используется для удержания оценки
#         в одной и той же точке при быстрых колебаниях.

# 6 ----- Соединение границ для краевой задачи
#    find_root_rand(method = None, tol = None, ratio = 1.0)
#       -- Пытается найти решение краевой задачи, используя указанный алгоритм
#          (method) и начальную точку. Начальной точкой является решение
#          линейной краевой задачи, разделенное на ratio.
#    find_root_many(tol_root_finder = None, tol_bvp_matching = None)
#       -- Пытается найти решение краевой задачи, используя несколько
#          алгоритмов и начальных точек. Возвращает список найденных решений.
#    check_solution_for_boundary_matching(deriv_psi = None, tol = None)
#       -- Проверяет, решает ли производная psi краевую задачу (на основе
#          смоделированного решения).
#    find_root_min(tol_root_finder = None, tol_bvp_matching = None)
#       -- Устанавливает производную к производной, которая решает краевую задачу.

# 7 ----- Поиск тока
#    current_old() -- Возвращает ток для случая решения линейной краевой задачи
#         с параметрами кольца.
#    current_bvp() -- Вычисляет краевую задачу и возвращает ток этого
#         нелинейного решения, используя модель для тока current_alt.
#    current_new() -- Возвращает ток нелинейного решения сохраненного кольца,
#         основанное на постоянном члене точного решения. Эта оценка для тока,
#         скорее всего, неверна.
#    current_alt() -- Возвращает ток нелинейного решения сохраненного кольца,
#         основываясь на предположительной форме этого тока. Скорее всего, это
#         верно.
#    current_calc(psi=None, psi_pr=None)   -- Возвращает ток нелинейного
#         решения кольца с сохраненными или указанными значениями psi и
#         производной psi на основе уравнения для расчета тока.
#                                             

# ---- КЛАСС LOOP ----------
class loop:

    # конструктор
    # При создании он принимает R, B, dk, mu, k и amp в качестве параметров.
    # R - это радиус в процентах от R_max.
    # B - это магнитное поле в процентах от B_max.
    # Волновая функция электрона равна k_el = k + dk/R_max /2,0.
    # Значение k по умолчанию равно kFAu (при условии, что проводником является
    #     золото).
    # psi(0) = amp
    # psi'(0) изначально определяется на основе известного решения для mu = 0.
    def __init__(self, R, B, dk, mu, k = kFAu, amp=1.):

        # параметры
        self.R   = R * R_max # R в микрометрах
        self.k   = k + dk/R_max/2.0 # dk прибавляет к минимуму k
        self.B   = B * B_max # измеряется в единицах 130 Гаусс
        self.mu  = mu
        self.amp = complex(amp)
        self.dk  = dk
        self.k0  = k 

        # значения по умолчанию
        # Они изменяются вручную или путем изменения указанных констант в коде.
        # Они используются только в том случае, если указанный параметр не
        # передан методам.
        self.k_model_bvp = default_k_model_bvp
        self.k_model_ivp = default_k_model_ivp
        self.M_model     = default_M_model

        self.tol_root_finder            = default_tol_root_finder
        self.tol_bvp_matching           = default_tol_bvp_matching
        self.default_rootfinder         = default_rootfinder
        self.default_integration_method = default_integration_method
        self.default_no_ave_init_osc    = default_number_averaged_initial_oscillations
        self.default_no_int_points      = default_number_of_points_from_integration
        self.default_recovery_rate      = default_number_fast_oscillations_between_recoveries

        self.solve_mu_0 = True # Установите значение False вручную, чтобы
            # сэкономить время на оценку полупериода быстрых колебаний.

        # Это указывает на то, какие шаги были предприняты в процессе решения.
        self.deriv_set  = False # False = производная по умолчанию, 
                                # True = выбранная производная
        self.bvp_deriv  = None  # Это значение еще не было рассчитано.
        self.clear_solution()   # Инициализирует переменные, которые начинаются
                                # неустановленными.

        self.calcInit() # Вычисляет множество производных величин.

    def ivp_solved(self, ivp_type = "er"):
        if self.ivp[ivp_type] is None: 
            return False
        else:
            return True

    # Удалите решения начальной задачи (поскольку они могут быть большими) и
    # укажите, что ничего не было решено (поскольку любое предыдущее решение
    # теперь может быть ошибочным).
    def clear_solution(self):
        self.k_ee_estimated = False
        self.k_0_estimated  = False

        self.ivp = {"er": None,   # с e-e взаимодействием, восстановленное решение
                    "ed": None,   # с e-e взаимодействием, убывающее решение
                    "em": None,   # с e-e взаимодействием, восстановленное решение,
                                  #    с интервалом, основанным на прогнозируемом k
                    "0r": None,   # без e-e взаимодействия, восстановленное решение
                    "0d": None}   # без e-e взаимодействия, убывающее решение
        self.percent_R_solved = {"er": None,   
                                  # с e-e взаимодействием, восстановленное решение
                    "ed": None,   # с e-e взаимодействием, убывающее решение
                    "em": None,   # с e-e взаимодействием, восстановленное решение,
                                  #    с интервалом, основанным на прогнозируемом k
                    "0r": None,   # без e-e взаимодействия, восстановленное решение
                    "0d": None}   # без e-e взаимодействия, убывающее решение

        self.bvp_solved = False

        self.solu    = None
        self.solu_d  = None
        self.solu_m  = None
        self.solu0   = None
        self.solu0_r = None

    # Значение по умолчанию -1 означает, что этот параметр не подлежит изменению.
    # В противном случае параметр обновляется до заданных значений.
    # Затем все шаги инициализации выполняются с новыми параметрами, 
    # удаляются старые данные, которые теперь неверны.
    def update_params(self, R=-1., B=-1., dk=-1., mu=-1., k=-1., amp=-1.):
        if R > 0.:
            self.R = R * R_max
        if B > 0.:           
            self.B = B * B_max
        if amp > 0.:
            self.amp = complex(amp)
        if mu > 0.:
            self.mu = mu
        if k > 0. and dk > 0.:
            self.k = k + dk /R_max/ 2.0
            self.dk = dk
            self.k0 = k
        elif dk > 0.:
            self.k = self.k0 + dk / R_max/ 2.0
            self.dk = dk
        elif k > 0.:
            self.k = k + self.dk / R_max / 2.0
            self.k0 = k

        self.clear_solution()  # Удаляются старые решения.
        self.deriv_set = False # False = производная по умолчанию,
                               # True = выбранная производная
        self.bvp_deriv = None  # Очистите старое решение.

        self.calcInit() # Пересчитайте производные величины.
        
    # Задает набор констант на основе данных, определяющих кольцо.
    def calcInit(self):
        self.lngt     = 2 * pi * self.R
        self.period_k = 2 * self.R * self.k
        self.T0 = 2. * pi / self.k
        
        self.M      = self.B*self.R*phi0inv
        self.denrem = self.rem(self.R*self.M)
        self.aj     = self.aj_calc()
        self.bj     = self.bj_calc()
        self.aj0    = self.aj
        self.bj0    = self.bj
        
        self.psi0_deriv_0 = self.psi_prime_0()

        self.T_fast_mod      = pi / self.k_model_ivp(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_fast_mod_true = pi / self.k_model_bvp(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_slow_mod      = 2 * pi / self.M_model(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        
    # Этот метод предназначен для установки начального значения psi' для кольца.
    # Он предназначен для многократного вызова.
    def setDeriv(self, p_prime):
        self.psi0_deriv_0 = p_prime

        self.aj = self.ajN_calc()
        self.bj = self.bjN_calc()

        self.clear_solution()
        self.deriv_set = True # False = производная по умолчанию,
                              # True = выбранная производная

        self.T_fast_mod      = pi / self.k_model_ivp(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_fast_mod_true = pi / self.k_model_bvp(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
        self.T_slow_mod      = 2 * pi / self.M_model(self.psi0_deriv_0, self.mu, 0., self.B, self.R, self.amp, self.k)
                

    # 2 ----- Аналитические вычисления 
    # для случая без e-e взаимодействия и для моделей с e-e взаимодействием.

    # термины aj и bj, так что aj + bj = amp
    def rem(self, number):
        resnum = np.modf(number)
        return resnum[0]
    
    def ajdiv1r(self):
        numrem = self.rem(self.B*np.square(self.R)*phi0inv-self.R*self.k)
        denom = np.exp(2j*pi*self.denrem) * 2j * np.sin(2*pi*self.k*self.R)
        num = 1 - np.exp(2j*pi*(numrem))
        return - num / denom
    
    def bjdiv1r(self):
        numrem = self.rem(self.B*np.square(self.R)*phi0inv+self.R*self.k)
        denom = np.exp(2j*pi*self.denrem) * 2j * np.sin(2*pi*self.k*self.R)
        num = 1 - np.exp(2j*pi*(numrem))
        return num / denom
    
    def normToAmp(self):
        return self.amp / (self.ajdiv1r()+self.bjdiv1r())
    
    def aj_calc(self):
        return self.ajdiv1r()*self.normToAmp()
    
    def bj_calc(self):
        return self.bjdiv1r()*self.normToAmp()
    
    # Используются при задании psi' с помощью setDeriv.
    def ajN_calc(self):
        den = 1.0 / 2.0 / self.k
        reA = den * (np.imag(self.psi0_deriv_0) - self.M + self.k)
        imA = - den * np.real(self.psi0_deriv_0)

        ret = 0.5*((self.psi0_deriv_0 - 1j * self.M * self.amp)/(1j*self.k)+self.amp)

        return ret
    
    def bjN_calc(self):
        den = 1.0 / 2.0 / self.k
        reB = den * (-np.imag(self.psi0_deriv_0) + self.M + self.k)
        imB = - den * np.real(self.psi0_deriv_0)

        ret = 0.5*(self.amp - (self.psi0_deriv_0 - 1j * self.M * self.amp)/(1j*self.k))

        return ret
    
    # Точное решение с текущей производной
    def psij(self, x):
        return self.aj*np.exp(1j*x*(self.k+self.M)) + self.bj*np.exp(1j*x*(-self.k+self.M))
    
    # Смоделированное решение с текущей производной, с ошибкой
    def psij_pred(self, x):
        kn = pi / self.T_fast_mod
        Mn = 2 * pi / self.T_slow_mod
        return self.aj*np.exp(1j*x*(kn+Mn)) + self.bj*np.exp(1j*x*(-kn+Mn))

    # Смоделированное решение с текущей производной, без ошибки
    def psij_pred_true(self, x):
        kn = pi / self.T_fast_mod_true
        Mn = 2 * pi / self.T_slow_mod
        return self.aj*np.exp(1j*x*(kn+Mn)) + self.bj*np.exp(1j*x*(-kn+Mn))
    
    # Точное решение краевой задачи
    def psij0(self, x):
        return self.aj0*np.exp(1j*x*(self.k+self.M)) + self.bj0*np.exp(1j*x*(-self.k+self.M))
    
    # Производная от точного решения краевой задачи
    def psi_prime_0(self):
        return 1j * (self.aj0*(self.k+self.M)+self.bj0*(-self.k+self.M))

    # Это максимальный модуль psi' без учета e-e взаимодействия, которое решает
    # краевую задачу.
    def psi_prime_0_max(self):
        #psi = e^(iMx)*[ae^(ikx)+be^(-ikx)]
        #psi' = iM*psi + ik* e^(iMx)*[ae^(ikx)-be^(-ikx)]
        #abs(psi') = abs[M *(ae^(ikx)+be^(-ikx)) + k* (ae^(ikx)-be^(-ikx)]
        # <= (M+k)*[abs(a)+abs(b)]
        d_max = (self.M+self.k)*(np.abs(self.aj0)+np.abs(self.bj0))
        return d_max
    
    # Это максимальный модуль psi для текущей производной.
    def amp_max(self):
        return np.sqrt(np.abs(self.aj)**2 + np.abs(self.bj)**2 + 2*np.abs(self.aj*np.conj(self.bj)))
    
    # Это максимальный модуль psi без учета e-e взаимодействия, которое решает
    # краевую задачу.
    def amp_max_0(self):
        return np.sqrt(np.abs(self.aj0)**2 + np.abs(self.bj0)**2 + 2*np.abs(self.aj0*np.conj(self.bj0)))
    
    # 3 ---- Методы для решения ОДУ

    # Преобразуют интуитивно понятную систему plot_code в неинтуитивную систему
    # целочисленных кодов для решения в solve_ivp.
    def get_solve_code(solve_arr):
        code = -1
        if "er" in solve_arr: code = code * -1
        if "ed" in solve_arr: code = code * 2
        if "0r" in solve_arr: code = code * 5
        if "0d" in solve_arr: code = code * 3
        if "em" in solve_arr: code = code * 7
        return code

    # Решает указанные начальную задачу с использованием указанного числового
    # алгоритма решения ОДУ.
    def solve_ivp(self, n = None, percent_range = 1.0, method = None, rtol = rtol, atol = atol, solve=1):
        # Я изменил способ установки значений по умолчанию, чтобы разрешить
        # редактирование значений по умолчанию или в объекте или в коде.
        if method is None:
            method = self.default_integration_method
        if n is None:
            n = self.default_no_int_points

        if (solve % 3 == 0 or solve % 5 == 0) and (self.solve_mu_0 == False):
            self.solve_mu_0 = True
            self.find_fast_oscillations()

        if self.k_ee_estimated == False:
            self.find_fast_oscillations()
 
        t0h = self.sth
        tfh = self.lngt * percent_range  # t_evalh[-1]
        #t0l = self.stl
        #tfl = self.lngt * percent_range  #t_evall[-1]

        if solve % 3 == 0 or solve % 5 == 0:
            t0h0 = self.sth0
            tfh0 = self.lngt * percent_range #t_vals[1]
            #t0l0 = self.stl0
            #tfl0 = self.lngt * percent_range #t_vals[3]

        # вычислить y0
        y0h = self.amp
        yp0h = self.psi0_deriv_0
        #y0l = self.amp
        #yp0l = self.psi0_deriv_0

        y0h0, yp0h0 = self.amp, self.psi0_deriv_0 #self.calc_yval_old(n, t0h0)
        #y0l0, yp0l0 = self.amp, self.psi0_deriv_0 #self.calc_yvals_old(n, t0l0)

        # вызывать метод для решения ОДУ несколько раз  
        # с e-e взаимодействием -- разделено на кусочки (+), без разделения(2), 
        # с предсказанным k_fast (7)
        # без e-e взаимодействия -- без разделения(3), разделено на кусочки(5)
        if solve > 0:      
            self.solu = self.ivp_solver_steps(t0h, tfh, y0h, yp0h, n, 
                            fullSol = False, ee_int = True, method = method,
                            rtol = rtol, atol = atol)
            #self.soll = self.ivp_solver_steps(t0l, tfl, y0l, yp0l, n,
            #               fullSol = False, ee_int = True, method = method, 
            #               rtol = rtol, atol = atol)
            self.percent_R_solved["er"] = percent_range
            self.ivp["er"] = self.solu
        if abs(solve)%2 == 0:
            self.solu_d = self.call_ivp_solver(t0h, tfh, y0h, yp0h, n,
                            fullSol = False, ee_int = True, method = method,
                            rtol = rtol, atol = atol)
            #self.soll_d = self.call_ivp_solver(t0l, tfl, y0l, yp0l, n, 
            #               fullSol = False, ee_int = True, method = method,
            #               rtol = rtol, atol = atol)
            self.percent_R_solved["ed"] = percent_range
            self.ivp["ed"] = self.solu_d
        if abs(solve)%3 == 0:
            self.solu0 = self.call_ivp_solver(t0h0, tfh0, y0h0, yp0h0, n,
                            fullSol = False, ee_int = False, method = method,
                            rtol = rtol, atol = atol)
            #self.soll0 = self.call_ivp_solver(t0l0, tfl0, y0l0, yp0l0, n,
            #               fullSol = False, ee_int = False, method = method,
            #               rtol = rtol, atol = atol)
            self.percent_R_solved["0d"] = percent_range
            self.ivp["0d"] = self.solu0
        if abs(solve)%5 == 0:
            self.solu0_r = self.ivp_solver_steps(t0h0, tfh0, y0h0, yp0h0, n,
                            fullSol = False, ee_int = False, method = method,
                            rtol = rtol, atol = atol)
            #self.soll0_r = self.ivp_solver_steps(t0l0, tfl0, y0l0, yp0l0, n,
            #               fullSol = False, ee_int = False, method = method,
            #               rtol = rtol, atol = atol)
            self.percent_R_solved["0r"] = percent_range
            self.ivp["0r"] = self.solu0_r
        if abs(solve)%7 == 0: 
            self.solu_m = self.ivp_solver_steps(t0h, tfh, y0h, yp0h, n,
                            fullSol = False, ee_int = True, method = method,
                            rtol = rtol, atol = atol, estimate_k = True)
            #self.soll_f = self.call_ivp_solver(t0l, tfl, y0l, yp0l, n,
            #               fullSol = False, ee_int = True, method = method,
            #               rtol = rtol, atol = atol)  
            self.percent_R_solved["em"] = percent_range
            self.ivp["em"] = self.solu_m      

    # Это сделано для того, чтобы компенсировать падение мощности в течение
    # непрерывных циклов, просто добавляя ее обратно. Вызывает алгоритм для
    # решения ОДУ несколько раз для более коротких интервалов.
    def ivp_solver_steps(self, t0, tf, y0, yp0, n=None, ee_int=True, m = None, method = None, rtol = rtol, atol = atol, fullSol = False, estimate_k = False):

        if method is None:
            method = self.default_integration_method
        if n is None:
            n = self.default_no_int_points
        if m is None:
            m = 4

        # Находить все точки, для которых мы хотим решить.
        if ee_int:
            if estimate_k:
                t_eval_full = self.find_t_points(n, t_max = tf, t_start = t0, T = 2*self.T_fast_mod)
            else:
                t_eval_full = self.find_t_points(n, t_max = tf, t_start = t0, T = 2*self.T_fast)
        else:
            t_eval_full = self.find_t_points(n,t_max = tf, t_start = t0, T = self.T_fast0)

        # желаемый размер шага для нашего алгоритм
        first_step = max_step = t_eval_full[1]-t_eval_full[0]

        # инициализация
        t0s = t0
        y0s = y0
        yp0s = yp0
        sol = []
        start_high = [[],[]]
        end_high = [[],[]]
        i = 1
        if m > len(t_eval_full): m = len(t_eval_full)
        if m == len(t_eval_full): tfs = tf
        else: tfs = t_eval_full[m + 1]

        t0_start = False

        # Решить до конца цепочки.
        while tfs <= tf:
            
            # определить точки, которые нужно найти
            if i+m+1 > len(t_eval_full): 
                t_eval_s = t_eval_full[i:]
                tfs = tf
            else: 
                t_eval_s = t_eval_full[i:i+m] 
                tfs = t_eval_full[i+m]

            if len(t_eval_s) == 0:
                break 

            # Решить следующий шаг.
            sol.append(self.call_ivp_solver(t0s, tfs, y0s, yp0s, fullSol = False,
                        t0_start=t0_start, ee_int = ee_int, method = method, 
                        rtol = rtol, atol = atol, first_step=first_step, 
                        max_step = max_step, t_eval = t_eval_s))
            
            # Если этот шаг слишком короткий, удалить данные шага и продолжать.
            if len(sol[-1]['y']) == 0:
                sol.pop()
                i += m
                continue

            # проблемы, связанные с запуском события 
            if np.shape(sol[-1]['y_events'][0])[0] < 2:
                sol.pop()
                i += m
                continue

            # найти окончательные решения
            yfs = sol[-1]['y_events'][0][-1,0]
            ypfs = sol[-1]['y_events'][0][-1,1]
            tfs = sol[-1]['t_events'][0][-1]

            y0a = sol[-1]['y_events'][0][0,0]
            y0a2 = sol[-1]['y_events'][0][1,0]

            t0_start = True
            
            # Выберить ту же сторону события модуля psi.
            
            if abs(yfs) < abs(y0a) and abs(yfs) < abs(y0a2):
                rto = min(abs(y0a), abs(y0a2)) /abs(yfs)
                end_high[0].append(False)
            else:
                rto = max(abs(y0a), abs(y0a2))/abs(yfs)
                end_high[0].append(True)
            if abs(y0a) < abs(y0a2):
                start_high[0].append(False)
            else:
                start_high[0].append(True)

            # Выберите ту же сторону события действительной часть psi.
            y_event_first = np.real(sol[-1]['y_events'][1][0,0])
            y_event_last = np.real(sol[-1]['y_events'][1][-1,0])

            if y_event_last < 0:
                end_high[1].append(False)
            else:
                end_high[1].append(True)
            if y_event_first < 0:
                start_high[1].append(False)
            else:
                start_high[1].append(True)

            # Измените масштаб для нового запуска.

            i += m
            t0s = tfs
            y0s = yfs * rto
            yp0s = ypfs * rto


        # объедините целое решение
        # t, y, t_events, y_events, status   
        # t

        sol_t_list = []
        sol_y_list = []
        sol_out_tev = []
        sol_out_yev = []

        for ev in sol:
            sol_t_list.append(ev['t'][1:])
            sol_y_list.append(ev['y'][:,1:])

        if len(sol) == 0:
            return {
                't': np.array([0]),
                'y': np.array([0,0]),
                'status': [3],
                't_events': [np.array([0]), np.array([0])],
                'y_events': [np.array([0,0]),np.array([0,0])]
            }    

        for i in range(len(sol[0]['t_events'])):
            sol_tev_list = []
            sol_yev_list = []

            for j,ev in enumerate(sol):

                if j > 0 and end_high[i][j-1]==start_high[i][j]:
                    sol_tev_list.append(ev['t_events'][i][1:])
                    sol_yev_list.append(ev['y_events'][i][1:, :])
                else:
                    sol_tev_list.append(ev['t_events'][i])
                    sol_yev_list.append(ev['y_events'][i])


            sol_out_tev.append(np.concatenate(tuple(sol_tev_list)))
            sol_out_yev.append(np.concatenate(tuple(sol_yev_list), axis = 0))


        sol_out_t = np.concatenate(tuple(sol_t_list))
        sol_out_y = np.concatenate(tuple(sol_y_list), axis = 1)
        
        sol_out_status = []
        for ev in sol:
            sol_out_status.append(ev['status'])

        
        return {
            't': sol_out_t,
            'y': sol_out_y,
            'status': sol_out_status,
            't_events': sol_out_tev,
            'y_events': sol_out_yev
        }

    # дескриптор функции библиотеки scipy для решения ОДУ
    def call_ivp_solver(self, t0, tf, y0, yp0, n = 0, fullSol = True, 
                        t0_start=False, ee_int = True, method = None, 
                        rtol = rtol, atol = atol, first_step=None, 
                        max_step = np.inf, t_eval = None):

        if method is None:
            method = self.default_integration_method

        # выбрать уравнение Шредингера
        if ee_int:
            y_hand = psi_deriv
        else:
            y_hand = psi_deriv_old
            
        # диапазон, начальные значения и параметры
        if t0_start:
            t_range = [t0,tf]
        else: 
            t_range = [0.0, tf] # sequence
        y_0 = np.array([y0, yp0]) # array_like, shape (n,) 
        arg = (self.k, self.B, self.R, self.mu, 1000.) # tuple
        
        # события
        elist = [deriv_amp, deriv_real]
        
        # вызвать функцию
        if fullSol:
            sol = scipy.integrate.solve_ivp(y_hand, t_range, y_0, method=method,
                        first_step=first_step, max_step=max_step, t_eval=t_eval,
                        args=arg, events=elist, rtol = rtol, atol = atol) 
        else:
            sol = scipy.integrate.solve_ivp(y_hand, t_range, y_0, method=method,
                        first_step=first_step, max_step=max_step, t_eval=t_eval,
                        args=arg, events=elist, rtol = rtol, atol = atol) 
        # вернуть решение
        return sol

    # 4 ------ Вычисления для быстрых колебаний ------

    # Эти методы определяют (полу)период и начальную точку быстрых колебаний

    # Этот метод оценивает полупериод быстрых колебаний путем усреднения
    # интервала между n колебания квадрата модуля psi, найденного с помощью
    # нашего метода для решения ОДУ. self.solve_mu_0 можно изменить на значение
    # False, чтобы удалить нашу оценку линейного решения и ускорить вычисления.
    # Мы оцениваем это, потому что T0 неверно оценивает период из-за ошибки. 
    def find_fast_oscillations(self, n=None, method = None, rtol = rtol, atol = atol):

        if method is None:
            method = self.default_integration_method
        if n is None:
            n = self.default_no_ave_init_osc

        # выберите percent_range

        maxX = n*self.T0
        per_range = maxX / self.lngt

        sol1 = self.call_ivp_solver(0.0, self.lngt *per_range, self.amp,
                    self.psi0_deriv_0, method = method, rtol = rtol,
                    atol = atol) # find the first n cycles
        if self.solve_mu_0: 
            sol2 = self.call_ivp_solver(0.0, self.lngt *per_range, self.amp,
                    self.psi0_deriv_0, ee_int = False, method = method,
                    rtol = rtol, atol = atol) # find the first n cycles

        # найдите период событий
        self.T_fast = self.find_period_fast_oscillations(n, sol1)
        self.A_max = self.find_amplitude_fast_oscillations(sol1)
        if self.solve_mu_0: self.T_fast_0_calc = self.find_period_fast_oscillations(n, sol2)
        self.T_fast0 = self.T0
        self.T_fast_ex = self.T0

        # найдите начальную точку событий
        start_ee = self.find_start_fast_oscillations(n, sol1)

        if self.solve_mu_0: 
            start_0 =  self.find_start_fast_oscillations(n, sol2)

        self.stl = start_ee[0]
        self.sth = start_ee[1]
        if self.solve_mu_0:
            self.stl0 = start_0[0]
            self.sth0 = start_0[1]
            self.k_0_estimated  = True

        self.stl_ex, self.stu_ex = self.find_start_exact()

        self.k_ee_estimated = True

    # возвращать значения t_min, t_max
    def find_start_exact(self):
        # найти экстремумы -- максимум, минимум и две средние точки

        x1 = pi - np.angle(self.aj*np.conj(self.bj))/2.0
        x2 = x1 - pi / 2.0
        x1 = x1 / self.k
        x2 = x2 / self.k

        # упорядочиваем решения
        if self.psij(x1) > self.psij(x2):
            x3 = x1
            x1 = x2
            x2 = x3

        return x1, x2

    # Первый раз, когда модули в квадрате колебаний достигают минимума и
    # максимума значения, полученные из решения ОДУ.
    def find_start_fast_oscillations(self,n,sol):
        t_1 = sol['t_events'][0][0]
        t_2 = sol['t_events'][0][1]
        y_1 = np.abs(sol['y_events'][0][0][0])
        y_2 = np.abs(sol['y_events'][0][1][0])
        
        if np.abs(y_2) < np.abs(y_1):
            t_low = t_2
            t_high = t_1
        else:
            t_low = t_1
            t_high = t_2
        return [t_low, t_high]
        
    # Вычислять период колебаний по заданному решению, вычисляемый путем оценки
    # первых n событий (экстремумов модуля).
    def find_period_fast_oscillations(self,n,sol):
        t = sol['t_events'][0]
        t2 = t[1:n]
        t1 = t[0:n-1]
        td = t2 - t1
        dt = np.average(td)
        return dt
    
    # Найти максимальный модуль, записанное в событиях.
    def find_amplitude_fast_oscillations(self,sol):
        amp = np.abs(sol['y_events'][0][:, 0])
        max = np.max(amp)
        return max
    
    def find_period_shift_exact(self):
        fast = self.T0
        slow = 2.*pi / self.M
        pos = 2.*pi/(self.k + self.M)
        neg = 2.*pi/(self.k - self.M)
        return fast, slow, pos, neg
    
    def find_slow_oscillations_start(self):
        den = 2 * self.M
        x1 = 1j * (np.log(self.aj)+np.log(self.bj)) / den # max
        x1 = np.real(x1)
        dx = pi / den
        T = pi / den
        nT = np.floor(x1 / (2. * T))
        x2 = x1 - nT * 2. * T
        x3 = x2 + T
        if x3 > 2.* T:
            x3 = x3 - 2. * T

        return x3, x2

    def find_real_env_start(self):
        xTmin, xTmax = self.find_slow_oscillations_start()
        xtmin, xtmax = self.find_start_exact()
        t = self.T0

        ymin = self.psij(xtmin)
        ymax = self.psij(xtmax)
        y = max(abs(ymin), abs(ymax))
        if y == ymin: xt = xtmin
        else: xt = xtmax

        n_osc = np.ceil((xTmin - xt) / t)
        x_st1 = xt + n_osc * t
        if self.psij(x_st1) == 0: x_st1 += t/4
        n_osc = np.ceil((xTmax - xt) / t)
        x_st2 = xt + n_osc * t
        if self.psij(x_st2) == 0: x_st2 += t/4
        y_st = max(abs(self.psij(x_st1)), abs(self.psij(x_st2)))
        if y_st == abs(self.psij(x_st1)): x_st = x_st1
        else: x_st = x_st2
        return x_st

    # Возвращает n_point величины t для верхней или нижней огибающей колебаний
    # с периодом T. Эти точки расположены на линейном расстоянии друг от друга,
    # начиная с точки t_start и заканчивая как можно ближе к точке t_max.
    def find_t_points(self, n_points, t_max, t_start, T):
        p0 = t_start
        nT = np.floor((t_max - t_start) / T)  
        count = np.floor(nT / (n_points - 1)) 
        
        pf = t_start + count * (n_points -1) * T
        x1 = np.linspace(p0, pf, n_points)
        return x1

    
    # 6 -- Методы для решения краевых задач

    # Это позволит найти случайное решение для краевой задачи. Изменение method и
    # prep приведет к изменению того, к какому решению алгоритм сходится.
    def find_root_rand(self, method = None, tol = None, ratio = 1.0):
        mu = self.mu
        dk = self.dk
        B = self.B
        R = self.R

        if tol is None:
            tol = self.tol_root_finder
        if method is None:
            method = self.default_rootfinder


        model_k = self.k_model_bvp
        model_M = self.M_model

        #kfun = lambda dpsi, mu, dk, B, R, A, k0: self.k

        # Найди нуль функции соединения линейного уравнения
        #yy = lambda x: fun(x, mu, dk, B, R, k_calc_f=kfun, M_calc_f=M_calc_0)
        #jj = lambda x: jac(x, mu, dk, B, R, k_calc_f=k_calc_0, M_calc_f=M_calc_0, dk_calc_f=dk_calc_0, dM_calc_f=dM_calc_0)

        der_0 = self.psi_prime_0()
        #sol = scipy.optimize.root(yy, [-1e12, 1e12], method='hybr')

        xs = np.array([np.real(der_0), np.imag(der_0)])
        #print(sol.x)

        # Найди нуль функции соединения нелинейного уравнения
        yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=model_k, M_calc_f=model_M)
        #jj = lambda x: jac(x, mu, dk, B, R)

        sol = scipy.optimize.root(yy, xs/ratio, method=method, tol=tol)
        dpsi0c = sol.x

        deriv_psi = dpsi0c[0] + 1j * dpsi0c[1]
        #deriv_psi_o = xs[0]+1j*xs[1]

        return deriv_psi

    # Это позволит найти несколько решений для краевой задачи. Представляется,
    # это работает достаточно хорошо, чтобы найти способ обойти проблемы
    # решения. Решение с наименьшей амплитудой является для нас самым важным,
    # и найдены не все решения, а только самые маленькие из них. Основная
    # проблема с этим алгоритмом - это затраты времени, которые будут
    # значительными при использовании большой сетки. 
    def find_root_many(self, tol_root_finder = None, tol_bvp_matching = None):

        if tol_root_finder is None:
            tol_root_finder = self.tol_root_finder
        if tol_bvp_matching is None:
            tol_bvp_matching = self.tol_bvp_matching

        mu = self.mu
        dk = self.dk
        B = self.B
        R = self.R

        model_k = self.k_model_bvp
        model_M = self.M_model

        der_0 = self.psi_prime_0()
        xs = np.array([np.real(der_0), np.imag(der_0)])

        method = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']
        prep = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0]

        # Найди нуль функции соединения нелинейного уравнения
        yy = lambda x: function_wrapper_bvp(x, mu, dk, B, R, k_calc_f=model_k, M_calc_f=model_M)
        #jj = lambda x: jac(x, mu, dk, B, R)

        dpsi0c = []

        for mm in method:
            for pp in prep:
                try:
                    sol = scipy.optimize.root(yy, xs/pp, method=mm, tol=tol_root_finder)
                    dpsi0c.append(sol.x)
                except:
                    pass

        dpsi0f = []

        # Удалите неудачные варианты
        for ii in dpsi0c:
            deriv_psi = ii[0] + 1j * ii[1]
            try:
                self.setDeriv(deriv_psi)
                sol = self.psij_pred_true(2*pi*self.R) - 1.0
                if np.abs(sol) < tol_bvp_matching:
                    new = True
                    for jj in dpsi0f:
                        if np.abs((jj - deriv_psi)/(jj+deriv_psi)) < tol_bvp_matching:
                            new = False
                            break
                    if new: dpsi0f.append(deriv_psi)
            except:
                pass
            
        return dpsi0f

    # Это проверит, является ли производная (измененная или сохраненная)
    # решением краевой задачи в пределах заданного допуска.
    def check_solution_for_boundary_matching(self, deriv_psi = None, tol = None):

        if tol is None:
            tol = self.tol_bvp_matching

        if deriv_psi is not None:
            self.setDeriv(deriv_psi)
        sol = self.psij_pred_true(2*pi*self.R) - 1.0

        if np.abs(sol) < tol:
            return True
        else:
            return False
    
    # Это приведет к тому, что производная будет соответствовать правильному
    # решению краевой задачи. Будет отмечен тот факт, что краевая задача решена,
    # и решение сохранено. bvp_solved обновляется при вызове setDeriv, но
    # bvp_deriv обновляется только для update_params.
    # (На самом деле я не использую это в коде.)
    def find_root_min(self, tol_root_finder = None, tol_bvp_matching = None):

        if tol_root_finder is None:
            tol_root_finder = self.tol_root_finder
        if tol_bvp_matching is None:
            tol_bvp_matching = self.tol_bvp_matching

        deriv_list = self.find_root_many(tol_root_finder, tol_bvp_matching)
        A_list = []
        for deriv_psi in deriv_list:
            self.setDeriv(deriv_psi)
            A_list.append(self.l_calc.amp_max())
        if len(A_list) > 0:
            i = np.argmin(A_list)
            self.setDeriv(deriv_list[i])
            self.bvp_solved = True
            self.bvp_deriv = deriv_list[i]

    # 7 -- Ток для точного решения и нелинейного решения

    # Это ток линейного решения краевой задачи.
    def current_old(self):
        ajsq = np.square(np.real(self.aj0)) + np.square(np.imag(self.aj0))
        bjsq = np.square(np.real(self.bj0)) + np.square(np.imag(self.bj0))
        return ajsq - bjsq

    # Это ток, который мы ожидаем для нелинейного решения.
    def current_bvp(self):
        if self.bvp_solved == False:
            if self.bvp_deriv is None:
                self.find_root_min()
            else:
                self.setDeriv(self.bvp_deriv)
        ajsq = np.square(np.real(self.aj)) + np.square(np.imag(self.aj))
        bjsq = np.square(np.real(self.bj)) + np.square(np.imag(self.bj))
        return ajsq - bjsq
    
    # Эти методы, приведенные ниже, не предназначены для проверки того,
    # соответствует ли данное решение условиям на границе, поэтому будьте
    # осторожны.

    def current_new(self):
        kn = pi / self.T_fast_mod
        mn = 2* pi /self.T_slow_mod
        ko = self.k
        mo = self.M

        ajsq = np.square(np.real(self.aj)) + np.square(np.imag(self.aj))
        bjsq = np.square(np.real(self.bj)) + np.square(np.imag(self.bj))

        # Это слагаемое не зависимое от положения решения для тока данного
        # смоделированного нелинейного решения с предположением, что
        # быстрые колебания являются синусоидальными.
        cur_new = (ajsq + bjsq)*(mn-mo)/ko + kn/ko * (ajsq - bjsq)

        return cur_new
    
    # Я подозреваю, что это фактический ток из-за того, как, по-видимому,
    # встроена сходимость.
    def current_alt(self):
        ajsq = np.square(np.real(self.aj)) + np.square(np.imag(self.aj))
        bjsq = np.square(np.real(self.bj)) + np.square(np.imag(self.bj))
        return ajsq - bjsq

    # Этот третий метод определяется в нашей начальной точке (или в другой
    # точке для переданных значений).
    def current_calc(self, psi=None, psi_pr=None):
        if psi is None: psi = self.amp
        if psi_pr is None: psi_pr = self.psi0_deriv_0
        t1 = 1/ self.k / 2j
        t2 = -2j * self.B * self.R * phi0inv
        return t1*(np.conj(psi)*psi_pr - psi*np.conj(psi_pr) + t2*np.conj(psi)*psi)
