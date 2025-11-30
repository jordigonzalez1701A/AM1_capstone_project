from temporal_schemes import Cauchy_problem, Cauchy_problem_intersect_y, RK4
from ordinary_differential_equations import CRTBP_variacional_JPL
from numpy import zeros, identity, reshape, array, sqrt, imag, pi, where, savetxt, any, hstack, loadtxt
from numpy.linalg import solve, eigvals
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Folder where THIS script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build universal paths
filename = os.path.join(BASE_DIR, "Lyap_family_Sun_Earth_L1.txt")
filename_CIs_principales = os.path.join(BASE_DIR, "L1_Sun_Earth_CI.txt")

def build_initial_condition(U0):
    """
    Construye la condición inicial del problema variacional a partir del vector (x,y,z,vx,vy,vz)
    Inputs:
    U0: Condición inicial del CRTBP en el espacio de fases.
    Outputs:
    V0 en forma de vector de 42 dimensiones.
    """

    V0 = zeros(6*6+6)
    V0[0:6] = U0[0:6]
    Id = identity(6)
    V0[6:] = reshape(Id, (6*6), copy=True)
    return V0

def Lagrange_points_position(mu):
    """
    Devuelve la posición de los 5 puntos de Lagrange.
    Inputs:
    mu: Masa reducida del CRTBP.
    Outputs:
    array de 5x2 con las posiciones ((x,y)_1, (x,y)_2, ..., (x,y)_5) de los puntos de Lagrange
    """
    def l1_poly(gamma):
        return gamma**5 + (mu-3)*gamma**4 + (3-2*mu)*gamma**3 - mu*gamma**2 + 2*mu*gamma - mu
    def l2_poly(gamma):
        return gamma**5 + (3-mu)*gamma**4 + (3-2*mu)*gamma**3 - mu*gamma**2 - 2*mu*gamma - mu
    def l3_poly(gamma):
        return gamma**5 +(2+mu)*gamma**4+(1+2*mu)*gamma**3-(1-mu)*gamma**2-2*(1-mu)*gamma-(1-mu)
    def newton(func, gamma0):
        N_max = 10000
        delta = 1e-11
        gamma_n = gamma0
        gamma_n_plus_one = gamma0
        tol = 1e-10
        for n in range(0, N_max):
            f_n = func(gamma_n)
            f_prime_n = (func(gamma_n + delta) - func(gamma_n-delta)) / (2*delta)
            gamma_n_plus_one = gamma_n - f_n/f_prime_n

            if(abs(gamma_n_plus_one-gamma_n)<tol):
                return gamma_n_plus_one
            
            gamma_n = gamma_n_plus_one
        
        return gamma_n_plus_one
    l1x = newton(l1_poly, (mu/3)**(1/3))
    l2x = newton(l2_poly, (1-(7/12)*mu))
    l3x = newton(l3_poly, (1-(7/12)*mu))
    print(f"l3x:{l3x}")
    l4 = array([-mu+0.5, sqrt(3)/2])
    l5 = array([-mu+0.5, -sqrt(3)/2])
    return array([array([-(l1x+mu-1),0]), array([-(-l2x+mu-1),0]), array([-(l3x+mu),0]), l4, l5])

def estimate_T_half_0(mu, l_point_index):
    """
    Estima el semiperiodo de una órbita de Lyapunov alrededor de un punto de Lagrange.
    INPUTS:
    mu: Parámetro de masa.
    l_point_index: Índice del punto de Lagrange: 1 para L1, 2 para L2, 3 para L3
    OUTPUTS:
    T_half_0: Semiperiodo estimado en unidades de tiempo.
    """
    # Encontrar punto de Lagrange objetivo
    lagrange = Lagrange_points_position(mu)
    l_point_vector = array([lagrange[l_point_index-1,0], lagrange[l_point_index-1,1], 0, 0,0,0])
    # Estimar semiperiodo
    d1 = sqrt((l_point_vector[0]+mu)**2 + l_point_vector[1]**2 + l_point_vector[2]**2)
    d2 = sqrt((l_point_vector[0]-1+mu)**2 + l_point_vector[1]**2 + l_point_vector[2]**2)
    Uxx = 1+3*(1-mu)*(l_point_vector[0]+mu)**2/d1**5 - (1-mu)/d1**3 + 3*mu*(l_point_vector[0]-1+mu)**2/d2**5 - mu/d2**3
    Uyy = 1-(1-mu)*(d1**2-3*l_point_vector[1]**2)/d1**5 -mu*(d2**2-3*l_point_vector[1]**2)/d2**5
    beta1 = 2-0.5*(Uxx+Uyy)
    beta2_square = -Uxx*Uyy
    s = sqrt(beta1+sqrt(beta1**2 + beta2_square))
    beta3 = (s**2-Uxx)/(2*s)
    xpert0 = 1e-8
    vy0 = -xpert0*beta3*s
    return (pi)/s

def find_Lyapunov_orbit(V0, mu, T0, temporal_scheme, **kwargs):
    """
    Encuentra la condición inicial para una órbita planar de Lyapunov partiendo de una
    condición inicial del tipo (x0,0,0,0,vy0,0, Id6x6)
    Inputs:
    mu: Masa reducida
    V0: Condición inicial
    T0: Estimación inicial semiperiodo.
    temporal_scheme: Esquema temporal para la integración
    Outputs:
    converged: booleano True si ha habido convergencia, False en caso contrario
    V0: Condición inicial de la órbita de Lyapunov
    T: Periodo de la órbita de Lyapunov
    """
    epsilon = 1e-9

    N_max = 800000
    n_iter = 0
    converged = False
    T_half = T0
    plot_intermediate_steps = False
    while(n_iter < N_max):
        sol = solve_ivp(CRTBP_variacional_JPL, t_span=(0, 1.25*T_half), y0=V0, method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12)

        # Calcular punto de cruce
        for i in range(0, sol.y.shape[1]):
            if i>0 and sol.y[1,i]*sol.y[1,i-1]<0:
                y0 = sol.y[1, i-1]
                y1 = sol.y[1, i]
                alpha = y0 / (y0 - y1)  # fraction between i-1 and i
                V_cross = sol.y[:, i-1] + alpha * (sol.y[:, i] - sol.y[:, i-1])
                T_half = sol.t[i-1] + alpha * (sol.t[i] - sol.t[i-1])
                #print(f"T_half interpolate: {T_half}")
                break
        
        if plot_intermediate_steps == True:
            plt.plot(sol.y[0], sol.y[1], marker="o", label="Lyapunov Orbit")
            plt.scatter(V_cross[0], V_cross[1], label="Vcross", color="green", s=100)
            plt.scatter(V0[0], V0[1], color="yellow", label="V0", s=100)
            plt.axhline(0, color='black', linewidth=1) 
            plt.axis("equal")
            plt.legend()
            plt.show()

        # Paso de corrección de vy
        d1 = sqrt((V_cross[0]-mu)**2 + V_cross[1]**2 + V_cross[2]**2)
        d2 = sqrt((V_cross[0]-mu+1)**2 + V_cross[1]**2 + V_cross[2]**2)
        Phi = reshape(V_cross[6:], (6,6))

        Jacobian = zeros((2,2))
        ddotx = V_cross[0] + 2*V_cross[4]-(1-mu)*(V_cross[0]+mu)/d1**3-mu*(V_cross[0]-1+mu)/d2**3
        ddotz = -((1-mu)/d1**3 + mu/d2**3)*V_cross[2]
        Jacobian[0,0] = Phi[3,2]-Phi[1,2]*(ddotx/V_cross[4])
        Jacobian[0,1] = Phi[3,4]-Phi[1,4]*(ddotx/V_cross[4])
        Jacobian[1,0] = Phi[5,2]-Phi[1,2]*(ddotz/V_cross[4])
        Jacobian[1,1] = Phi[5,4]-Phi[1,4]*(ddotz/V_cross[4])

        minus_F = zeros(2)
        minus_F[0] = -V_cross[3]
        minus_F[1] = -V_cross[5]
        delta_z_doty = solve(Jacobian, minus_F)

        v_y_new = V0[4] + delta_z_doty[1]
        z_new = V0[2] + delta_z_doty[0]
        V0[4] = v_y_new
        V0[2] = z_new
        n_iter = n_iter + 1
        
        print(f"n_iter: {n_iter}. vx(T): {V_cross[3]:.4e}\r", end="\r", flush=True)
        if abs(V_cross[3])< epsilon:
            converged = True
            print(f"T_half: {T_half}")
            return converged, V0, 2*T_half
    
    return converged, V0, 2*T_half

def cargar_CIs_principales(filename):
    """
    Carga las condiciones iniciales principales (las que se usan para hacer continuación).
    Los archivos TXT deben contener las CIs como x,y,z,vx,vy,vz\r\n.
    INPUTS: 
    filename: nombre del archivo.
    Returns:
        ics: NumPy array de tamaño (N,6) donde N es la cantidad de CIs
    """
    ics_list = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  
            numbers = line.split(',')
            numbers = [float(n) for n in numbers]
            numbers[1] = abs(numbers[1])
            ics_list.append(numbers)

    ics = array(ics_list, dtype=float)
    
    return ics

def guardar_CIs(V0_Lyap_family, Lyap_p_fam, filename):
    """
    Guarda las condiciones iniciales encontradas por continuación en un archivo CSV. 
    Iputs:
    V0_Lyap_family: Tensor N_familyx42 con las condiciones iniciales encontradas.
    Lyap_p_fam: Periodos de las orbitas.
    filename: Nombre de archivo donde guardar.
    """
    # Only save orbits that were found
    found_indices = where(any(V0_Lyap_family, axis=1))[0]
    V0_to_save = V0_Lyap_family[found_indices, :]
    periods_to_save = Lyap_p_fam[found_indices]
    # Combine periods as first column (optional) + initial conditions
    data_to_save = hstack((periods_to_save.reshape(-1,1), V0_to_save))

    # Save as CSV
    with open(filename, 'a') as f:
        savetxt(f, data_to_save, delimiter=",", comments='')

def cargar_CIs(filename):
    """
    Carga las condiciones iniciales de un archivo CSV.
    
    Inputs:
    filename: nombre del archivo que contiene las condiciones iniciales.
    
    Returns:
    periods: array 1D con los periodos de las órbitas.
    V0_Lyap_family: array 2D (N_family x 42) con las condiciones iniciales.
    """
    # Load data, skipping the header
    data = loadtxt(filename, delimiter=",", skiprows=1)
    periods = data[:, 0]
    V0_Lyap_family = data[:, 1:]
    
    return periods, V0_Lyap_family

def Lyapunov_family(V0, mu, temporal_scheme, N_family, delta_x0, Th0, filename, **kwargs):
    """
    Continua condiciones iniciales a partir de una CI dada para encontrar una familia de órbitas de Lyapunov.
    Guarda las condiciones iniciales en un txt.
    INPUTS:
    V0: Condición inicial que continuar.
    mu: Masa reducida.
    temporal_scheme: Esquema temporal.
    N_family: Número de condiciones iniciales a encontrar. 
    delta_x0: Intervalo de perturbación en la componente x de la condición inicial.
    Th0: Estimación inicial del semiperiodo de la primera órbita de la continuación.
    filename: Nombre del archivo donde guardar las condiciones iniciales.
    OUTPUT:
    CSV con las condiciones iniciales continuadas.
    """
    V0_Lyap_family = zeros((N_family,6*6+6))
    Lyap_period_family = zeros(N_family)
    for i in range(0, N_family):
        print(f"Orbit number {i+1}/{N_family}\r")
        delta_X = zeros(6*6+6)
        delta_X[0] = i*delta_x0
        V0_IN = V0 + delta_X
        Lyap_found, V0_Lyap, Lyap_period = find_Lyapunov_orbit(V0_IN, mu, Th0, temporal_scheme)
        if Lyap_found == True:
            V0_Lyap_family[i,:] = V0_Lyap
            Lyap_period_family[i] = Lyap_period
    guardar_CIs(V0_Lyap_family, Lyap_period_family, filename)
    return V0_Lyap_family, Lyap_period_family

def plot_Lyapunov_family(mu, temporal_scheme, N_CI_pp, N_family, V0_Lyap_family, Lyap_period_family):
    for i in range(0, N_CI_pp):
        for j in range(0, N_family): 
            sol_Lyap = solve_ivp(CRTBP_variacional_JPL, t_span=(0, Lyap_period_family[i,j]), y0=V0_Lyap_family[i,j], method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12)
            print(f"i:{i}")
            plt.plot(sol_Lyap.y[0,:], sol_Lyap.y[1,:], marker="o")  
    plt.axis("equal")
    plt.show()

    return

def plot_Lyapunov_family_one_source(mu, temporal_scheme, N_family, V0_Lyap_family, Lyap_period_family):

    for j in range(0, N_family): 
        sol_Lyap = solve_ivp(CRTBP_variacional_JPL, t_span=(0, Lyap_period_family[j]), y0=V0_Lyap_family[j], method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12)
        print(f"i:{i}")
        plt.plot(sol_Lyap.y[0,:], sol_Lyap.y[1,:], marker="o")  
    plt.axis("equal")
    plt.show()

    return

def Jacobi_constant_equation(U, C, mu):
    d = sqrt((U[0]+mu)**2 + U[1]**2 + U[2]**2)
    r = sqrt((U[0]-1+mu)**2 + U[1]**2 + U[2]**2)
    return C + U[3]**2  + U[4]**2 + U[5]**2 - 2*((U[0]**2 + U[1]**2)/2 + mu/r + (1-mu)/d)

def stability(V_T):
    """
    Calcula los autovalores de la matriz de monodromía y los índices de estabilidad s_i = 0.5*(lambda_i + 1/lambda_i)
    Inputs:
    V_T: Vector estado (x,y,z,vx,vy,vz,Phi_6x6) en el tiempo T.
    T: Periodo de la órbita cerrada.
    Outputs:
    s1, s2: Índices de estabilidad
    """
    Phi_T = reshape(V_T[6:], (6,6))
    eigenvals_Phi_T = eigvals(Phi_T)
    tol = 1e-6
    nontrivial_eig = array([eig for eig in eigenvals_Phi_T if abs(abs(eig)-1)>tol or abs(imag(eig))>tol])
    pair_1 = zeros((2,), dtype=complex)
    pair_2 = zeros((2,), dtype=complex)
    used_indices = set()
    count = 0
    for index, eig in enumerate(nontrivial_eig):
        rec = 1/eig
        if index + 1<len(nontrivial_eig) and index not in used_indices:
            for j in range(index+1, len(nontrivial_eig)):
                if(abs(rec-nontrivial_eig[j])<tol and j not in used_indices):
                    if count == 0:
                        pair_1[0] = eig
                        pair_1[1] = nontrivial_eig[j]
                        used_indices.update([j,index])
                    if count == 1:
                        pair_2[0] = eig
                        pair_2[1] = nontrivial_eig[j]
                        used_indices.update([j,index])
                    count = count + 1

        if count > 1:
            break
 
    s1 = 0.5*(pair_1[0] + pair_1[1])
    s2 = 0.5*(pair_2[0] + pair_2[1])

    return s1, s2

def plot_stability(T, S_1, S_2):
    """
    Hace un plot de los índices de estabilidad s1 y s2 en función del periodo T.
    INPUTS:
    T: Array de periodos de las órbitas.
    S_1: Array de índices de estabilidad.
    S_2: Array de índices de estabilidad.
    """
    s_main = []      # continuous branch
    s_sec  = []      # the other branch

    for k in range(len(T)):
        
        s1 = S_1[k]
        s2 = S_2[k]

        if k == 0:
            s_main.append(s1)
            s_sec.append(s2)
        else:
            if abs(s1.real - s_main[-1].real) < abs(s2.real - s_main[-1].real):
                s_main.append(s1.real)
                s_sec.append(s2.real)
            else:
                s_main.append(s2.real)
                s_sec.append(s1.real)
    plt.plot(T, s_main, marker="o")
    plt.xlabel("T [TU]")
    plt.ylabel("s1")
    plt.show()
    plt.plot(T, s_sec, marker="o")
    plt.xlabel("T [TU]")
    plt.ylabel("s2")
    plt.show()
    return
    
mu_E_M = 0.0121505856 #Earth-Moon
mu_S_E = 3.054200000000000E-6 #Sun-Earth
mu = mu_S_E
#Encontrar y plotear puntos de Lagrange para un mu determinado
lagrange = Lagrange_points_position(mu)

plt.scatter(-mu,0, color="black")
plt.scatter(1-mu,0, color="black")
for i in range(0,5):
    plt.scatter(lagrange[i,0], lagrange[i,1])
plt.show()

U0_L1_x = 8.3690888734309465E-1+1e-7
U0_L1 = array([8.3690888734309465E-1+1e-7,	0,	0,	0,	5e-4,	0]) #Esta es buena!!!

V0_L1 = build_initial_condition(U0_L1)

perform_continuation = False
N_family = 5

delta_x0 = 2e-5
lagrange_point_index = 3


if perform_continuation == True:
    with open(filename, 'w') as f:
        pass  # This empties the file

    U0_group = cargar_CIs_principales(filename_CIs_principales)
    V0_Lyap_family = zeros((U0_group.shape[0],N_family,6*6+6))
    Lyap_period_family = zeros((U0_group.shape[0],N_family))
    for index, U0 in enumerate(U0_group):
        V0 = build_initial_condition(U0)
        T_half0 = estimate_T_half_0(mu, lagrange_point_index)
        print(f"T_half0: {T_half0}")
        V0_Lyap_family[index], Lyap_period_family[index] = Lyapunov_family(V0, mu, RK4, N_family, delta_x0, T_half0, filename)
    print(f"U0_group.shape[0]: {U0_group.shape[0]}")    
    plot_Lyapunov_family(mu, RK4, U0_group.shape[0], N_family, V0_Lyap_family, Lyap_period_family)

else:
    # Coge CIs de un file, calcula la estabilidad y la plotea
    # TODO: Arreglar el plotting de esta parte...
    periods, V0_Lyap_family = cargar_CIs(filename)
    print(f"CIs de archivo: {V0_Lyap_family}")
    S1 = []
    S2 = []
    for i in range(0, V0_Lyap_family.shape[0]):
        sol_Lyap = solve_ivp(CRTBP_variacional_JPL, t_span=(0, periods[i]), y0=V0_Lyap_family[i], method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12, dense_output=True)
        V_T = sol_Lyap.sol(periods[i])
        s1, s2 = stability(V_T)
        S1.append(s1)
        S2.append(s2)
    print(f"S1: {S1}")
    print(f"S2: {S2}")
    plot_Lyapunov_family_one_source(mu, RK4, V0_Lyap_family.shape[0], V0_Lyap_family, periods)
    plot_stability(periods, S1, S2)
