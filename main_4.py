from Cauchy import Cauchy_problem
from temporal_schemes import RK4, RK45, RK547M, RK56, RK658M, RK78, RK8713M, GBS
from ordinary_differential_equations import CRTBP_variacional_JPL
from numpy import (
    zeros, identity, reshape, array, sqrt, imag, pi, where, savetxt,
    any, hstack, loadtxt, arange, linspace
)
from numpy.linalg import solve, eigvals
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def build_initial_condition(U0):
    V0 = zeros(6*6+6)
    V0[0:6] = U0[0:6]
    Id = identity(6)
    V0[6:] = reshape(Id, (6*6), copy=True)
    return V0


def Lagrange_points_position(mu):
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
    lagrange = Lagrange_points_position(mu)
    l_point_vector = array([lagrange[l_point_index-1,0], lagrange[l_point_index-1,1], 0, 0,0,0])
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
    epsilon = 1e-10
    N_max = 800000
    n_iter = 0
    converged = False
    T_half = T0
    plot_intermediate_steps = False

    while n_iter < N_max:
        if isinstance(temporal_scheme, str):
            sol = solve_ivp(
                CRTBP_variacional_JPL, 
                t_span=(0, 1.25*T_half), 
                y0=V0, 
                method=temporal_scheme, 
                args=(mu,), 
                rtol=kwargs.get("rtol", 1e-10),
                atol=kwargs.get("atol", 1e-12)
            )
            for i in range(1, sol.y.shape[1]):
                if i>0 and sol.y[1, i] * sol.y[1, i-1] < 0:
                    y0 = sol.y[1, i-1]
                    y1 = sol.y[1, i]
                    alpha = y0 / (y0 - y1)
                    V_cross = sol.y[:, i-1] + alpha * (sol.y[:, i] - sol.y[:, i-1])
                    T_half = sol.t[i-1] + alpha * (sol.t[i] - sol.t[i-1])
                    break
            if plot_intermediate_steps:
                plt.plot(sol.y[0], sol.y[1], marker="o")
                plt.scatter(V_cross[0], V_cross[1], color="green", s=100)
                plt.scatter(V0[0], V0[1], color="yellow", s=100)
                plt.axhline(0, color='black', linewidth=1) 
                plt.axis("equal")
                plt.show()
        elif callable(temporal_scheme):            
            Nt = 1000
            t_span = linspace(0, 1.25*T_half, Nt)
            U = Cauchy_problem(CRTBP_variacional_JPL, V0, mu, t_span, temporal_scheme, **kwargs)
            
            for i in range(1, Nt):
                if i>0 and U[i, 1] * U[i-1, 1] < 0:
                    y0 = U[i-1, 1]
                    y1 = U[i, 1]
                    alpha = y0 / (y0 - y1)
                    V_cross = U[i-1, :] + alpha * (U[i, :] - U[i-1, :])
                    T_half = t_span[i-1] + alpha * (t_span[i] - t_span[i-1])
                    break
            if plot_intermediate_steps:
                plt.plot(U[:, 0], U[:, 1], marker="o")
                plt.scatter(V_cross[0], V_cross[1], color="green", s=100)
                plt.scatter(V0[0], V0[1], color="yellow", s=100)
                plt.axhline(0, color='black', linewidth=1) 
                plt.axis("equal")
                plt.show()        
        else:
            raise ValueError("temporal_scheme debe ser str o callable.")

        d1 = sqrt((V_cross[0] + mu)**2 + V_cross[1]**2 + V_cross[2]**2)
        d2 = sqrt((V_cross[0] - 1 + mu)**2 + V_cross[1]**2 + V_cross[2]**2)
        Phi = reshape(V_cross[6:], (6,6))
        ddotx = V_cross[0] + 2*V_cross[4]-(1-mu)*(V_cross[0]+mu)/d1**3-mu*(V_cross[0]-1+mu)/d2**3
        ddotz = -((1-mu)/d1**3 + mu/d2**3)*V_cross[2]
        Jacobian = zeros((2,2))
        Jacobian[0,0] = Phi[3,2]-Phi[1,2]*(ddotx/V_cross[4])
        Jacobian[0,1] = Phi[3,4]-Phi[1,4]*(ddotx/V_cross[4])
        Jacobian[1,0] = Phi[5,2]-Phi[1,2]*(ddotz/V_cross[4])
        Jacobian[1,1] = Phi[5,4]-Phi[1,4]*(ddotz/V_cross[4])
        minus_F = array([-V_cross[3], -V_cross[5]])
        delta_z_doty = solve(Jacobian, minus_F)
        V0[4] += delta_z_doty[1]
        V0[2] += delta_z_doty[0]
        n_iter += 1
        print(f"n_iter: {n_iter}. vx(T): {V_cross[3]:.4e}\r", end="\r", flush=True)
        if abs(V_cross[3]) < epsilon:
            converged = True
            print(f"\nT_half: {T_half}")
            return converged, V0, 2*T_half
    return converged, V0, 2*T_half


def cargar_CIs_principales(filename):
    ics_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            numbers = [float(n) for n in line.split(',')]
            numbers[1] = abs(numbers[1])
            ics_list.append(numbers)
    return array(ics_list)


def guardar_CIs(V0_Lyap_family, Lyap_p_fam, filename):
    found = where(any(V0_Lyap_family, axis=1))[0]
    V_save = V0_Lyap_family[found]
    T_save = Lyap_p_fam[found]
    data = hstack([T_save.reshape(-1,1), V_save])
    with open(filename, 'w') as f:  # 'w' para sobrescribir (mejor que 'a')
        savetxt(f, data, delimiter=",", fmt="%.12e")


def cargar_CIs(filename):
    data = loadtxt(filename, delimiter=",")
    periods = data[:,0]
    V0 = data[:,1:]
    return periods, V0


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
    V0_fam = zeros((N_family, 42))
    T_fam = zeros(N_family)
    for i in range(N_family):
        print(f"Orbita {i+1}/{N_family}")
        V0_IN = V0.copy()
        V0_IN[0] += i * delta_x0
        converged, V_opt, T_opt = find_Lyapunov_orbit(V0_IN, mu, Th0, temporal_scheme, **kwargs)
        if converged:
            V0_fam[i] = V_opt
            T_fam[i] = T_opt
    guardar_CIs(V0_fam, T_fam, filename)
    return V0_fam, T_fam


def plot_Lyapunov_family(mu, temporal_scheme, N_CI_pp, N_family, V0_Lyap_family, Lyap_period_family):
    for i in range(0, N_CI_pp):
        for j in range(0, N_family): 
            if isinstance(temporal_scheme, str):
                sol_Lyap = solve_ivp(CRTBP_variacional_JPL, t_span=(0, Lyap_period_family[i,j]), y0=V0_Lyap_family[i,j], method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12)
                plt.plot(sol_Lyap.y[0,:], sol_Lyap.y[1,:], marker="o")
            else:
                t_eval = linspace(0, Lyap_period_family[i,j], 1000)
                U = Cauchy_problem(CRTBP_variacional_JPL, V0_Lyap_family[i,j], mu, t_eval, temporal_scheme, **kwargs)                  
                plt.plot(U[:, 0], U[:, 1], marker="o")
    plt.axis("equal")
    plt.show()

    return


def plot_Lyapunov_family_one_source(mu, temporal_scheme, V0_Lyap_family, periods, **kwargs):
    plt.figure(figsize=(8,6))
    for j in range(len(periods)):
        if not any(V0_Lyap_family[j]): continue
        T = periods[j]
        if isinstance(temporal_scheme, str):
            sol = solve_ivp(
                CRTBP_variacional_JPL, (0, T), V0_Lyap_family[j],
                method=temporal_scheme, args=(mu,),
                rtol=kwargs.get("rtol", 1e-10),
                atol=kwargs.get("atol", 1e-12)
            )
            x, y = sol.y[0], sol.y[1]
        else:        
            t_eval = linspace(0, T, 10000)            
            U = Cauchy_problem(CRTBP_variacional_JPL, V0_Lyap_family[j], mu, t_eval, temporal_scheme, **kwargs)
            x, y = U[:,0], U[:,1]
        plt.plot(x, y, lw=0.8)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Familia de Lyapunov")
    plt.grid(True, alpha=0.3)
    plt.show()


def stability(V_T):
    Phi_T = reshape(V_T[6:], (6,6))
    eigs = eigvals(Phi_T)
    tol = 1e-6
    nontrivial = [eig for eig in eigs if abs(abs(eig)-1) > tol or abs(imag(eig)) > tol]
    pairs = []
    used = set()
    for i, lam in enumerate(nontrivial):
        if i in used: continue
        for j in range(i+1, len(nontrivial)):
            if j in used: continue
            if abs(lam * nontrivial[j] - 1) < tol:
                pairs.append((lam, nontrivial[j]))
                used.update([i,j])
                break
    if len(pairs) < 2:
        return 0+0j, 0+0j
    s1 = 0.5 * (pairs[0][0] + pairs[0][1])
    s2 = 0.5 * (pairs[1][0] + pairs[1][1])
    return s1, s2


def plot_stability(T, S1, S2):    
    s_main, s_sec = [], []
    for k in range(len(T)):
        if k == 0:
            s_main.append(S1[k]); s_sec.append(S2[k])
        else:
            if abs(S1[k].real - s_main[-1].real) < abs(S2[k].real - s_main[-1].real):
                s_main.append(S1[k]); s_sec.append(S2[k])
            else:
                s_main.append(S2[k]); s_sec.append(S1[k])
    plt.figure(figsize=(10,4))
    plt.subplot(121); plt.plot(T, [s.real for s in s_main], 'o-'); plt.ylabel("Re(s₁)"); plt.grid()
    plt.subplot(122); plt.plot(T, [s.real for s in s_sec], 'o-', color='orange'); plt.ylabel("Re(s₂)"); plt.grid()
    plt.tight_layout()
    plt.show()


# ==================== CONFIGURACIÓN PRINCIPAL ====================
mu_E_M = 0.0121505856
mu_S_E = 3.0542e-6
mu = mu_S_E

# Plot Lagrange points
lagrange = Lagrange_points_position(mu)
plt.figure(figsize=(6,4))
plt.scatter([-mu, 1-mu], [0, 0], c='black', s=100, label=['M₁', 'M₂'])
for i, (x,y) in enumerate(lagrange):
    plt.scatter(x, y, s=80, label=f'L{i+1}')
plt.axis('equal'); plt.legend(); plt.grid(True, alpha=0.3); plt.title("Puntos de Lagrange")
plt.show()

# Condiciones iniciales
U0_L1 = array([0.83690888734309465 + 1e-7, 0, 0, 0, 5e-4, 0])
V0_L1 = build_initial_condition(U0_L1)

# ==================== PARÁMETROS DE CONTROL ====================
perform_continuation = False
N_family = 2
delta_x0 = 2e-5
lagrange_point_index = 1  # L1
filename = "Lyap_family_Sun_Earth_L1.txt"
filename_CIs_principales = "L1_Sun_Earth_CI.txt"

# ELIGE AQUÍ TU ESQUEMA 

# PARA UTILIZAR solve_ivp, PONER ENTRE COMILLAS ALGUNO DE ESTOS:
# 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
# Por ejemplo: temporal_scheme = "RK45"

# PARA UTILIZAR Cauchy_problem, PONER ALGUNO DE LOS DE temporal_schemes.py:
temporal_scheme = GBS       

# Parámetros adicionales
kwargs = {}

# ==================== EJECUCIÓN ====================
if perform_continuation:
    print("▶ Generando nueva familia...")
    U0_group = cargar_CIs_principales(filename_CIs_principales)
    T_fam = zeros((U0_group.shape[0], N_family))
    V0_fam = zeros((U0_group.shape[0], N_family, 42))
    for idx, U0 in enumerate(U0_group):
        print(f"\n--- CI #{idx+1} ---")
        V0 = build_initial_condition(U0)
        T_half0 = estimate_T_half_0(mu, lagrange_point_index)
        V0_fam[idx], T_fam[idx] = Lyapunov_family(V0, mu, temporal_scheme, N_family, delta_x0, T_half0, filename, **kwargs)                
    print("✅ Familia generada.")
    plot_Lyapunov_family(mu, temporal_scheme, U0_group.shape[0], N_family, V0_fam, T_fam)

else:
    print("▶ Analizando familia existente...")
    try:
        periods, V0_Lyap_family = cargar_CIs(filename)
        print(f"Cargadas {len(periods)} órbitas.")
    except Exception as e:
        print(f"❌ Error al cargar {filename}: {e}")
        exit()

    # --- CÁLCULO DE ESTABILIDAD (soporta ambos modos) ---
    S1, S2 = [], []
    for i in range(len(periods)):
        if not any(V0_Lyap_family[i]): continue
        T = periods[i]

        if isinstance(temporal_scheme, str):
            sol = solve_ivp(
                CRTBP_variacional_JPL, (0, T), V0_Lyap_family[i],
                method=temporal_scheme, args=(mu,),
                rtol=kwargs.get("rtol", 1e-10),
                atol=kwargs.get("atol", 1e-12),
                dense_output=True
            )
            V_T = sol.sol(T)
        else: 
            Nt = 1000          
            t_eval = linspace(0, T, Nt)            
            U = Cauchy_problem(CRTBP_variacional_JPL, V0_Lyap_family[i], mu, t_eval, temporal_scheme)
            V_T = U[Nt-1]

        s1, s2 = stability(V_T)
        S1.append(s1); S2.append(s2)
        print(f"[{i+1}] T={T:.4f} | s1={s1:.2e} | s2={s2:.2e}")

    # --- VISUALIZACIÓN ---
    plot_Lyapunov_family_one_source(mu, temporal_scheme, V0_Lyap_family, periods, **kwargs)
    plot_stability(periods, S1, S2)

    print("\n✔ Análisis completado.")