# -*- coding: utf-8 -*-
"""
Análisis de estabilidad de órbitas de Lyapunov en el CRTBP
usando esquemas temporales propios (RK4, RK45, DOPRI5, GBS, etc.)
"""

# === IMPORTS ===
from numpy import (
    zeros, identity, reshape, array, sqrt, imag, pi, where, savetxt, any,
    hstack, loadtxt, linspace, arange, abs, dot, real, copy
)
from numpy.linalg import solve, eigvals, norm
import matplotlib.pyplot as plt

# === IMPORTS LOCALES (asumimos que están en el mismo directorio) ===
# from Cauchy import Cauchy_problem, Cauchy_problem_intersect_y  # ya no usamos intersect_y
from temporal_schemes import (
    RK4, RK45, RK547M, RK56, RK658M, RK78, RK8713M, GBS, ERK, RK_stages
)
from ordinary_differential_equations import CRTBP_variacional, CRTBP_variacional_JPL
from Cauchy import Cauchy_problem

# Si no puedes importarlos, define aquí las funciones clave (ver abajo)


# === FUNCIONES AUXILIARES ===

def build_initial_condition(U0):
    V0 = zeros(42)
    V0[:6] = U0
    V0[6:] = identity(6).ravel()
    return V0


def Lagrange_points_position(mu):
    def l1_poly(gamma):
        return gamma**5 + (mu-3)*gamma**4 + (3-2*mu)*gamma**3 - mu*gamma**2 + 2*mu*gamma - mu
    def l2_poly(gamma):
        return gamma**5 + (3-mu)*gamma**4 + (3-2*mu)*gamma**3 - mu*gamma**2 - 2*mu*gamma - mu
    def l3_poly(gamma):
        return gamma**5 + (2+mu)*gamma**4 + (1+2*mu)*gamma**3 - (1-mu)*gamma**2 - 2*(1-mu)*gamma - (1-mu)
    
    def newton(func, gamma0, tol=1e-12, max_iter=1000):
        gamma = gamma0
        delta = 1e-12
        for _ in range(max_iter):
            f = func(gamma)
            f_prime = (func(gamma + delta) - func(gamma - delta)) / (2 * delta)
            if abs(f_prime) < 1e-15:
                break
            gamma_new = gamma - f / f_prime
            if abs(gamma_new - gamma) < tol:
                return gamma_new
            gamma = gamma_new
        return gamma

    l1x = newton(l1_poly, (mu/3)**(1/3))
    l2x = newton(l2_poly, 1 - (7/12)*mu)
    l3x = newton(l3_poly, -1 + (1/3)*mu)
    l4 = array([-mu + 0.5, sqrt(3)/2, 0, 0, 0, 0])
    l5 = array([-mu + 0.5, -sqrt(3)/2, 0, 0, 0, 0])
    L1 = array([-(l1x + mu - 1), 0, 0, 0, 0, 0])
    L2 = array([-(-l2x + mu - 1), 0, 0, 0, 0, 0])
    L3 = array([-(l3x + mu), 0, 0, 0, 0, 0])
    return array([L1[:2], L2[:2], L3[:2], l4[:2], l5[:2]])


def estimate_T_half_0(mu, l_point_index):
    lagrange = Lagrange_points_position(mu)
    L = array([lagrange[l_point_index-1,0], lagrange[l_point_index-1,1], 0, 0, 0, 0])
    
    d1 = sqrt((L[0]+mu)**2 + L[1]**2)
    d2 = sqrt((L[0]-1+mu)**2 + L[1]**2)
    Uxx = 1 + 3*(1-mu)*(L[0]+mu)**2/d1**5 - (1-mu)/d1**3 + 3*mu*(L[0]-1+mu)**2/d2**5 - mu/d2**3
    Uyy = 1 - (1-mu)*(d1**2 - 3*L[1]**2)/d1**5 - mu*(d2**2 - 3*L[1]**2)/d2**5
    beta1 = 2 - 0.5*(Uxx + Uyy)
    beta2_sq = -Uxx * Uyy
    s = sqrt(beta1 + sqrt(beta1**2 + beta2_sq))
    return pi / s


def find_crossing_y(U_traj, t_eval):
    """
    Busca primer cruce con y=0 en la trayectoria (interpolación lineal).
    Retorna (V_cross, t_cross) o (None, None) si no hay cruce.
    """
    y = U_traj[:, 1]
    for i in range(1, len(y)):
        if y[i-1] * y[i] < 0:
            alpha = y[i-1] / (y[i-1] - y[i])
            V_cross = U_traj[i-1] + alpha * (U_traj[i] - U_traj[i-1])
            t_cross = t_eval[i-1] + alpha * (t_eval[i] - t_eval[i-1])
            return V_cross, t_cross
    return None, None


def find_Lyapunov_orbit(V0, mu, T0, temporal_scheme, dt=None, **kwargs):
    epsilon = 1e-9
    N_max = 500  # reducido: cada iteración es costosa
    T_half = T0
    V0 = copy(V0)
    
    for n_iter in range(1, N_max + 1):
        T_end = 1.25 * T_half
        
        # Generar malla temporal
        if dt is None:
            dt_auto = min(1e-3, T0 / 200)
            N_steps = max(1000, int(T_end / dt_auto))
            t_eval = linspace(0, T_end, N_steps + 1)
        else:
            t_eval = arange(0, T_end + dt, dt)
            if t_eval[-1] < T_end:
                t_eval = hstack([t_eval, T_end])

        # Integrar
        try:
            U = Cauchy_problem(CRTBP_variacional_JPL, V0, mu, t_eval, temporal_scheme, **kwargs)
        except Exception as e:
            print(f"❌ Error en integración (iter {n_iter}): {e}")
            return False, V0, 2 * T_half
        
        # Encontrar cruce
        V_cross, t_cross = find_crossing_y(U, t_eval)
        if V_cross is None:
            print(f"❌ No crossing found (iter {n_iter})")
            return False, V0, 2 * T_half
        
        T_half = t_cross
        vx_T = V_cross[3]
        print(f"Iter {n_iter:3d} | T_half = {T_half:8.5f} | vx(T) = {vx_T: .3e}", end="\r")

        # Corrección (solo si y ≈ 0 — ya garantizado por cruce)
        if abs(V_cross[1]) > 1e-10:
            continue  # seguridad adicional

        # Matriz de estado en cruce
        Phi = V_cross[6:].reshape(6, 6)
        
        # Distancias
        d1 = sqrt((V_cross[0]+mu)**2 + V_cross[1]**2 + V_cross[2]**2)
        d2 = sqrt((V_cross[0]-1+mu)**2 + V_cross[1]**2 + V_cross[2]**2)
        
        # Aceleraciones
        ddotx = V_cross[0] + 2*V_cross[4] - (1-mu)*(V_cross[0]+mu)/d1**3 - mu*(V_cross[0]-1+mu)/d2**3
        ddotz = -((1-mu)/d1**3 + mu/d2**3) * V_cross[2]
        
        # Jacobiano 2×2
        J = zeros((2,2))
        J[0,0] = Phi[3,2] - Phi[1,2] * (ddotx / V_cross[4])
        J[0,1] = Phi[3,4] - Phi[1,4] * (ddotx / V_cross[4])
        J[1,0] = Phi[5,2] - Phi[1,2] * (ddotz / V_cross[4])
        J[1,1] = Phi[5,4] - Phi[1,4] * (ddotz / V_cross[4])
        
        # Resolver Δ = -J⁻¹·[vx; vz]
        try:
            delta = solve(J, array([-V_cross[3], -V_cross[5]]))
        except Exception:
            print(f"\n❌ Jacobiano singular en iter {n_iter}")
            return False, V0, 2 * T_half

        V0[2] += delta[0]   # z ← z + Δz
        V0[4] += delta[1]   # vy ← vy + Δvy

        if abs(vx_T) < epsilon:
            print(f"\n✅ Convergencia alcanzada en {n_iter} iteraciones. T = {2*T_half:.6f}")
            return True, V0, 2 * T_half

    print(f"\n❌ No convergió tras {N_max} iteraciones.")
    return False, V0, 2 * T_half


def cargar_CIs_principales(filename):
    ics_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            numbers = [float(x) for x in line.split(',')]
            numbers[1] = abs(numbers[1])  # simetría
            ics_list.append(numbers)
    return array(ics_list)


def guardar_CIs(V0_Lyap_family, Lyap_p_fam, filename):
    found = any(V0_Lyap_family, axis=1)
    V_save = V0_Lyap_family[found]
    T_save = Lyap_p_fam[found]
    data = hstack([T_save.reshape(-1,1), V_save])
    savetxt(filename, data, delimiter=",", fmt="%.12e")


def cargar_CIs(filename):
    data = loadtxt(filename, delimiter=",")
    periods = data[:,0]
    V0 = data[:,1:]
    return periods, V0


def Lyapunov_family(V0, mu, temporal_scheme, N_family, dx0, Th0, filename, dt=None, **kwargs):
    V0_family = zeros((N_family, 42))
    T_family = zeros(N_family)
    
    for i in range(N_family):
        print(f"\nOrbita {i+1}/{N_family}")
        delta = zeros(42)
        delta[0] = i * dx0
        V0_pert = V0 + delta
        
        converged, V_opt, T_opt = find_Lyapunov_orbit(
            V0_pert, mu, Th0, temporal_scheme, dt=dt, **kwargs
        )
        
        if converged:
            V0_family[i] = V_opt
            T_family[i] = T_opt
            print(f"  → T = {T_opt:.6f}")
        else:
            print("  → No convergió")

    guardar_CIs(V0_family, T_family, filename)
    return V0_family, T_family


def plot_Lyapunov_family_one_source(mu, temporal_scheme, V0_Lyap_family, Lyap_period_family, dt=None, **kwargs):
    plt.figure(figsize=(8,6))
    for j in range(len(Lyap_period_family)):
        if not any(V0_Lyap_family[j]):
            continue
        T = Lyap_period_family[j]
        if dt is None:
            N_steps = max(1000, int(T * 200))
            t_eval = linspace(0, T, N_steps + 1)
        else:
            t_eval = arange(0, T + dt, dt)
            if t_eval[-1] < T:
                t_eval = hstack([t_eval, T])

        U = Cauchy_problem(CRTBP_variacional_JPL, V0_Lyap_family[j], mu, t_eval, temporal_scheme, **kwargs)
        plt.plot(U[:,0], U[:,1], lw=0.8, alpha=0.7)

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Familia de órbitas de Lyapunov")
    plt.grid(True, alpha=0.3)
    plt.show()


def stability(V_T):
    Phi_T = V_T[6:].reshape(6,6)
    eigs = eigvals(Phi_T)
    # Filtrar pares (λ, 1/λ) no triviales (|λ| ≠ 1 o Im(λ) ≠ 0)
    nontrivial = []
    used = set()
    tol = 1e-6
    
    for i, lam in enumerate(eigs):
        if i in used:
            continue
        found = False
        for j in range(i+1, len(eigs)):
            if j in used:
                continue
            if abs(lam * eigs[j] - 1) < tol:
                nontrivial.append((lam, eigs[j]))
                used.update([i, j])
                found = True
                break
        if not found and (abs(abs(lam) - 1) > tol or abs(imag(lam)) > tol):
            # λ real ≠ ±1 → par consigo mismo (caso marginal)
            nontrivial.append((lam, lam))
            used.add(i)

    if len(nontrivial) < 2:
        return complex(0), complex(0)
    
    s1 = 0.5 * (nontrivial[0][0] + nontrivial[0][1])
    s2 = 0.5 * (nontrivial[1][0] + nontrivial[1][1])
    return s1, s2


def plot_stability(T, S1, S2):
    s_main = []
    s_sec = []
    for k in range(len(T)):
        if k == 0:
            s_main.append(S1[k])
            s_sec.append(S2[k])
        else:
            if abs(real(S1[k]) - real(s_main[-1])) < abs(real(S2[k]) - real(s_main[-1])):
                s_main.append(S1[k])
                s_sec.append(S2[k])
            else:
                s_main.append(S2[k])
                s_sec.append(S1[k])
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(T, [real(s) for s in s_main], 'o-', label="s₁")
    plt.axhline(1, color='r', ls='--', alpha=0.7)
    plt.axhline(-1, color='r', ls='--', alpha=0.7)
    plt.xlabel("Periodo T")
    plt.ylabel("Re(s₁)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(T, [real(s) for s in s_sec], 'o-', color='orange', label="s₂")
    plt.axhline(1, color='r', ls='--', alpha=0.7)
    plt.axhline(-1, color='r', ls='--', alpha=0.7)
    plt.xlabel("Periodo T")
    plt.ylabel("Re(s₂)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# === CONFIGURACIÓN Y EJECUCIÓN ===

if __name__ == "__main__":
    # Parámetros
    mu_S_E = 3.0542e-6
    mu_E_M = 0.0121505856
    mu = mu_S_E  # ← Cambia aquí si deseas Earth-Moon

    # Mostrar puntos de Lagrange
    lagrange = Lagrange_points_position(mu)
    plt.figure(figsize=(6,4))
    plt.scatter([-mu, 1-mu], [0, 0], c='black', s=80, label=['M₁', 'M₂'])
    labels = ['L₁', 'L₂', 'L₃', 'L₄', 'L₅']
    for i, (x, y) in enumerate(lagrange):
        plt.scatter(x, y, s=60, label=labels[i])
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"Puntos de Lagrange (μ = {mu:.2e})")
    plt.show()

    # --- CONFIGURACIÓN DE LA FAMILIA ---
    perform_continuation = False  # ← Cambia a True para generar nueva familia
    filename = "Lyap_family_Sun_Earth_L1.txt"
    filename_CIs_principales = "L1_Sun_Earth_CI.txt"
    lagrange_point_index = 1  # L1
    N_family = 5
    delta_x0 = 2e-5

    # Esquema a usar (elige uno):
    # temporal_scheme = RK4
    # temporal_scheme = RK45
    # temporal_scheme = RK547M  # DOPRI5
    # temporal_scheme = RK78
    # temporal_scheme = GBS
    temporal_scheme = RK45  # ← por defecto
    dt = None  # None → auto; o e.g., dt=1e-3 para fijo

    if perform_continuation:
        print("▶ Generando nueva familia de órbitas...")
        with open(filename, 'w'):
            pass  # vaciar archivo

        U0_group = cargar_CIs_principales(filename_CIs_principales)
        print(f"Cargadas {len(U0_group)} CIs iniciales.")

        for idx, U0 in enumerate(U0_group):
            print(f"\n--- CI inicial #{idx+1} ---")
            V0 = build_initial_condition(U0)
            T_half0 = estimate_T_half_0(mu, lagrange_point_index)
            print(f"T_half estimado: {T_half0:.6f}")
            
            V_fam, T_fam = Lyapunov_family(
                V0, mu, temporal_scheme, N_family, delta_x0,
                T_half0, filename, dt=dt
            )
        print("✅ Familia generada y guardada.")

    else:
        print("▶ Cargando y analizando familia existente...")
        try:
            periods, V0_Lyap_family = cargar_CIs(filename)
            print(f"Cargadas {len(periods)} órbitas.")
        except Exception as e:
            print(f"❌ Error al cargar {filename}: {e}")
            exit()

        # --- ANÁLISIS DE ESTABILIDAD ---
        S1, S2 = [], []
        for i in range(len(periods)):
            if not any(V0_Lyap_family[i]):
                continue
            T = periods[i]
            # Integrar EXACTAMENTE hasta T
            if dt is None:
                N_steps = max(2000, int(T * 300))
                t_eval = linspace(0, T, N_steps + 1)
            else:
                t_eval = arange(0, T + dt, dt)
                if t_eval[-1] < T:
                    t_eval = hstack([t_eval, T])

            U = Cauchy_problem(CRTBP_variacional_JPL, V0_Lyap_family[i], mu, t_eval, temporal_scheme)
            V_T = U[-1]
            s1, s2 = stability(V_T)
            S1.append(s1)
            S2.append(s2)
            print(f"[{i+1}] T = {T:6.3f} | s₁ = {s1: .3e} | s₂ = {s2: .3e}")

        # --- VISUALIZACIÓN ---
        plot_Lyapunov_family_one_source(mu, temporal_scheme, V0_Lyap_family, periods, dt=dt)
        plot_stability(periods, S1, S2)

        # Guardar resultados de estabilidad (opcional)
        stab_data = array([
            periods,
            [real(s) for s in S1],
            [imag(s) for s in S1],
            [real(s) for s in S2],
            [imag(s) for s in S2]
        ]).T
        savetxt("stability_results.csv", stab_data,
                header="T, Re(s1), Im(s1), Re(s2), Im(s2)",
                delimiter=",", comments="")

        print("\n✔ Análisis completado. Resultados guardados.")