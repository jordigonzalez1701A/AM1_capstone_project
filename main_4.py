from Cauchy import Cauchy_problem
from temporal_schemes import RK4, RK45, RK547M, RK56, RK658M, RK78, RK8713M, GBS
from ordinary_differential_equations import CRTBP_variacional_JPL
from numpy import (zeros, any, linspace, array)
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from CRTBP_utils import Lagrange_points_position, build_initial_condition, estimate_T_half_0
from utilities import cargar_CIs_principales, cargar_CIs 
from Lyapunov import Lyapunov_family
from plotting import plot_Lyapunov_family, plot_stability, plot_Lyapunov_family_one_source
from stability import stability

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
    plot_Lyapunov_family(mu, temporal_scheme, U0_group.shape[0], N_family, V0_fam, T_fam, **kwargs)

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