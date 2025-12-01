from Cauchy import Cauchy_problem, Cauchy_problem_intersect_y
from temporal_schemes import RK4, RK45, RK547M, RK56, RK658M, RK78, RK8713M, GBS
from ordinary_differential_equations import CRTBP_variacional_JPL
from numpy import linspace, zeros, identity, reshape, array, sqrt, concatenate, pi, abs
from numpy.linalg import solve, eigvals, norm, lstsq
import matplotlib.pyplot as plt


def build_initial_condition(U0):

    V0 = zeros(6*6+6)
    V0[0:6] = U0
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

    l4 = array([mu-0.5, sqrt(3)/2])
    l5 = array([mu-0.5, -sqrt(3)/2])

    return array([array([l1x+mu-1,0]), array([-l2x+mu-1,0]), array([l3x+mu,0]), l4, l5])

def find_Lyapunov_orbit(V0, mu, T0, temporal_scheme, dt=1e-3, **kwargs):
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

    epsilon = 1e-8
    N_max = 10
    n_iter = 0
    converged = False
    T_half = T0
    plot_steps = False
    
    # Ajustar tiempo máximo según período estimado
    T_max = 2.0 * T0
    t = linspace(0, T_max, int(T_max/dt) + 1)
    
    while n_iter < N_max:
        # Integrar hasta cruce con y=0 usando esquema personalizado
        V_traj, n_intersect = Cauchy_problem_intersect_y(CRTBP_variacional_JPL, V0, mu, t, temporal_scheme, **kwargs)
        
        # Verificar si encontró cruce
        if n_intersect <= 1:
            print("Advertencia: No se encontró cruce con y=0. Aumentando tiempo máximo.")
            T_max *= 1.5
            t = linspace(0, T_max, int(T_max/dt) + 1)
            continue
            
        # Interpolar en y=0
        y0 = V_traj[n_intersect-1, 1]
        y1 = V_traj[n_intersect, 1]
        alpha = abs(y0) / (abs(y0) + abs(y1))
        V_cross = V_traj[n_intersect-1,:] + alpha * (V_traj[n_intersect,:] - V_traj[n_intersect-1,:])
        T_half = t[n_intersect-1] + alpha * (t[n_intersect] - t[n_intersect-1])
        
        # Visualización intermedia (opcional)
        if plot_steps:
            plt.figure(figsize=(10, 8))
            plt.plot(V_traj[:n_intersect+1, 0], V_traj[:n_intersect+1, 1], 'b-', linewidth=1.5)
            plt.plot([V0[0], V_cross[0]], [V0[1], V_cross[1]], 'ro', markersize=8)
            plt.axhline(0, color='k', linestyle='--', alpha=0.3)
            plt.grid(True)
            plt.axis('equal')
            plt.title(f'Iteración {n_iter+1}: vx={V_cross[3]:.2e}, vz={V_cross[5]:.2e}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        # Calcular Jacobiano para corrección
        d1 = sqrt((V_cross[0]+mu)**2 + V_cross[1]**2 + V_cross[2]**2)
        d2 = sqrt((V_cross[0]-1+mu)**2 + V_cross[1]**2 + V_cross[2]**2)
        Phi = reshape(V_cross[6:], (6, 6))
        
        ddotx = V_cross[0] + 2*V_cross[4] - (1-mu)*(V_cross[0]+mu)/d1**3 - mu*(V_cross[0]-1+mu)/d2**3
        ddotz = -((1-mu)/d1**3 + mu/d2**3) * V_cross[2]
        
        Jacobian = zeros((2, 2))
        Jacobian[0, 0] = Phi[3, 2] - Phi[1, 2] * (ddotx / V_cross[4])
        Jacobian[0, 1] = Phi[3, 4] - Phi[1, 4] * (ddotx / V_cross[4])
        Jacobian[1, 0] = Phi[5, 2] - Phi[1, 2] * (ddotz / V_cross[4])
        Jacobian[1, 1] = Phi[5, 4] - Phi[1, 4] * (ddotz / V_cross[4])
        
        # Resolver sistema para corrección
        minus_F = array([-V_cross[3], -V_cross[5]])
        try:
            delta = solve(Jacobian, minus_F)
        except:
            print("Error: Jacobiano singular. Intentando con mínimos cuadrados.")
            delta = lstsq(Jacobian, minus_F, rcond=None)[0]
        
        # Aplicar corrección con factor de amortiguamiento
        damping = 0.7
        V0[2] = V0[2] + damping * delta[0]  # Corrección en z
        V0[4] = V0[4] + damping * delta[1]  # Corrección en vy
        
        n_iter += 1
        print(f"Iter {n_iter:2d}: |vx|={abs(V_cross[3]):.2e}, |vz|={abs(V_cross[5]):.2e}, T_half={T_half:.4f}")

        # Verificar convergencia
        if abs(V_cross[3]) < epsilon and abs(V_cross[5]) < epsilon:
            converged = True
            break
    
    return converged, V0, 2*T_half

def analyze_stability(V0, mu, period, temporal_scheme, dt=1e-3):
    """Analiza estabilidad de la órbita usando monodromía"""
    # Integrar un período completo
    t = linspace(0, period, int(period/dt) + 1)
    V_final = Cauchy_problem(CRTBP_variacional_JPL, V0, mu, t, temporal_scheme)[-1]
    
    # Extraer matriz de monodromía
    Phi_T = reshape(V_final[6:], (6, 6))
    
    # Calcular autovalores
    eigenvals = eigvals(Phi_T)
    print("\nAutovalores de la matriz de monodromía:")
    for ev in eigenvals:
        print(f"{ev:.6f} (magnitud: {abs(ev):.6f})")
    
    # Determinar estabilidad
    unstable = any(abs(ev) > 1.01 for ev in eigenvals)
    print(f"\nEstabilidad: {'INESTABLE' if unstable else 'ESTABLE'}")
    print(f"Magnitud máxima de autovalores: {max(abs(ev) for ev in eigenvals):.6f}")
    
    return Phi_T, eigenvals

# ----------------- EJECUCIÓN PRINCIPAL -----------------
if __name__ == "__main__":
    mu = 0.0121505856  # Earth-Moon
    
    # 1. Calcular y graficar puntos de Lagrange
    lagrange_points = Lagrange_points_position(mu)
    print("Posiciones de puntos de Lagrange:")
    for i, L in enumerate(lagrange_points, 1):
        print(f"L{i}: ({L[0]:.6f}, {L[1]:.6f})")
    
    plt.figure(figsize=(10, 8))
    plt.scatter([-1+mu, mu], [0, 0], s=200, c=['blue', 'gray'], label=['Tierra', 'Luna'])
    labels = ['L1', 'L2', 'L3', 'L4', 'L5']
    for i, L in enumerate(lagrange_points):
        plt.scatter(L[0], L[1], s=100, marker='x', label=labels[i])
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title('Puntos de Lagrange - Sistema Tierra-Luna')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 2. Configurar condición inicial cerca de L1
    L1 = lagrange_points[0]
    xpert = 1e-4  # Perturbación inicial en x
    U0_L1 = array([L1[0] + xpert, 0, 0, 0, 0, 0])
    V0_L1 = build_initial_condition(U0_L1)
    
    # Estimación inicial del semiperiodo
    d1 = sqrt((L1[0]+mu)**2 + L1[1]**2)
    d2 = sqrt((L1[0]-1+mu)**2 + L1[1]**2)
    Uxx = 1 + 3*(1-mu)*(L1[0]+mu)**2/d1**5 - (1-mu)/d1**3 + 3*mu*(L1[0]-1+mu)**2/d2**5 - mu/d2**3
    Uyy = 1 - (1-mu)*(d1**2-3*L1[1]**2)/d1**5 - mu*(d2**2-3*L1[1]**2)/d2**5
    s = sqrt(0.5*(Uxx + Uyy) + sqrt(0.25*(Uxx - Uyy)**2 + 1))
    T_half0 = pi / s
    print(f"\nEstimación inicial: s={s:.6f}, T_half0={T_half0:.6f}")

    # 3. Buscar órbita de Lyapunov con diferentes esquemas
    schemes = [        
        # ("RK8(7)13M", RK8713M, {}),  
        ("GBS", GBS, {})      
    ]
    
    for name, scheme, kwargs in schemes:
        print(f"\n{'='*50}")
        print(f"BUSCANDO ÓRBITA CON {name.upper()}")
        print(f"{'='*50}")
        
        converged, V0_lyap, period = find_Lyapunov_orbit(V0_L1.copy(), mu, T_half0, scheme, dt=1e-3, **kwargs)
        
        if converged:
            print(f"\n¡Órbita encontrada con {name}!")
            print(f"Período: {period:.6f}")
            print(f"Condición inicial corregida: {V0_lyap[0:6]}")
            
            # Graficar órbita completa
            t_orbit = linspace(0, period, int(period/1e-3) + 1)
            V_orbit = Cauchy_problem(CRTBP_variacional_JPL, V0_lyap, mu, t_orbit, scheme, **kwargs)
            
            plt.figure(figsize=(10, 8))
            plt.plot(V_orbit[:,0], V_orbit[:,1], 'b-', linewidth=2, label=f'Órbita ({name})')
            plt.scatter(V_orbit[0,0], V_orbit[0,1], s=100, c='g', marker='o', label='Inicio')
            plt.scatter(V_orbit[-1,0], V_orbit[-1,1], s=100, c='r', marker='x', label='Fin (1 período)')
            plt.scatter([-1+mu, mu], [0, 0], s=200, c=['blue', 'gray'], label=['Tierra', 'Luna'])
            for i, L in enumerate(lagrange_points):
                plt.scatter(L[0], L[1], s=80, marker='x', label=f'L{i+1}')
            plt.axhline(0, color='k', linestyle='--', alpha=0.3)
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.title(f'Órbita de Lyapunov alrededor de L1 - {name}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
            
            # 4. Analizar estabilidad
            print(f"\nAnalizando estabilidad con {name}...")
            Phi_T, eigs = analyze_stability(V0_lyap, mu, period, scheme, dt=1e-3)
            
            # Guardar resultados para comparación
            if name == "RK4":
                V0_ref = V0_lyap.copy()
                period_ref = period
        else:
            print(f"No convergió con {name}.")
    
    # 5. Comparar resultados entre esquemas (opcional)
    if 'V0_ref' in locals():
        print("\n" + "="*60)
        print("COMPARACIÓN DE RESULTADOS ENTRE ESQUEMAS")
        print("="*60)
        print(f"RK4 (referencia):")
        print(f"  Condición inicial: {V0_ref[0:6]}")
        print(f"  Período: {period_ref:.10f}")
        
        for name, scheme, kwargs in schemes[1:]:
            if name in locals():
                V0_comp = locals()[name]
                period_comp = locals()[f"{name}_period"]
                diff = norm(V0_comp[0:6] - V0_ref[0:6])
                period_diff = abs(period_comp - period_ref)
                print(f"\n{name}:")
                print(f"  Diferencia en CI: {diff:.2e}")
                print(f"  Diferencia en período: {period_diff:.2e}")