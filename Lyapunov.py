from scipy.integrate import solve_ivp
from ordinary_differential_equations import CRTBP_variacional_JPL
import matplotlib.pyplot as plt
from numpy import linspace, sqrt, reshape, array, zeros
from Cauchy import Cauchy_problem
from numpy.linalg import solve

"""
===============================================================================
 Archivo:       Lyapunov.py
 Creado:        20/11/2025
 Descripción:    
 
 Encontrar familias de Lyapunov y órbitas de Lyapunov. 
 
 Dependencias:
    - NumPy
    
 Notas:
===============================================================================
"""

def find_Lyapunov_orbit(V0, mu, T0, temporal_scheme, **kwargs):
    """
    Encuentra una órbita de Lyapunov periódica usando single shooting.
    INPUTS:
    V0:                 Condición inicial en el problema variacional.
    mu:                 Parámetro de masa.
    T0:                 Semi-periodo inicial de la órbita.
    temporal_scheme:    Esquema temporal.
    kwargs:             kwargs para Cauchy problem.
    OUTPUTS:
    converged:          Booleano, True si ha convergido y se ha encontrado una órbita periódica, False en caso contrario.
    V0:                 Condición inicial de la órbita periódica convergida.
    T:                  Periodo de la órbita convergida.
    """
    epsilon = 1e-10
    N_max = 800000
    n_iter = 0
    converged = False
    T_half = T0
    plot_intermediate_steps = False
    V_cross = zeros(42)

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
                #plt.show()
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
    return V0_fam, T_fam