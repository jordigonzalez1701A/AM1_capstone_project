"""
===============================================================================
 Archivo:       plotting.py
 Creado:        02/12/2025
 Descripción:    
 
 Crear representaciones gráficas de la estabilidad y de familias de órbitas de Lyapunov.

 Dependencias:
    - Matplotlib
    - scipy
    - NumPy
    - ordinary_differential_equations
    - Cauchy
    
 Notas:
===============================================================================
"""


import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ordinary_differential_equations import CRTBP_variacional_JPL
from numpy import linspace
from Cauchy import Cauchy_problem

def plot_Lyapunov_family(mu, temporal_scheme, N_CI_pp, N_family, V0_Lyap_family, Lyap_period_family, **kwargs):
    """
    Crea representaciones gráficas de familias de Lyapunov.
    INPUTS:
    mu:                 Masa reducida.
    temporal_scheme:    Esquema temporal.
    N_CI_pp:            Cantidad de condiciones iniciales por familia.
    N_family:           Cantidad de familias.
    V0_Lyap_family:     Tensor de condiciones iniciales (N_CI_pp, N_family, 42) de las familias en el problema variacional.
    Lyap_period_family: Tansor de periodos (N_CI_pp, N_family, 1) de las familias de Lyapunov.
    kwargs:             kwargs para el Cauchy problem.
    OUTPUTS:
    Ninguno.
    """
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
    """
    Crea representaciones gráficas de familias de Lyapunov si estas provienen de una sola condición inicial principal (es decir, las que generan familias).
    INPUTS:
    mu:                 Masa reducida.
    temporal_scheme:    Esquema temporal.
    V0_Lyap_family:     Condiciones iniciales de las órbitas de la familia de Lyapunov en el problema variacional.
    periods:            Array de periodos de las órbitas de la familia de Lyapunov.
    kwargs:             Kwargs del Cauchy_problem.
    OUTPUTS:
    Ninguno.                
    """
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


def plot_stability(T, S1, S2):   
    """
    Crea una representación gráfica de la estabilidad de una familia de órbitas de Lyapunov, Re(s_i) vs T.
    INPUTS:
    T:      Periodos en unidades adimensionales.
    S1:     Array con los índices de estabilidad s1.
    S2:     Array con los índices de estabilidad s2.
    OUTPUTS:
    Ninguno.
    """ 
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

