from matplotlib.ticker import *
from CRTBP_utils import *
from Lyapunov import *

def plot_lagrange_points(ax_2d, ax_3d, mu, M1_label, M2_label, M1_color, M2_color, M1_size, M2_size):
        
    lagrange_pts = Lagrange_points_position(mu)
    for collection in ax_2d.collections[:]:
        if hasattr(collection, '_lagrange') and collection._lagrange:
            collection.remove()
    for text in ax_2d.texts[:]:
        if hasattr(text, '_lagrange') and text._lagrange:
            text.remove()
    for collection in ax_3d.collections[:]:
        if hasattr(collection, '_lagrange') and collection._lagrange:
            collection.remove()
    # Posiciones de los primarios
    M1 = (-mu, 0)
    M2 = (1 - mu, 0)
        
    labels = [r"L$_1$",r"L$_2$",r"L$_3$",r"L$_4$",r"L$_5$"]
        
    sc_m1_2d = ax_2d.scatter(-mu, 0, c=M1_color, s=M1_size, label=M1_label, edgecolors='black', linewidth=0.5)
    sc_m2_2d = ax_2d.scatter(1 - mu, 0, c=M2_color, s=M2_size, label=M2_label, edgecolors='black', linewidth=0.5)
    sc_m1_2d._lagrange = True
    sc_m2_2d._lagrange = True
        
    for i, (x,y) in enumerate(lagrange_pts):
        sc = ax_2d.scatter(x, y, s=80, label=labels[i])
        sc._lagrange = True
            
    sc_m1_3d = ax_3d.scatter(-mu, 0, 0, c=M1_color, s=M1_size, label=M1_label, edgecolors='black', linewidth=0.5)
    sc_m2_3d = ax_3d.scatter(1 - mu, 0, 0, c=M2_color, s=M2_size, label=M2_label, edgecolors='black', linewidth=0.5)
    sc_m1_3d._lagrange = True
    sc_m2_3d._lagrange = True

    for i, (x, y) in enumerate(lagrange_pts):
        sc = ax_3d.scatter(x, y, 0, s=80, label=labels[i])
        sc._lagrange = True
        
    # === Restaurar título original y configuración ===
    ax_2d.set_title("Órbita 2D")
    ax_2d.set_xlabel(r"$x$")
    ax_2d.set_ylabel(r"$y$")
    ax_2d.grid(True, alpha=0.3)
    ax_2d.set_aspect('equal')
    ax_2d.legend()

    ax_3d.set_title("Órbita 3D")
    ax_3d.set_xlabel(r"$x$")
    ax_3d.set_ylabel(r"$y$")
    ax_3d.set_zlabel(r"$z$")
    ax_3d.legend()
        
def plot_Lyapunov_family_GUI(ax_2d, ax_3d, mu, temporal_scheme, N_CI_pp, N_family, V0_Lyap_family, Lyap_period_family,
                             lagrange_point_index, **kwargs):
    """
    Dibuja órbitas de Lyapunov en los ejes dados (2D y/o 3D).
    Si no se proporcionan ejes, usa plt.gca() (para compatibilidad).
    """
        
    # Obtener posición del punto de Lagrange actual
    # lagrange_text = self.lagrange_combo.currentText()
    # lagrange_index = int(lagrange_text[1]) - 1
    L_points = Lagrange_points_position(mu)
    Lx, Ly = L_points[lagrange_point_index]  # Coordenadas del punto de Lagrange

    all_x = [Lx]
    all_y = [Ly]
        
    # Resolver cada órbita
        
    for i in range(0, N_CI_pp):
        for j in range(0, N_family):
            if Lyap_period_family[i, j] == 0:
                continue  # Saltar órbitas no convergidas

            if isinstance(temporal_scheme, str):
                sol_Lyap = solve_ivp(CRTBP_variacional_JPL, t_span=(0, Lyap_period_family[i,j]), y0=V0_Lyap_family[i,j], method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12)
                t_plot = linspace(0, Lyap_period_family[i, j], 1000)
                y_plot = sol_Lyap.sol(t_plot)
                x, y, z = y_plot[0], y_plot[1], y_plot[2]
            else:
                t_eval = linspace(0, Lyap_period_family[i, j], 1000)
                U = Cauchy_problem(CRTBP_variacional_JPL, V0_Lyap_family[i,j], mu, t_eval, temporal_scheme, **kwargs) 
                x, y, z = U[:, 0], U[:, 1], 0
                    
            all_x.extend(x)
            all_y.extend(y)
        
    if len(all_x) == 1:  # Solo el punto de Lagrange, nada más
        x_center, y_center = Lx, Ly
        x_range = 0.01  # valor arbitrario pequeño
        y_range = 0.01
    else:
        x_min_data, x_max_data = min(all_x), max(all_x)
        y_min_data, y_max_data = min(all_y), max(all_y)
        x_center = (x_min_data + x_max_data) / 2
        y_center = (y_min_data + y_max_data) / 2
        x_range = x_max_data - x_min_data
        y_range = y_max_data - y_min_data

    # Aplicar margen del 10% extra (es decir, 1.1 del rango total)
    margin = 0.1
    x_span = x_range * (1 + margin)
    y_span = y_range * (1 + margin)
        
    if x_span > y_span:
            
        x_min = x_center - x_span / 2
        x_max = x_center + x_span / 2
        y_min = y_center - x_span / 2
        y_max = y_center + x_span / 2
            
    else:
        
        x_min = x_center - y_span / 2
        x_max = x_center + y_span / 2
        y_min = y_center - y_span / 2
        y_max = y_center + y_span / 2

    # Dibujar cada órbita

    for i in range(0, N_CI_pp):
        for j in range(0, N_family):
            if Lyap_period_family[i, j] == 0:
                continue  # Saltar órbitas no convergidas

            if isinstance(temporal_scheme, str):
                sol_Lyap = solve_ivp(CRTBP_variacional_JPL, t_span=(0, Lyap_period_family[i,j]), y0=V0_Lyap_family[i,j], method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12)
                t_plot = linspace(0, Lyap_period_family[i, j], 1000)
                y_plot = sol_Lyap.sol(t_plot)
                x, y, z = y_plot[0], y_plot[1], y_plot[2]
            else:
                t_eval = linspace(0, Lyap_period_family[i, j], 1000)
                U = Cauchy_problem(CRTBP_variacional_JPL, V0_Lyap_family[i,j], mu, t_eval, temporal_scheme, **kwargs) 
                x, y, z = U[:, 0], U[:, 1], 0
                    
            # Dibujar en 2D
            ax_2d.plot(x, y, linewidth=1.2, alpha=0.8)
                
            # Dibujar en 3D
            ax_3d.plot(x, y, z, linewidth=1.2, alpha=0.8)

    # Aplicar límites fijos a ambas gráficas
    ax_2d.set_xlim(x_min, x_max)
    ax_2d.set_ylim(y_min, y_max)
    ax_2d.set_aspect('equal', adjustable='box')
    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    
    # Si no se pasaron ejes, mostrar ventana
    if ax_2d is None and ax_3d is None:
        plt.axis("equal")
            
def plot_Lyapunov_family_one_source_GUI(ax_2d, ax_3d, mu, temporal_scheme, V0_Lyap_family, periods, lagrange_point_index, **kwargs):
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
        
    # Obtener posición del punto de Lagrange actual
    # lagrange_text = self.lagrange_combo.currentText()
    # lagrange_index = int(lagrange_text[1]) - 1
    L_points = Lagrange_points_position(mu)
    Lx, Ly = L_points[lagrange_point_index]  # Coordenadas del punto de Lagrange

    all_x = [Lx]
    all_y = [Ly]
        
    # Resolver cada órbita
        
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
            x, y , z = U[:,0], U[:,1], 0

        all_x.extend(x)
        all_y.extend(y)

    if len(all_x) == 1:  # Solo el punto de Lagrange, nada más
        x_center, y_center = Lx, Ly
        x_range = 0.01  # valor arbitrario pequeño
        y_range = 0.01
    else:
        x_min_data, x_max_data = min(all_x), max(all_x)
        y_min_data, y_max_data = min(all_y), max(all_y)
        x_center = (x_min_data + x_max_data) / 2
        y_center = (y_min_data + y_max_data) / 2
        x_range = x_max_data - x_min_data
        y_range = y_max_data - y_min_data

    # Aplicar margen del 10% extra (es decir, 1.1 del rango total)
    margin = 0.1
    x_span = x_range * (1 + margin)
    y_span = y_range * (1 + margin)
        
    if x_span > y_span:
            
        x_min = x_center - x_span / 2
        x_max = x_center + x_span / 2
        y_min = y_center - x_span / 2
        y_max = y_center + x_span / 2
            
    else:
        
        x_min = x_center - y_span / 2
        x_max = x_center + y_span / 2
        y_min = y_center - y_span / 2
        y_max = y_center + y_span / 2

    # Dibujar cada órbita

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
            x, y , z = U[:,0], U[:,1], 0
                
        # Dibujar en 2D
        ax_2d.plot(x, y, linewidth=1.2, alpha=0.8)
                
        # Dibujar en 3D
        ax_3d.plot(x, y, z, linewidth=1.2, alpha=0.8)
        
    # Aplicar límites fijos a ambas gráficas
    ax_2d.set_xlim(x_min, x_max)
    ax_2d.set_ylim(y_min, y_max)
    ax_2d.set_aspect('equal', adjustable='box')
    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
                
    # Si no se pasaron ejes, mostrar ventana
    if ax_2d is None and ax_3d is None:
        plt.axis("equal")
        
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

    plt.close('all')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Análisis de Estabilidad de la Familia de Órbitas", fontsize=14)
    
    # Subplot 1: Re(s1) vs T
    ax1.plot(T, [s.real for s in s_main], 'o-', color='blue', label=r'$\mathrm{Re}(s_1)$')
    ax1.set_xlabel('Periodo $T$')
    ax1.set_ylabel(r'$\mathrm{Re}(s_1)$')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    #ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.legend()

    # Subplot 2: Re(s2) vs T
    ax2.plot(T, [s.real for s in s_sec], 'o-', color='red', label=r'$\mathrm{Re}(s_2)$')
    ax2.set_xlabel('Periodo $T$')
    ax2.set_ylabel(r'$\mathrm{Re}(s_2)$')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    #ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.legend()

    # Ajustar layout para evitar solapamientos
    plt.tight_layout()
    plt.show()