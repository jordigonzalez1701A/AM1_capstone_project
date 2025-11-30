from Cauchy import Cauchy_problem, Cauchy_problem_intersect_y
from temporal_schemes import RK4
from ordinary_differential_equations import CRTBP_variacional, CRTBP_variacional_JPL
from numpy import linspace, zeros, identity, reshape, array, sqrt, concatenate, imag, pi
from numpy.linalg import solve, eigvals
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
    l4 = array([mu-0.5, sqrt(3)/2])
    l5 = array([mu-0.5, -sqrt(3)/2])
    return array([array([l1x+mu-1,0]), array([-l2x+mu-1,0]), array([l3x+mu,0]), l4, l5])


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

    N_max = 1000000
    n_iter = 0
    converged = False
    T_half = T0
    plot_intermediate_steps = False
    while(n_iter < N_max):

        #V_intersect, n_intersect = Cauchy_problem_intersect_y(CRTBP_variacional, V0, mu, t, temporal_scheme, **kwargs)
        #print(f"T_half:{T_half}")
        sol = solve_ivp(CRTBP_variacional_JPL, t_span=(0, 1.25*T_half), y0=V0, method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12)
        # Interpolar en y=0
        #y0 = V_intersect[n_intersect-1, 1]
        #y1 = V_intersect[n_intersect, 1]
        #alpha = y0 / (y0 - y1)
        #V_cross = V_intersect[n_intersect-1,:] + alpha*(V_intersect[n_intersect,:] - V_intersect[n_intersect-1,:])
        # Plot orbit with crossings
        for i in range(0, sol.y.shape[1]):
            if i>0 and sol.y[1,i]*sol.y[1,i-1]<0:
                y0 = sol.y[1, i-1]
                y1 = sol.y[1, i]
                alpha = y0 / (y0 - y1)  # fraction between i-1 and i
                V_cross = sol.y[:, i-1] + alpha * (sol.y[:, i] - sol.y[:, i-1])
                T_half = sol.t[i-1] + alpha * (sol.t[i] - sol.t[i-1])
                break
        
        # All crossing times
        #print("Crossing times:", sol.t[i])
        # Corresponding states
        #print("Crossing states:", V_cross)
        if plot_intermediate_steps == True:
            plt.plot(sol.y[0], sol.y[1], marker="o", label="Lyapunov Orbit")
            plt.scatter(V_cross[0], V_cross[1], label="Vcross", color="green")
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
  
        #damping = 0.2
        #v_y_new = V0[4] - damping*V_cross[3]/V_cross[28]
        v_y_new = V0[4] + delta_z_doty[1]
        z_new = V0[2] + delta_z_doty[0]
        V0[4] = v_y_new
        V0[2] = z_new
        n_iter = n_iter + 1
        print(f"n_iter: {n_iter}. vx(T): {V_cross[3]:.4e}\r", end="\r", flush=True)
        
        if abs(V_cross[3])< epsilon:
            converged = True
            
            return converged, V0, 2*T_half
    
    return converged, V0, 2*T_half



t = linspace(0, 1, 1000)


mu = 0.0121505856 #Earth-Moon
#U0 = array([mu-0.5, sqrt(3)/2, 0, 0, 0, 0])
#V0 = build_initial_condition(U0)

#V = Cauchy_problem(CRTBP_variacional, V0, mu, t, RK4)

#Encontrar y plotear puntos de Lagrange para un mu determinado
lagrange = Lagrange_points_position(mu)
print(lagrange)
plt.scatter(mu-1,0, color="black")
plt.scatter(mu,0, color="black")
for i in range(0,5):
    plt.scatter(lagrange[i,0], lagrange[i,1])

#
plt.show()
#Construye una CI para encontrar una órbita de Lyapunov cerca de L1
l_point_vector = array([lagrange[0,0], lagrange[0,1], 0, 0,0,0])

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
print(f"vy0: {vy0}")
T_half0 = (pi)/s
# A_y = 1e-2
# Jacobian_L1 = Jacobian_CRTBP(mu, array([lagrange[0,0], lagrange[0,1],0,0,0,0]))
# eigenvalues_Jacobian_L1 = eigvals(Jacobian_L1)
# omega = abs(imag(eigenvalues_Jacobian_L1[imag(eigenvalues_Jacobian_L1)>0][0]))
# T_half0 = pi / omega
# vy0 = A_y*omega

#U0_L1 = concatenate((lagrange[0,:]+array([xpert0,0]),array([0, 0, vy0, 0])))

#U0_L1 = array([8.3690888734309465E-1,	2.6569853876973453E-29,	-6.5553023513506719E-32,	6.0314258522129136E-16,	5.2232242080210143E-5,	5.1825223150917431E-31])
U0_L1 = array([8.3690888734309465E-1+1e-7,	0,	0,	0,	5e-4,	0])
print(f"T_half0: {T_half0}, vy0: {vy0}, omega:{s}")
V0_L1 = build_initial_condition(U0_L1)



Lyap_found, V0_Lyap, Lyap_period = find_Lyapunov_orbit(V0_L1, mu, T_half0, RK4)

if Lyap_found == True:
    print(f"Lyap_period: {Lyap_period}, V0_Lyap: {V0_Lyap}")
    sol_Lyap = solve_ivp(CRTBP_variacional_JPL, t_span=(0, Lyap_period), y0=V0_Lyap, method="DOP853", args=(mu,), rtol=1e-10, atol=1e-12)
    print(f"sol_Lyap: {sol_Lyap.y}")


plt.plot(sol_Lyap.y[0,:], sol_Lyap.y[1,:], marker="o")
plt.axis("equal")
plt.show()