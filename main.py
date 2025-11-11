from temporal_schemes import Cauchy_problem, RK4
from ordinary_differential_equations import CRTBP_variacional
from numpy import linspace, zeros, identity, reshape, array, sqrt, concatenate
import matplotlib.pyplot as plt

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


t = linspace(0, 1, 1000)


mu = 0.0121505856 #Earth-Moon
#U0 = array([mu-0.5, sqrt(3)/2, 0, 0, 0, 0])
#V0 = build_initial_condition(U0)

#V = Cauchy_problem(CRTBP_variacional, V0, mu, t, RK4)

lagrange = Lagrange_points_position(mu)
print(lagrange)
plt.scatter(mu-1,0, color="black")
plt.scatter(mu,0, color="black")
for i in range(0,5):
    plt.scatter(lagrange[i,0], lagrange[i,1])

#plt.scatter(V[:,0], V[:,1])

plt.show()