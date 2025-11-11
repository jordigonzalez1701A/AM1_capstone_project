from temporal_schemes import Cauchy_problem, RK4
from ordinary_differential_equations import CRTBP_variacional
from numpy import linspace, zeros, identity, reshape, array, sqrt
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




t = linspace(0, 1, 1000)


mu = 0.0121505856 #Earth-Moon
U0 = array([mu-0.5, sqrt(3)/2, 0, 0, 0, 0])
V0 = build_initial_condition(U0)

V = Cauchy_problem(CRTBP_variacional, V0, mu, t, RK4)

plt.scatter(V[:,0], V[:,1])
plt.show()