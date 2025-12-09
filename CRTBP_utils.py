from numpy import zeros, identity, reshape, array, sqrt, pi

"""
===============================================================================
 Archivo:       CRTBP_utils.py
 Creado:        02/12/2025
 Descripción:    
 
 Utilidades para el CRTBP.
 build_initial_condition:   Coge una CI de dimensión 6 y la expande con la matriz variacional Id_6x6.
 Lagrange_points_position:  Calcula la posición de los puntos de Lagrange.
 estimate_T_half_0:         Calcula la estimación inicial del periodo de la órbita.

 Dependencias:
    - NumPy
 Notas:
===============================================================================
"""
def build_initial_condition(U0):
    """
    Construye una CI de dimensión 42 partiendo de una CI de dimensión 6 (x,y,z,vx,vy,vz), expandiéndola con la identidad en forma de tensor 1D.
    INPUTS:
    U0: CI tipo (x,y,z,vx,vy,vz)
    OUTPUTS:
    V0: CI tipo (x,y,z,vx,vy,vz,Id_6x6)
    """
    V0 = zeros(6*6+6)
    V0[0:6] = U0[0:6]
    Id = identity(6)
    V0[6:] = reshape(Id, (6*6), copy=True)
    return V0

def Lagrange_points_position(mu):
    """
    Calcula la posición de los puntos de Lagrange del CRTBP en el plano XY. 
    INPUTS:
    mu: El parámetro de masa del sistema.
    OUTPUTS:
    array con [L1, L2, L3, L4, L5], donde Li es la posición Li=[Lix, Liy] en el plano XY.
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
    l4 = array([-mu+0.5, sqrt(3)/2])
    l5 = array([-mu+0.5, -sqrt(3)/2])
    return array([array([-(l1x+mu-1),0]), array([-(-l2x+mu-1),0]), array([-(l3x+mu),0]), l4, l5])

def estimate_T_half_0(mu, l_point_index):
    """
    Da una estimación inicial del semi-periodo de la órbita periódica.
    INPUTS: 
    mu:             Masa reducida del sistema.
    l_point_index:  Índice del punto de Lagrange (1,...,5).
    OUTPUTS:
    T_half_0:       Semi-periodo inicial.
    """
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
