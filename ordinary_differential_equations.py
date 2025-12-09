from numpy import zeros, sqrt, reshape, dot

"""
===============================================================================
 Archivo:       ordinary_differential_equations.py
 Creado:        20/11/2025
 Descripción:    
 
 Crear representaciones gráficas de la estabilidad y de familias de órbitas de Lyapunov.

 Dependencias:
    - NumPy
    
 Notas:
===============================================================================
"""

def CRTBP_variacional_JPL(t, V, mu):
    """
    Calcula G(V, mu, t) (ecuación variacional del CRTBP), donde V es el vector 
    con 

    Inputs:
    V: Vector estado del sistema. V=(x,y,z,vx,vy,vz,phi_11,phi_12,...)
    mu: Masa reducida.
    t: tiempo adimensional.

    """
    # CRTBP
    d1 = sqrt((V[0]+mu)**2 + V[1]**2 + V[2]**2)
    d2 = sqrt((V[0]-1+mu)**2 + V[1]**2 + V[2]**2)
    G = zeros(6+6*6)
    Phi = zeros((6,6))
    Phi = reshape(V[6:], (6,6), copy=True)
    G[0:3] = V[3:6]
    G[3] = V[0] + 2*V[4]-(1-mu)*(V[0]+mu)/d1**3-mu*(V[0]-1+mu)/d2**3
    G[4] = -2*V[3]+V[1]-((1-mu)/d1**3 + mu/d2**3)*V[1]
    G[5] = -((1-mu)/d1**3 + mu/d2**3)*V[2]

    # Ecuación variacional

    Uxx = 1+3*(1-mu)*(V[0]+mu)**2/d1**5 - (1-mu)/d1**3 + 3*mu*(V[0]-1+mu)**2/d2**5 - mu/d2**3
    Uyy = 1-(1-mu)*(d1**2-3*V[1]**2)/d1**5 -mu*(d2**2-3*V[1]**2)/d2**5
    Uzz = -(1-mu)*(d1**2-3*V[2]**2)/d1**5 -mu*(d2**2-3*V[2]**2)/d2**5
    Uxy = 3*(1-mu)*((V[0]+mu)*V[1])/d1**5+3*mu*(V[0]-1+mu)*V[1]/d2**5
    Uxz = (3*(1-mu)*(V[0]+mu)/d1**5 + 3*mu*(V[0]-1+mu)/d2**5)*V[2]
    Uyz = 3*V[1]*V[2]*((1-mu)/d1**5 + mu/d2**5)
    
    J = zeros((6,6))
    J[0,3] = 1
    J[1,4] = 1
    J[2,5] = 1
    J[3,0] = Uxx
    J[4,0] = Uxy
    J[5,0] = Uxz
    J[3,1] = Uxy
    J[4,1] = Uyy
    J[5,1] = Uyz
    J[3,2] = Uxz
    J[4,2] = Uyz
    J[5,2] = Uzz
    J[4,3] = -2
    J[3,4] = 2

    J_times_Phi = dot(J, Phi)

    G[6:]=reshape(J_times_Phi,(6*6), copy=True)

    return G







