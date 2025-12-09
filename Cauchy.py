from numpy import zeros

"""
===============================================================================
 Archivo:       Cauchy.py
 Creado:        20/11/2025
 Descripción:    
 
 Funciones para resolver el problema de Cauchy mediante esquemas temporales.
 Dependencias:
    - NumPy
    
 Notas:
===============================================================================
"""

def Cauchy_problem(F, U0, mu, t, temporal_scheme, **kwargs):
    """
    Function to integrate a Cauchy problem using a given numeric method.
    Implicit methods will be solved using Newton's method.
    
    Inputs:
    F: F(U,t) of the Cauchy problem.
    U0: Inital condition
    mu: Reduced mass
    t: np.array of time values on which to integrate.
    
    KWARGS:
    temporal_scheme: function to call for the numeric method.
    tol_jacobian: Tolerance of the computation of the Jacobian (if applicable).
    N_max: Max. number of iterations for Newton's method (if applicable).
    newton_tol: Tolerance for Newton's method (if applicable).
    Returns:
    U
    """
    N = len(t) - 1
    Nv = len(U0)
    U = zeros((N+1, Nv))
    U[0,:] = U0

    for n in range(0, N):
        U[n+1,:] = temporal_scheme(F, U[n,:], mu, t[n], t[n+1], **kwargs)

    return U


def Cauchy_problem_intersect_y(F, U0, mu, t, temporal_scheme, **kwargs):
    """
    Function to integrate a Cauchy problem using a given numeric method until y=0.
    Implicit methods will be solved using Newton's method.
    
    Inputs:
    F: F(U,t) of the Cauchy problem.
    U0: Inital condition
    mu: Reduced mass
    t: np.array of time values on which to integrate.
    temporal_scheme: function to call for the numeric method.
    
    KWARGS:
    
    tol_jacobian: Tolerance of the computation of the Jacobian (if applicable).
    N_max: Max. number of iterations for Newton's method (if applicable).
    newton_tol: Tolerance for Newton's method (if applicable).
    Returns:
    U
    """
    N = len(t) - 1
    Nv = len(U0)
    U = zeros((N+1, Nv))
    U[0,:] = U0
    n_stop = 0
    for n in range(0, N):
        U[n+1,:] = temporal_scheme(F, U[n,:], mu, t[n], t[n+1], **kwargs)
        if (U[n+1,1]*U[n,1]<0) and (t[n]>0):
            n_stop = n+1
            break
        if n == N-2:
            n_stop = N-1

    return U, n_stop