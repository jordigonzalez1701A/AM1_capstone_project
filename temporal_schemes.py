from numpy import zeros

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
    N = len(t)
    N_v = len(U0)
    U = zeros((N, N_v))
    U[0,:] = U0

    for n in range(0, N-1):
        U[n+1,:] = temporal_scheme(F, U[n,:], mu, t[n], t[n+1], **kwargs)

    return U

def RK4(F,U, mu, t1, t2, **kwargs):
    """
    Performs an iteration of Runge-Kutta 4.
    F: The function F(U, mu, t) of the problem.
    U: The variable U of the Cauchy problem.
    mu: Reduced mass.
    t1: The initial time of the step.
    t2: The final time of the step.
    """
    k1 = F(U, mu, t1)
    k2 = F(U+0.5*k1*(t2-t1), mu, t1+0.5*(t2-t1))
    k3 = F(U+0.5*k2*(t2-t1), mu, t1+0.5*(t2-t1))
    k4 = F(U+k3*(t2-t1), mu, t1+(t2-t1))

    return U + (1.0/6.0)*(t2-t1)*(k1 + 2*k2+2*k3 + k4)