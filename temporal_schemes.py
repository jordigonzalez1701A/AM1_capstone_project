from numpy import zeros, array, dot, linspace, arange, prod, sum, asarray
from numpy.linalg import norm
from Butcher import Butcher_tableau
from Cauchy import Cauchy_problem


def RK4(F, U1, mu, t1, t2, **kwargs):
    """
    Performs an iteration of Runge-Kutta 4.
    F: The function F(U, mu, t) of the problem.
    U: The variable U of the Cauchy problem.
    mu: Reduced mass.
    t1: The initial time of the step.
    t2: The final time of the step.
    """
    dt = t2 - t1

    k1 = F(t1, U1, mu)
    k2 = F(t1 + 1/2*dt, U1 + 1/2*k1*dt, mu)
    k3 = F(t1 + 1/2*dt, U1 + 1/2*k2*dt, mu)
    k4 = F(t2, U1 + k3*dt, mu)

    return U1 + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def RK45(F, U1, mu, t1, t2):
    """
    RK4(5) from "LOW-ORDER CLASSICAL: RUNGE-KUTTA FORMULAS WITH STEPSIZE CONTROL AND THEIR APPLICATION TO SOME HEAT TRANSFER PROBLEMS", Erwin Fehlberg
    """ 

    return ERK(F, U1, mu, t1, t2, "RK4(5)")    


def RK547M(F, U1, mu, t1, t2):
    """
    RK5(4)7M from "A family of embedded Runge-Kutta formulae", J. R. Dormand and P. J. Prince
    """  

    return ERK(F, U1, mu, t1, t2, "RK5(4)7M")


def RK56(F, U1, mu, t1, t2):
    """
    RK5(6) from "CLASSICAL FIFTH-, SIXTH-, SEVENTH-, AND EIGHTH-ORDER RUNGE-KUTTA FORMULAS WITH STEPSIZE CONTROL", Erwin Fehlberg
    """ 

    return ERK(F, U1, mu, t1, t2, "RK5(6)")


def RK658M(F, U1, mu, t1, t2):
    """
    RK6(5)8M from "High order embedded Runge-Kutta formulae", P. J. Prince and J. R. Dormand
    """ 

    return ERK(F, U1, mu, t1, t2, "RK6(5)8M")


def RK67(F, U1, mu, t1, t2):
    """
    RK6(7) from "CLASSICAL FIFTH-, SIXTH-, SEVENTH-, AND EIGHTH-ORDER RUNGE-KUTTA FORMULAS WITH STEPSIZE CONTROL", Erwin Fehlberg
    """ 

    return ERK(F, U1, mu, t1, t2, "RK6(7)")


def RK78(F, U1, mu, t1, t2):
    """
    RK7(8) from "CLASSICAL FIFTH-, SIXTH-, SEVENTH-, AND EIGHTH-ORDER RUNGE-KUTTA FORMULAS WITH STEPSIZE CONTROL", Erwin Fehlberg
    """ 

    return ERK(F, U1, mu, t1, t2, "RK7(8)")


def RK8713M(F, U1, mu, t1, t2):
    """
    RK8(7)13M from "High order embedded Runge-Kutta formulae", P. J. Prince and J. R. Dormand
    """    

    return ERK(F, U1, mu, t1, t2, "RK8(7)13M")


def ERK(F, U1, mu, t1, t2, ERK_scheme):
    """
    U_sol: Estimation of the solution using the RK method of order q
    U_error: Estimation of the error using the RK method of order p
    """

    def step_size(U_high, U_low, dt, q, eps=1e-7):
    
        T = norm(U_high-U_low)
        if T > eps:
            return dt*(eps/T)**(1/(q)) 
        else:
            return dt 

    dt = t2 - t1
    Nv = len(U1)

    c, a, b, bs, q, p, s = Butcher_tableau(ERK_scheme)
    k = RK_stages(F, U1, mu, t1, t2, s, a, c)

    dU_sol = zeros(Nv)
    dU_error = zeros(Nv)
    for i in range(s):    
        dU_sol += dt*b[i]*k[i]
        dU_error += dt*bs[i]*k[i]    

    U_sol = U1 + dU_sol
    U_error = U1 + dU_error     
    
    dt_max = step_size(U_sol, U_error, dt, q)

    N = int(dt/dt_max) + 1 # Si dt_max = dt, N = 2
    h = dt/N
    t_step = linspace(t1, t2, N+1) # Divido el intervalo [t1, t2] en N puntos     
       
    U2 = U1.copy()
    for n in range(N):
        k = RK_stages(F, U2, mu, t_step[n], t_step[n+1], s, a, c)
        dU = zeros(Nv)
        for i in range(s):    
            dU += h*b[i]*k[i]       

        U2 += dU

    return U2


def RK_stages(F, U1, mu, t1, t2, s, a, c):

    dt = t2 - t1
    Nv = len(U1)

    k = zeros((s, Nv))
    for i in range(s):  
        dU_stage = zeros(Nv)  
        for j in range(i): # range(i) porque a[i,j] = 0 para j>i
            dU_stage += dt*a[i,j]*k[j] 

        k[i,:] = F(t1 + c[i]*dt, U1 + dU_stage, mu)

    return k
    

def GBS(F, U1, mu, t1, t2, Nl=5):

    def modified_midpoint_scheme(F, U1, mu, t1, t2, Ni):
   
        h = (t2 - t1)/(2*Ni)
        t = linspace(t1, t2 + h, 2*Ni+1)
        Nv = len(U1)
        
        U = zeros((2*Ni+2, Nv)) # 2*Ni+2 para que funcione la linea de U2 abajo
        U[0,:] = U1
        U[1,:] = U1 + h*F(t1, U1, mu)

        # Leap Frog goes to t2 + h = t_(2Ni+1)
        for i in range(1, 2*Ni+1):
            U[i+1,:] = U[i-1,:] + 2*h*F(t[i], U[i,:], mu)
        
        # Average value at t2 = t1 + 2*Ni*h
        U2 = (U[2*Ni+1,:] + 2*U[2*Ni,:] + U[2*Ni-1,:])/4

        return U2       


    def corrected_solution_richardson(N, U):

        Nl = len(N)
        h = 1/(2*N)
        x = h**2

        Lagrange = zeros(Nl)
        w = zeros(Nl)
        
        if Nl == 1:
            Lagrange[:] = 1
            w[:] = 1
        else:
            for j in range(Nl):
                idx = arange(Nl) != j
                Lagrange[j] = prod(x[idx] / (x[idx] - x[j]))
                w[j] = 1.0/prod(x[j] - x[idx]) 
                
        wx = w / x 
        Uc = wx @ U / sum(w/x)

        return Uc 


    def mesh_refinement(levels):
        
        N_Romberg   = array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        N_Burlirsch = array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
        N_Harmonic  = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        return N_Harmonic[0:levels]
    
    
    Nv = len(U1)
    N = mesh_refinement(Nl)

    U = zeros((Nl, Nv))
    for i in range(Nl):
        U[i, :] = modified_midpoint_scheme(F, U1, mu, t1, t2, N[i])

    U2 = corrected_solution_richardson(N, U)

    return U2




