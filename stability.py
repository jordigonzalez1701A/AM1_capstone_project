from numpy import reshape, imag
from numpy.linalg import eigvals

"""
===============================================================================
 Archivo:       stability.py
 Creado:        02/12/2025
 Descripción:    
 
 Cálculo de la estabilidad de una órbita de Lyapunov.

 Dependencias:
    - NumPy
 Notas:
===============================================================================
"""

def stability(V_T):
    """
    Calcula los índices de estabilidad de una órbita periódica.
    INPUTS:
    V_T: Vector de estado de la ecuación variacional en el instante de tiempo T, donde T es el periodo de la órbita periódica.
    OUTPUTS:
    s1, s2: índices de estabilidad.
    """
    Phi_T = reshape(V_T[6:], (6,6))
    eigs = eigvals(Phi_T)
    tol = 1e-6
    nontrivial = [eig for eig in eigs if abs(abs(eig)-1) > tol or abs(imag(eig)) > tol]
    pairs = []
    used = set()
    for i, lam in enumerate(nontrivial):
        if i in used: continue
        for j in range(i+1, len(nontrivial)):
            if j in used: continue
            if abs(lam * nontrivial[j] - 1) < tol:
                pairs.append((lam, nontrivial[j]))
                used.update([i,j])
                break
    if len(pairs) < 2:
        return 0+0j, 0+0j
    s1 = 0.5 * (pairs[0][0] + pairs[0][1])
    s2 = 0.5 * (pairs[1][0] + pairs[1][1])
    return s1, s2
