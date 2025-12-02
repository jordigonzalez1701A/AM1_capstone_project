from numpy import where, hstack, array, savetxt, loadtxt
"""
===============================================================================
 Archivo:       utilities.py
 Creado:        02/12/2025
 Descripción:    

 Funciones de utilidad:

 cargar_CIs_principales:  Carga las CI principales de un archivo.
 guardar_CIs:             Guarda las CIs de una familia de Lyapunov en un archivo.
 cargar_CIs:              Carga CIs de un archivo.

 Dependencias:
    - NumPy

 Notas:
===============================================================================
"""

def cargar_CIs_principales(filename):
    """
    Carga las CIs principales de un archivo. Pasa las componentes y en valor absoluto para mejorar la convergancia del algoritmo de single shooting.
    INPUTS:
    filename: Nombre del archivo
    OUTPUTS:  
    ics_list: Array (N_CIs, 6) con todas las condiciones iniciales del archivo.
    """
    ics_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            numbers = [float(n) for n in line.split(',')]
            numbers[1] = abs(numbers[1])
            ics_list.append(numbers)
    return array(ics_list)

def guardar_CIs(V0_Lyap_family, Lyap_p_fam, filename):
    """
    Guarda las CIs de una familia de Lyapunov en un archivo.
    INPUTS:
    V0_Lyap_family:     Array (N_family, 42) de condiciones iniciales de la familia de Lyapunov.
    Lyap_p_fam:         Array (42) de periodos de las órbitas de la familia de Lyapunov.
    filename:           Nombre del archivo donde guardar las CIs.
    OUTPUTS:
    Ninguno.
    """
    found = where(any(V0_Lyap_family, axis=1))[0]
    V_save = V0_Lyap_family[found]
    T_save = Lyap_p_fam[found]
    data = hstack([T_save.reshape(-1,1), V_save])
    with open(filename, 'w') as f:  # 'w' para sobrescribir (mejor que 'a')
        savetxt(f, data, delimiter=",", fmt="%.12e")

def cargar_CIs(filename):
    """
    Carga CIs de las órbitas de una familia de Lyapunov de un archivo.
    INPUTS:
    filename: Nombre del archivo.
    OUTPUTS:
    periods: Array (N_CIs, 1) con el periodo de cada órbita en el archivo.
    V0:      Array (N_CIs, 42) con la condición inicial en el problema variacional de cada órbita de la familia de Lyapunov.
    """
    data = loadtxt(filename, delimiter=",")
    periods = data[:,0]
    V0 = data[:,1:]
    return periods, V0

