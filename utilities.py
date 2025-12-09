from numpy import where, hstack, array, savetxt, loadtxt, any
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

