# PyGrange: Un explorador de órbitas alrededor de los puntos de Lagrange colineales

PyGrange permite explorar órbitas periódicas de Lyapunov en el problema restringido circular de los tres cuerpos (CRTBP), incluye:

-un método single shooting para refinar condiciones iniciales de órbitas periódicas.
-Un método de continuación para hallar familias de órbitas periódicas.
-Cálculo de los modos de Floquet para estudiar la estabilidad de familia de órbitas.
-Herramientas de guardado y recuperación de archivos de condiciones iniciales.
-Visualizador de órbitas y gráficos de estabilidad.
-Diversos esquemas temporales de alto orden.
-Una GUI para configurar el programa.

Dependencias:
-PyQt6
-scipy
-NumPy
-matplotlib

Para ejecutar este software:
Abrir 'GUI.py', se mostrará la interfaz gráfica de usuario. Rellenar los campos y ejecutar. Aparecerán los resultados automáticamente.