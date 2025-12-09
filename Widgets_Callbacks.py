import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from matplotlib.backends.backend_qt5agg import *
from matplotlib.figure import *
from utilities import *
from temporal_schemes import *
from stability import *
from plotting import *

class Lyapunov_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Familia de Órbitas de Lyapunov - CRTBP")
        self.setGeometry(100, 100, 1000, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # === Contenedor principal con Grid Layout ===
        controls_container = QWidget()
        grid_layout = QGridLayout()
        controls_container.setLayout(grid_layout)
        main_layout.addWidget(controls_container)

        # --- Columna 1: Sistema, Punto Lagrange, Esquema numérico ---
        col1_layout = QVBoxLayout()
        col1_layout.addWidget(QLabel("Sistema:"))
        self.system_combo = QComboBox()
        self.system_combo.addItems(["Sun_Earth", "Earth_Moon"])
        col1_layout.addWidget(self.system_combo)

        col1_layout.addWidget(QLabel("Punto de Lagrange:"))
        self.lagrange_combo = QComboBox()
        self.lagrange_combo.addItems(["L1", "L2", "L3"])
        col1_layout.addWidget(self.lagrange_combo)

        col1_layout.addWidget(QLabel("Esquema numérico:"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems([
            "RK4", "RK45", "RK547M", "RK56",
            "RK658M", "RK78", "RK8713M", "GBS"
        ])
        self.scheme_dict = {
            "RK4": RK4,
            "RK45": RK45,
            "RK547M": RK547M,
            "RK56": RK56,
            "RK658M": RK658M,
            "RK78": RK78,
            "RK8713M": RK8713M,
            "GBS": GBS
        }
        
        col1_layout.addWidget(self.scheme_combo)
        grid_layout.addLayout(col1_layout, 0, 0, 3, 1)  # Ocupa 3 filas, 1 columna

        # --- Columna 2: Checkbox CIs + parámetros (Δx₀, N órbitas) ---
        col2_layout = QVBoxLayout()
        self.ci_checkbox = QCheckBox("Cargar CIs")
        self.ci_checkbox.setChecked(True)
        
        col2_layout.addWidget(self.ci_checkbox)

        col2_layout.addWidget(QLabel("Δx₀ (perturbación):"))
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(1e-8, 1e-3)
        self.dx_spin.setValue(2e-5)
        self.dx_spin.setDecimals(10)
        col2_layout.addWidget(self.dx_spin)

        col2_layout.addWidget(QLabel("N órbitas:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 15)
        self.n_spin.setValue(5)
        col2_layout.addWidget(self.n_spin)
        grid_layout.addLayout(col2_layout, 0, 1, 3, 1)  # Ocupa 3 filas, 1 columna

        # --- Columna 3: Checkbox Familia + Carpeta + Archivos ---
        col3_layout = QVBoxLayout()
        self.family_checkbox = QCheckBox("Cargar Familia de Órbitas")
        self.family_checkbox.setChecked(False)
        
        col3_layout.addWidget(self.family_checkbox)

        self.select_folder_button = QPushButton("Elegir carpeta...")
        
        col3_layout.addWidget(self.select_folder_button)

        self.ci_files_combo = QComboBox()
        self.ci_files_combo.setEnabled(False)
        self.ci_files_combo.setMaximumWidth(350)
        self.ci_files_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        col3_layout.addWidget(self.ci_files_combo)
        
        self.stability_checkbox = QCheckBox("Analizar estabilidad")
        self.stability_checkbox.setChecked(False)
        col3_layout.addWidget(self.stability_checkbox)
        grid_layout.addLayout(col3_layout, 0, 2, 1, 1)  # Ocupa 3 filas, 1 columna

        # --- Columna 4: Botones principales ---
        col4_layout = QVBoxLayout()
        self.run_button = QPushButton("Generar Órbitas")
        col4_layout.addWidget(self.run_button)

        self.save_button = QPushButton("Guardar CIs")
        
        col4_layout.addWidget(self.save_button)

        self.clear_button = QPushButton("Borrar órbitas")
        
        col4_layout.addWidget(self.clear_button)
        grid_layout.addLayout(col4_layout, 0, 3, 3, 1)  # Ocupa 3 filas, 1 columna

        # --- Espacio vertical entre controles y gráfico ---
        main_layout.addSpacing(10)

        # --- Gráficas: 2D y 3D, con etiquetas en los ejes ---
        fig_2d = Figure(figsize=(5, 5), dpi=100)
        self.ax_2d = fig_2d.add_subplot(111)
        self.ax_2d.set_title("Órbita 2D")
        self.ax_2d.set_xlabel(r"$x$")       # Etiqueta del eje X
        self.ax_2d.set_ylabel(r"$y$")       # Etiqueta del eje Y
        self.ax_2d.grid(True)
        self.ax_2d.set_aspect('equal')
        self.canvas_2d = FigureCanvasQTAgg(fig_2d)
        self.toolbar_2d = NavigationToolbar2QT(self.canvas_2d, self)

        # Figura 3D
        fig_3d = Figure(figsize=(5, 5), dpi=100)
        self.ax_3d = fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_title("Órbita 3D")
        self.ax_3d.set_xlabel(r"$x$")       # Eje X
        self.ax_3d.set_ylabel(r"$y$")       # Eje Y
        self.ax_3d.set_zlabel(r"$z$")       # Eje Z
        self.canvas_3d = FigureCanvasQTAgg(fig_3d)
        self.toolbar_3d = NavigationToolbar2QT(self.canvas_3d, self)

        # Layout vertical para 2D + toolbar
        vbox_2d = QVBoxLayout()
        vbox_2d.addWidget(self.canvas_2d)
        vbox_2d.addWidget(self.toolbar_2d)

        # Layout vertical para 3D + toolbar
        vbox_3d = QVBoxLayout()
        vbox_3d.addWidget(self.canvas_3d)
        vbox_3d.addWidget(self.toolbar_3d)

        # Layout horizontal para ambas columnas
        plot_layout = QHBoxLayout()
        plot_layout.addLayout(vbox_2d)
        plot_layout.addLayout(vbox_3d)
        main_layout.addLayout(plot_layout)

        # Etiqueta de estado (pequeña, al final)
        self.status_label = QLabel("Listo")
        main_layout.addWidget(self.status_label)

        # Añadir stretch al final para que las gráficas se expandan
        main_layout.addStretch(1)  # Esto hace que las gráficas ocupen todo el espacio restante

        # Desactivar controles según estado inicial
        self.update_input_states()
        self.ci_checkbox.toggled.connect(self.on_ci_toggled)
        self.family_checkbox.toggled.connect(self.on_family_toggled)
        self.select_folder_button.clicked.connect(self.select_ci_folder)
        self.save_button.clicked.connect(self.save_CIs)
        self.clear_button.clicked.connect(self.clear_orbits) 
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        self.ci_files_combo.currentTextChanged.connect(self.on_file_selected)
        self.on_system_changed()
        self.run_button.clicked.connect(self.on_run_button_clicked)
        
    def on_ci_toggled(self, checked):
        if checked:
            self.family_checkbox.setChecked(False)
        self.update_input_states()
        self.update_file_list()

    def on_family_toggled(self, checked):
        if checked:
            self.ci_checkbox.setChecked(False)
        self.update_input_states()
        self.update_file_list()
    
    def select_ci_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta con archivos de CIs",
            os.path.expanduser("~")  # Carpeta inicial: home del usuario
        )
        if folder:
            self.selected_ci_folder = folder
            self.update_file_list()
    
    def on_file_selected(self, filename):
        """
        Actualiza el sistema y el punto de Lagrange según el nombre del archivo seleccionado.
        """
        if not filename or filename.startswith("No se encontraron"):
            return

        # Normalizar el nombre: quitar extensión .txt
        name = filename.replace('.txt', '')

        # Caso 1: archivo de familia -> "Lyapunov_family_L1_Earth_Moon"
        if name.startswith("Lyapunov_family_"):
            parts = name[len("Lyapunov_family_"):].split('_')
            if len(parts) >= 2:
                lagrange = parts[0]  # Ej: "L1"
                system = '_'.join(parts[1:])  # Ej: "Earth_Moon"

        # Caso 2: archivo de CI -> "L1_Earth_Moon_CI"
        elif name.endswith("_CI"):
            parts = name[:-3].split('_')  # Quitar "_CI"
            if len(parts) >= 2:
                lagrange = parts[0]  # Ej: "L1"
                system = '_'.join(parts[1:])  # Ej: "Earth_Moon"

        else:
            # Nombre no reconocido → no hacer nada
            return

        # Validar y actualizar combo boxes
        if system in ["Sun_Earth", "Earth_Moon"]:
            # Bloquear señales temporalmente para evitar llamadas innecesarias
            self.system_combo.blockSignals(True)
            self.system_combo.setCurrentText(system)
            self.system_combo.blockSignals(False)

        if lagrange in ["L1", "L2", "L3"]:
            self.lagrange_combo.blockSignals(True)
            self.lagrange_combo.setCurrentText(lagrange)
            self.lagrange_combo.blockSignals(False)

        # Opcional: actualizar puntos de Lagrange tras el cambio
        self.on_system_changed()
    
    def update_input_states(self):
        ci_mode = self.ci_checkbox.isChecked()
        self.dx_spin.setEnabled(ci_mode)
        self.n_spin.setEnabled(ci_mode)

    def save_CIs(self):
        """
        Guarda las condiciones iniciales (CIs) generadas en un archivo de texto.
        Solo guarda aquellas con periodo > 0 (convergidas).
        """
        # Verificar que existan datos generados
        if not hasattr(self, 'V0_fam') or not hasattr(self, 'T_fam'):
            self.status_label.setText("No hay CIs para guardar.")
            return

        # Obtener parámetros actuales
        system = self.system_combo.currentText()
        lagrange_point = self.lagrange_combo.currentText()
        scheme = self.scheme_combo.currentText()
        if system == "Sun_Earth":
            mu = 3.0542e-6
        elif system == "Earth_Moon":
            mu = 1.215058560962404e-2

        # Diálogo para guardar archivo
        default_name = f"Lyapunov_family_{lagrange_point}_{system}.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Condiciones Iniciales (CIs)",
            default_name,
            "Archivos de texto (*.txt);;Todos los archivos (*)"
        )
        if not filename:
            return  # Cancelado por el usuario

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Escribir encabezado
                f.write(f"# Condiciones Iniciales (CIs) - Órbitas de Lyapunov\n")
                f.write(f"# Sistema: {system}\n")
                f.write(f"# Punto de Lagrange: {lagrange_point}\n")
                f.write(f"# Esquema numérico: {scheme}\n")
                f.write(f"# μ = {mu:.12e}\n")
                f.write(f"# Dimensiones: cada CI tiene 42 valores (x, y, z, vx, vy, vz + 36 variaciones)\n")
                f.write(f"# Periodo de la órbita completa al principio de cada línea\n")
                f.write("#\n")

                count_saved = 0
                for i in range(self.V0_fam.shape[0]):
                    for j in range(self.V0_fam.shape[1]):
                        period = self.T_fam[i, j]
                        if period > 1e-1:  # Solo CIs convergidas
                            ci_values = self.V0_fam[i, j]
                            # Formato: 42 valores de CI + 1 valor de periodo
                            line = f"{period:.12e}," + ",".join(f"{v:.12e}" for v in ci_values) + "\n"
                            f.write(line)
                            count_saved += 1

                f.write(f"# Total de CIs guardadas: {count_saved}\n")

            self.status_label.setText(f"CIs guardadas en: {os.path.basename(filename)}")

        except Exception as e:
            self.status_label.setText(f"Error al guardar: {str(e)}")

    def clear_orbits(self):
        """
        Borra todas las órbitas dibujadas (líneas y colecciones que no sean de Lagrange).
        No toca los puntos de Lagrange ni primarios (marcados con _lagrange).
        """
        # === Borrar líneas (órbitas) en 2D ===
        for line in self.ax_2d.lines[:]:
            line.remove()

        # === Borrar colecciones que no sean de Lagrange en 2D ===
        for collection in self.ax_2d.collections[:]:
            if not (hasattr(collection, '_lagrange') and collection._lagrange):
                collection.remove()

        # === Borrar líneas (órbitas) en 3D ===
        for line in self.ax_3d.lines[:]:
            line.remove()

        # === Borrar colecciones que no sean de Lagrange en 3D ===
        for collection in self.ax_3d.collections[:]:
            if not (hasattr(collection, '_lagrange') and collection._lagrange):
                collection.remove()

        # === Actualizar los lienzos ===
        self.canvas_2d.draw()
        self.canvas_3d.draw()

        # === Mensaje de estado ===
        self.status_label.setText("Órbitas borradas.")
        
    def on_system_changed(self):
        system = self.system_combo.currentText()
        if system == "Sun_Earth":
            mu = 3.0542e-6
            M1_label = "Sun"
            M2_label = "Earth"
            M1_color = "gold"
            M2_color = "deepskyblue"
            M1_size = 300
            M2_size = 80 
        elif system == "Earth_Moon":
            mu = 1.215058560962404e-2
            M1_label = "Earth"
            M2_label = "Moon"
            M1_color = "deepskyblue"
            M2_color = "lightgray"
            M1_size = 200
            M2_size = 60
        
        # Dibuja usando la función pura
        plot_lagrange_points(self.ax_2d, self.ax_3d, mu, M1_label, M2_label, M1_color, M2_color, M1_size, M2_size)
        
        # Actualiza los lienzos
        self.canvas_2d.draw()
        self.canvas_3d.draw()
    
    def update_file_list(self):
        if not hasattr(self, 'selected_ci_folder'):
            self.ci_files_combo.clear()
            self.ci_files_combo.setEnabled(False)
            return

        folder = self.selected_ci_folder
        try:
            all_files = [
                f for f in os.listdir(folder)
                if f.endswith('.txt') and os.path.isfile(os.path.join(folder, f))
            ]

            if self.ci_checkbox.isChecked():
                # Modo CIs: solo *_CI.txt
                files = [f for f in all_files if f.endswith('_CI.txt')]
                placeholder = "No se encontraron archivos *_CI.txt"
            else:
                # Modo Familia: solo *family*.txt (contiene 'family' en el nombre)
                files = [f for f in all_files if 'family' in f.lower()]
                placeholder = "No se encontraron archivos con 'family' en el nombre"

            files.sort()
            self.ci_files_combo.clear()
            if files:
                self.ci_files_combo.addItems(files)
                self.ci_files_combo.setEnabled(True)
                for i, file in enumerate(files):
                    self.ci_files_combo.setItemData(i, file, Qt.ItemDataRole.ToolTipRole)
                self.status_label.setText(f"Carpeta cargada: {os.path.basename(folder)}")
            else:
                self.ci_files_combo.addItem(placeholder)
                self.ci_files_combo.setEnabled(False)
                self.status_label.setText(placeholder)

        except Exception as e:
            self.status_label.setText(f"Error al leer carpeta: {str(e)}")
            self.ci_files_combo.clear()
            self.ci_files_combo.setEnabled(False)
    
    def on_run_button_clicked(self):
        for collection in self.ax_2d.collections[:]:
            if not (hasattr(collection, '_lagrange') and collection._lagrange):
                collection.remove()
        for collection in self.ax_3d.collections[:]:
            if not (hasattr(collection, '_lagrange') and collection._lagrange):
                collection.remove()
                
        system = self.system_combo.currentText()
        if system == "Sun_Earth":
            mu = 3.0542e-6
        elif system == "Earth_Moon":
            mu = 1.215058560962404e-2        
        perform_continuation = self.ci_checkbox.isChecked()
        lagrange_text = self.lagrange_combo.currentText()
        lagrange_point_index = int(lagrange_text[1])
        scheme_name = self.scheme_combo.currentText()
        temporal_scheme = self.scheme_dict[scheme_name]
        perform_stability = self.stability_checkbox.isChecked()
        kwargs = {}
    
        # Mostrar en la etiqueta de estado (opcional)
        label_text = f"Modo: {'CIs' if perform_continuation else 'Familia de órbitas'}, Esquema: {scheme_name}, Punto de Lagrange: L{lagrange_point_index}, μ: {mu} ({system})"
        self.status_label.setText(label_text)
        # --- Verificar que haya una carpeta y archivo seleccionado ---
        if not hasattr(self, 'selected_ci_folder'):
            self.status_label.setText("Error: selecciona una carpeta primero.")
            return

        if self.ci_files_combo.count() == 0 or self.ci_files_combo.currentText().startswith("No se encontraron"):
            self.status_label.setText("Error: selecciona un archivo válido.")
            return
        
        filename = self.ci_files_combo.currentText()
        full_path = os.path.join(self.selected_ci_folder, filename)
    
        if perform_continuation:
            print("Generando nueva familia...")
            dx0 = self.dx_spin.value()
            n_orbits = self.n_spin.value()
            U0_group = cargar_CIs_principales(full_path)
            self.T_fam = zeros((U0_group.shape[0], n_orbits))
            self.V0_fam = zeros((U0_group.shape[0], n_orbits, 42))
            for idx, U0 in enumerate(U0_group):
                print(f"\n--- CI #{idx+1} ---")
                V0 = build_initial_condition(U0)
                T_half0 = estimate_T_half_0(mu, lagrange_point_index)
                self.V0_fam[idx], self.T_fam[idx] = Lyapunov_family(V0, mu, temporal_scheme, n_orbits, dx0, T_half0, full_path, **kwargs)                
            print("Familia generada.")
            lagrange_text = self.lagrange_combo.currentText()
            lagrange_point_index = int(lagrange_text[1]) - 1
            plot_Lyapunov_family_GUI(self.ax_2d, self.ax_3d, mu, temporal_scheme, U0_group.shape[0], n_orbits, self.V0_fam, self.T_fam,
                                     lagrange_point_index, **kwargs)
            self.canvas_2d.draw()
            self.canvas_3d.draw()
            # plot_Lyapunov_family(
            # mu, temporal_scheme, U0_group.shape[0], n_orbits, V0_fam, T_fam,
            # ax_2d=self.ax_2d, ax_3d=self.ax_3d, **kwargs
            # )
        else:
            print("Analizando familia existente...")
            try:
                lagrange_text = self.lagrange_combo.currentText()
                lagrange_point_index = int(lagrange_text[1]) - 1
                periods, V0_Lyap_family = cargar_CIs(full_path)
                plot_Lyapunov_family_one_source_GUI(self.ax_2d, self.ax_3d, mu, temporal_scheme, V0_Lyap_family, periods, lagrange_point_index, **kwargs)
                #self.plot_Lyapunov_family_one_source_GUI(mu, temporal_scheme, V0_Lyap_family, periods, **kwargs)
                self.canvas_2d.draw()
                self.canvas_3d.draw()
                print(f"Cargadas {len(periods)} órbitas.")
            except Exception as e:
                print(f"Error al cargar {filename}: {e}")
                exit()
            
            if perform_stability:
                # --- CÁLCULO DE ESTABILIDAD (soporta ambos modos) ---
                S1, S2 = [], []
                for i in range(len(periods)):
                    if not any(V0_Lyap_family[i]): continue
                    T = periods[i]

                    if isinstance(temporal_scheme, str):
                        sol = solve_ivp(
                            CRTBP_variacional_JPL, (0, T), V0_Lyap_family[i],
                            method=temporal_scheme, args=(mu,),
                            rtol=kwargs.get("rtol", 1e-10),
                            atol=kwargs.get("atol", 1e-12),
                            dense_output=True
                        )
                        V_T = sol.sol(T)
                    else: 
                        Nt = 1000          
                        t_eval = linspace(0, T, Nt)            
                        U = Cauchy_problem(CRTBP_variacional_JPL, V0_Lyap_family[i], mu, t_eval, temporal_scheme)
                        V_T = U[Nt-1]

                    s1, s2 = stability(V_T)
                    S1.append(s1); S2.append(s2)
                    print(f"[{i+1}] T={T:.4f} | s1={s1:.2e} | s2={s2:.2e}")

                plot_stability(periods, S1, S2)
                print("\n✔ Análisis completado.")