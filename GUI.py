import sys
from Widgets_Callbacks import *
from Config_plotting import *

config_plotting()
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Establecer fuente global para todos los widgets de PyQt6
    font = app.font()
    font.setFamily("Times New Roman")
    font.setPointSize(12)
    app.setFont(font)
    window = Lyapunov_GUI()
    window.show()
    sys.exit(app.exec())