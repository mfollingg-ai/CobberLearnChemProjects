import sys
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class AlkanePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alkane Boiling Point Fitting")

        # Data: first 10 linear alkanes
        self.carbons = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        self.boiling_points = np.array(
            [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.6, 150.8, 174.1],
            dtype=float
        )

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Controls
        controls = QHBoxLayout()
        main_layout.addLayout(controls)

        controls.addWidget(QLabel("Fit function:"))

        self.fit_selector = QComboBox()
        self.fit_selector.addItems([
            "None (Data Only)",
            "Linear: y = ax + b",
            "Quadratic: y = ax² + bx + c",
            "Exponential: y = a·e^(bx)"
        ])
        controls.addWidget(self.fit_selector)

        self.plot_button = QPushButton("Plot / Fit")
        controls.addWidget(self.plot_button)

        controls.addStretch()

        # Matplotlib Figure
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)

        self.plot_button.clicked.connect(self.update_plot)

        # Initial plot
        self.update_plot()

    def update_plot(self):
        self.ax.clear()

        # Scatter plot of the data
        self.ax.scatter(self.carbons, self.boiling_points, s=60)
        self.ax.set_title("Boiling Point vs. Number of Carbons")
        self.ax.set_xlabel("Number of Carbons")
        self.ax.set_ylabel("Boiling Point (°C)")

        x_dense = np.linspace(self.carbons.min(), self.carbons.max(), 300)

        choice = self.fit_selector.currentText()

        if "Linear" in choice:
            coeffs = np.polyfit(self.carbons, self.boiling_points, 1)
            y_fit = np.polyval(coeffs, x_dense)
            self.ax.plot(x_dense, y_fit, label="Linear Fit")
            self.ax.legend()

        elif "Quadratic" in choice:
            coeffs = np.polyfit(self.carbons, self.boiling_points, 2)
            y_fit = np.polyval(coeffs, x_dense)
            self.ax.plot(x_dense, y_fit, label="Quadratic Fit")
            self.ax.legend()

        elif "Exponential" in choice:
            # Fit y = a * exp(bx) by linearizing: ln(y) = ln(a) + bx
            # Only use points with positive boiling points for this fit
            mask = self.boiling_points > 0
            x = self.carbons[mask]
            y = self.boiling_points[mask]

            ln_y = np.log(y)
            b, ln_a = np.polyfit(x, ln_y, 1)
            a = np.exp(ln_a)

            y_fit = a * np.exp(b * x_dense)
            self.ax.plot(x_dense, y_fit, label="Exponential Fit")
            self.ax.legend()

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AlkanePlotter()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())

