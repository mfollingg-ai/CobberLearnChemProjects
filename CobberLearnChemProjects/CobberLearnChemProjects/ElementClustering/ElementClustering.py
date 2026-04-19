import sys
import os
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.cluster import KMeans


class ClusterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Element Clustering with KMeans")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Enter k (number of clusters):"))

        self.k_input = QLineEdit()
        self.k_input.setFixedWidth(50)
        self.k_input.setPlaceholderText("e.g. 3")
        input_layout.addWidget(self.k_input)

        self.run_button = QPushButton("Run Clustering")
        input_layout.addWidget(self.run_button)
        layout.addLayout(input_layout)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'group_1_elements.csv')
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load CSV file:\n{e}")
            sys.exit(1)

        # Debug: print columns to console
        print("CSV Columns:", self.df.columns.tolist())

        # Check for possible element name columns
        possible_names = ["Element", "Name", "Symbol", "Element Name"]
        self.element_names = None
        for col in possible_names:
            if col in self.df.columns:
                self.element_names = self.df[col]
                break

        if self.element_names is None:
            QMessageBox.critical(self, "Error", "CSV must have a column with element names (e.g., 'Element', 'Name', or 'Symbol').")
            sys.exit(1)

        # Ensure the feature columns exist
        required_features = ["Atomic Radius (pm)", "First Ionization Energy (kJ/mol)"]
        for feature in required_features:
            if feature not in self.df.columns:
                QMessageBox.critical(self, "Error", f"CSV must have a '{feature}' column.")
                sys.exit(1)

        self.X = self.df[required_features]

        self.run_button.clicked.connect(self.run_clustering)

    def run_clustering(self):
        k_text = self.k_input.text().strip()
        if not k_text.isdigit():
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for k.")
            return

        k = int(k_text)
        if k < 1 or k > len(self.df):
            QMessageBox.warning(self, "Invalid k", f"k must be between 1 and {len(self.df)}")
            return

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.X)
        centers = kmeans.cluster_centers_

        self.df["Cluster"] = clusters

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        scatter = ax.scatter(
            self.df["Atomic Radius (pm)"],
            self.df["First Ionization Energy (kJ/mol)"],
            c=self.df["Cluster"],
            cmap="viridis",
            s=90,
            edgecolors="black"
        )

        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c="black",
            marker="X",
            s=350,
            label="Cluster Centers"
        )

        ax.set_xlabel("Atomic Radius (pm)")
        ax.set_ylabel("First Ionization Energy (kJ/mol)")
        ax.set_title(f"Element Clustering (k={k})")
        ax.legend()

        # Add labels for each point with a small offset
        for i, element in enumerate(self.element_names):
            x = self.df.loc[i, "Atomic Radius (pm)"]
            y = self.df.loc[i, "First Ionization Energy (kJ/mol)"]
            ax.text(x + 1, y + 1, element, fontsize=8, alpha=0.75)

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClusterApp()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())
