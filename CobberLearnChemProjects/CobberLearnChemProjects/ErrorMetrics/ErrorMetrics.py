import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QSizePolicy
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ElementClusteringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Element Clustering (KMeans)")

        # -----------------------------
        # Load CSV once
        # -----------------------------
        try:
            self.df = pd.read_csv("group_1_elements.csv")
        except FileNotFoundError:
            self.df = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # -----------------------------
        # Input row
        # -----------------------------
        input_layout = QHBoxLayout()

        input_layout.addWidget(QLabel("Enter k value:"))

        self.k_input = QLineEdit()
        self.k_input.setPlaceholderText("Example: 2")
        self.k_input.setText("2")
        input_layout.addWidget(self.k_input)

        self.run_button = QPushButton("Run Clustering")
        input_layout.addWidget(self.run_button)

        main_layout.addLayout(input_layout)

        # -----------------------------
        # Output text box
        # -----------------------------
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.text_output.setFixedHeight(140)
        main_layout.addWidget(self.text_output)

        # -----------------------------
        # Matplotlib Figure inside app
        # -----------------------------
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Connect button
        self.run_button.clicked.connect(self.run_clustering)

        # Run once at startup if data exists
        if self.df is None:
            self.text_output.setText("ERROR: Could not find group_1_elements.csv in this folder.")
        else:
            self.text_output.setText("Dataset loaded successfully.\nEnter a k value and click Run Clustering.")
            self.run_clustering()

    def run_clustering(self):
        if self.df is None:
            return

        # Validate k
        try:
            k = int(self.k_input.text().strip())
            if k < 1:
                raise ValueError
        except ValueError:
            self.text_output.setText("Please enter a valid integer k value (example: 2, 3, 4...).")
            return

        # Grab the exact columns from your CSV
        x_col = "Atomic Radius (pm)"
        y_col = "First Ionization Energy (kJ/mol)"

        X = self.df[[x_col, y_col]]

        # KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        # Output summary
        cluster_counts = pd.Series(clusters).value_counts().sort_index()

        lines = [
            f"Clustering complete (k={k})",
            "-" * 40,
        ]
        for cluster_id, count in cluster_counts.items():
            lines.append(f"Cluster {cluster_id}: {count} elements")

        self.text_output.setText("\n".join(lines))

        # -----------------------------
        # Draw plot
        # -----------------------------
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        scatter = ax.scatter(
            self.df[x_col],
            self.df[y_col],
            c=clusters,
            cmap="viridis",
            s=90,
            edgecolors="black"
        )

        # Cluster centers as big black X
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
        ax.set_title(f"KMeans Element Clustering (k={k})")
        ax.legend()

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ElementClusteringApp()
    window.resize(800, 700)
    window.show()
    sys.exit(app.exec())
