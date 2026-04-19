import sys
import pubchempy as pcp
import requests
from io import BytesIO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QSizePolicy
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class MoleculeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Molecule Viewer")

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Compound name:"))
        self.compound_input = QLineEdit()
        input_layout.addWidget(self.compound_input)
        self.fetch_button = QPushButton("Fetch Data")
        input_layout.addWidget(self.fetch_button)
        main_layout.addLayout(input_layout)

        # Output: text data
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.text_output)

        # Output: image
        self.image_label = QLabel("Molecule image will appear here")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedHeight(300)
        main_layout.addWidget(self.image_label)

        # Connect button
        self.fetch_button.clicked.connect(self.fetch_and_display)

    def fetch_and_display(self):
        compound_name = self.compound_input.text().strip()
        if not compound_name:
            self.text_output.setText("Please enter a compound name.")
            return

        # Fetch compound from PubChem
        compounds = pcp.get_compounds(compound_name, "name")
        if not compounds:
            self.text_output.setText(f"No compound found for '{compound_name}'.")
            self.image_label.setText("No image available")
            return

        compound = compounds[0]

        # Fingerprint-style data
        fingerprint_data = [
            ("Molecular Formula", compound.molecular_formula),
            ("Molecular Weight (g/mol)", compound.molecular_weight),
            ("SMILES", compound.connectivity_smiles),
            ("Heavy Atom Count", compound.heavy_atom_count),
            ("H-Bond Donors", compound.h_bond_donor_count),
            ("H-Bond Acceptors", compound.h_bond_acceptor_count),
            ("Rotatable Bonds", compound.rotatable_bond_count),
            ("TPSA", compound.tpsa),
        ]

        # Build text table
        text_lines = [f"{compound_name} – Molecular Fingerprint", "=" * 50]
        for label, value in fingerprint_data:
            text_lines.append(f"{label:<25} : {value}")
        text_lines.append("=" * 50)
        self.text_output.setText("\n".join(text_lines))

        # Fetch 2D structure image from PubChem
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/PNG"
            response = requests.get(url)
            response.raise_for_status()
            image_data = BytesIO(response.content)
            pixmap = QPixmap()
            pixmap.loadFromData(image_data.read())
            self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        except Exception as e:
            self.image_label.setText(f"Could not load image: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MoleculeViewer()
    window.resize(600, 700)
    window.show()
    sys.exit(app.exec())

