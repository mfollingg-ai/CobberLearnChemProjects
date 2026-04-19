import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, Superimposer, PDBIO
from tkinter import Tk, filedialog
import os

# -----------------------------
# FILE SELECTOR
# -----------------------------
def select_file(title):
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("PDB files", "*.pdb")])

# -----------------------------
# LOAD STRUCTURE
# -----------------------------
def load_structure(path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(path), path)

# -----------------------------
# BASIC STATS
# -----------------------------
def structure_stats(structure):
    atoms = 0
    residues = 0
    chains = set()

    for model in structure:
        for chain in model:
            chains.add(chain.id)
            for residue in chain:
                residues += 1
                for atom in residue:
                    atoms += 1

    return atoms, residues, len(chains)

# -----------------------------
# CA ATOMS
# -----------------------------
def get_ca_atoms(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atoms.append(residue["CA"])
    return atoms

# -----------------------------
# pLDDT EXTRACTION
# -----------------------------
def get_plddt(structure):
    return np.array([
        residue["CA"].get_bfactor()
        for model in structure
        for chain in model
        for residue in chain
        if "CA" in residue
    ])

# -----------------------------
# ALIGNMENT + RMSD
# -----------------------------
def align_structures(ref_atoms, mob_atoms):
    n = min(len(ref_atoms), len(mob_atoms))
    ref_atoms = ref_atoms[:n]
    mob_atoms = mob_atoms[:n]

    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    sup.apply(mob_atoms)

    return sup.rms, ref_atoms, mob_atoms

# -----------------------------
# DISTANCES
# -----------------------------
def residue_distances(ref_atoms, mob_atoms):
    return np.array([
        np.linalg.norm(r.coord - m.coord)
        for r, m in zip(ref_atoms, mob_atoms)
    ])

# -----------------------------
# WORST REGION FINDER
# -----------------------------
def find_worst_region(distances, window=3):
    best_score = 0
    best_start = 0

    for i in range(len(distances) - window):
        score = np.mean(distances[i:i+window])
        if score > best_score:
            best_score = score
            best_start = i

    return best_start, best_start + window

# -----------------------------
# PLOTS
# -----------------------------
def plot_distances(dist):
    plt.figure()
    plt.plot(dist)
    plt.title("Per-Residue Distance")
    plt.xlabel("Residue Index")
    plt.ylabel("Å")

def plot_plddt(plddt):
    plt.figure()
    plt.plot(plddt)
    plt.title("AlphaFold pLDDT Confidence")
    plt.xlabel("Residue Index")
    plt.ylabel("Score")
    plt.ylim(0, 100)

def plot_correlation(dist, plddt):
    plt.figure()
    plt.scatter(plddt[:len(dist)], dist, alpha=0.6)
    plt.title("pLDDT vs Structural Deviation")
    plt.xlabel("pLDDT")
    plt.ylabel("Distance (Å)")

def plot_heatmap_like(dist):
    matrix = np.abs(np.subtract.outer(dist, dist))
    plt.figure()
    plt.imshow(matrix, cmap="viridis")
    plt.title("Distance Matrix (Residue Variation)")
    plt.colorbar()

# -----------------------------
# SAVE PDB
# -----------------------------
def save_pdb(structure, filename):
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)

# -----------------------------
# PY M O L
# -----------------------------
def generate_pymol(exp_file, af_file):
    script = f"""
load {exp_file}, experimental
load {af_file}, alphafold

align alphafold, experimental

color blue, experimental
spectrum b, red_white_blue, alphafold, minimum=0, maximum=100

show cartoon
bg_color white
"""
    with open("visualize_alignment.pml", "w") as f:
        f.write(script)

# -----------------------------
# NGL VIEW (confidence coloring)
# -----------------------------
def generate_ngl(exp_file, af_file, plddt):

    html = f"""
<!DOCTYPE html>
<html>
<head>
<script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
</head>
<body>

<div id="viewport" style="width:800px;height:600px;"></div>

<script>
var stage = new NGL.Stage("viewport");

stage.loadFile("{exp_file}").then(function(o) {{
    o.addRepresentation("cartoon", {{color: "blue"}});
    o.autoView();
}});

stage.loadFile("{af_file}").then(function(o) {{

    var plddt = {plddt.tolist()};

    var scheme = NGL.ColormakerRegistry.addScheme(function(params) {{
        this.atomColor = function(atom) {{
            var i = atom.residue.index;
            var score = plddt[i % plddt.length];

            if (score < 50) return 0xff0000;
            if (score < 80) return 0xffa500;
            return 0x0000ff;
        }};
    }});

    o.addRepresentation("cartoon", {{color: scheme}});
    o.autoView();
}});
</script>

</body>
</html>
"""
    with open("structure_view.html", "w") as f:
        f.write(html)

# -----------------------------
# MAIN
# -----------------------------
def main():

    exp_path = select_file("Experimental PDB")
    af_path = select_file("AlphaFold PDB")

    exp = load_structure(exp_path)
    af = load_structure(af_path)

    # stats
    exp_stats = structure_stats(exp)
    af_stats = structure_stats(af)

    print("\nExperimental:", exp_stats)
    print("AlphaFold:", af_stats)

    # CA atoms
    exp_ca = get_ca_atoms(exp)
    af_ca = get_ca_atoms(af)

    # RMSD
    rmsd, exp_ca, af_ca = align_structures(exp_ca, af_ca)
    print(f"\nRMSD: {rmsd:.3f} Å")

    # distances + pLDDT
    dist = residue_distances(exp_ca, af_ca)
    plddt = get_plddt(af)

    # worst region
    start, end = find_worst_region(dist)
    print(f"Worst region: {start}-{end}")

    # plots
    plot_distances(dist)
    plot_plddt(plddt)
    plot_correlation(dist, plddt)
    plot_heatmap_like(dist)

    plt.show()

    # save
    save_pdb(exp, "experimental_aligned.pdb")
    save_pdb(af, "alphafold_aligned.pdb")

    # visuals
    generate_pymol("experimental_aligned.pdb", "alphafold_aligned.pdb")
    generate_ngl("experimental_aligned.pdb", "alphafold_aligned.pdb", plddt)

    print("\nDONE:")
    print("- aligned PDBs")
    print("- PyMOL script")
    print("- NGL HTML viewer")

if __name__ == "__main__":
    main()