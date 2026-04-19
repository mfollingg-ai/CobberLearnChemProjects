from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def calculate_descriptors(smiles: str):
    """
    Given a SMILES string, generate an RDKit molecule object
    and calculate several chemical descriptors.

    Returns a dictionary of descriptor names and values.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = {
        "Exact Molecular Weight": Descriptors.ExactMolWt(mol),
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "logP (hydrophobicity)": Descriptors.MolLogP(mol),
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol)
    }

    return descriptors


def print_descriptors(smiles: str, descriptors: dict):
    """
    Nicely prints a table of descriptors for a given SMILES string.
    """
    print("\n" + "=" * 60)
    print(f"Molecule: {smiles}")
    print("-" * 60)
    for name, value in descriptors.items():
        if isinstance(value, float):
            print(f"{name:<25} : {value:.2f}")
        else:
            print(f"{name:<25} : {value}")
    print("=" * 60)


def main():
    """
    Main function to repeatedly prompt the user for SMILES strings
    and calculate descriptors until they choose to quit.
    """
    print("RDKit Molecular Descriptor Calculator")
    print("Type 'quit' to exit.\n")

    while True:
        smiles = input("Enter a SMILES string: ").strip()
        if smiles.lower() in ["quit", "exit"]:
            print("Exiting program.")
            break

        descriptors = calculate_descriptors(smiles)
        if descriptors is None:
            print(f"Invalid SMILES string: {smiles}")
        else:
            print_descriptors(smiles, descriptors)


if __name__ == "__main__":
    main()
