import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Data
data = {
    'Molecular Weight': [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    'Hydrogen Bond Donors': [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    'Hydrogen Bond Acceptors': [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    'Water Solubility': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)
y = df['Water Solubility']

# Output directory for plots
output_dir = 'DecisionTreeClassifier'
os.makedirs(output_dir, exist_ok=True)


def train_and_plot(features, max_depth=None):
    X = df[features]
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)

    plt.figure(figsize=(12, 8))
    plot_tree(
        clf,
        feature_names=features,
        class_names=['Not Soluble', 'Soluble'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    title = f"Decision Tree trained on {', '.join(features)}"
    if max_depth:
        title += f" (max depth={max_depth})"
    plt.title(title)
    plt.tight_layout()

    filename = f"decision_tree_{'_'.join(f.lower().replace(' ', '_') for f in features)}"
    if max_depth:
        filename += f"_maxdepth_{max_depth}"
    filename += ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Saved tree plot at: {filepath}")

    plt.show()


def main():
    print("Choose feature(s) to train the Decision Tree on:")
    print("1: Molecular Weight")
    print("2: Hydrogen Bond Donors")
    print("3: Hydrogen Bond Acceptors")
    print("4: All features")

    choice = input("Enter choice number (1-4): ").strip()

    if choice == '1':
        features = ['Molecular Weight']
    elif choice == '2':
        features = ['Hydrogen Bond Donors']
    elif choice == '3':
        features = ['Hydrogen Bond Acceptors']
    elif choice == '4':
        features = ['Molecular Weight', 'Hydrogen Bond Donors', 'Hydrogen Bond Acceptors']
    else:
        print("Invalid choice, defaulting to all features.")
        features = ['Molecular Weight', 'Hydrogen Bond Donors', 'Hydrogen Bond Acceptors']

    max_depth_input = input("Enter max depth for tree (press Enter for no limit): ").strip()
    max_depth = int(max_depth_input) if max_depth_input.isdigit() else None

    train_and_plot(features, max_depth)


if __name__ == "__main__":
    main()






