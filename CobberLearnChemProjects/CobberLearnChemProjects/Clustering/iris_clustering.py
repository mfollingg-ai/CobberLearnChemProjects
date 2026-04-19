import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load dataset
# -------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# -------------------------
# Scale features (important)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)

# -------------------------
# K-Means clustering
# -------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

X_scaled["cluster"] = clusters
X_scaled["true_species"] = y

# -------------------------
# Create plot
# -------------------------
plt.figure(figsize=(8, 6))

colors = ["purple", "green", "orange"]
labels = ["Cluster 0 (likely Setosa)", "Cluster 1 (Versicolor)", "Cluster 2 (Virginica)"]

for i in range(3):
    plt.scatter(
        X_scaled[X_scaled["cluster"] == i]["petal length (cm)"],
        X_scaled[X_scaled["cluster"] == i]["petal width (cm)"],
        c=colors[i],
        label=labels[i]
    )

# -------------------------
# Axis labels + title
# -------------------------
plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("Iris Clustering Using K-Means")

# -------------------------
# REAL LEGEND (KEY)
# -------------------------
plt.legend(title="Cluster Key")

plt.grid(True)
plt.show()

# -------------------------
# Comparison table
# -------------------------
print("\nCluster vs True Species:")
print(pd.crosstab(y, clusters))