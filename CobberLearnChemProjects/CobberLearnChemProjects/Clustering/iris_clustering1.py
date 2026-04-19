import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from matplotlib.widgets import Slider

# -------------------------
# Load data
# -------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)

# -------------------------
# Initial clustering
# -------------------------
k_init = 3
kmeans = KMeans(n_clusters=k_init, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# -------------------------
# Create figure
# -------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

scatter = ax.scatter(
    X_scaled["petal length (cm)"],
    X_scaled["petal width (cm)"],
    c=clusters,
    cmap="viridis"
)

ax.set_xlabel("Petal Length (scaled)")
ax.set_ylabel("Petal Width (scaled)")
title = ax.set_title(f"K-Means Clustering (k = {k_init})")

# -------------------------
# Slider axis
# -------------------------
ax_k = plt.axes([0.2, 0.1, 0.6, 0.03])
k_slider = Slider(ax_k, "Clusters (k)", 2, 8, valinit=k_init, valstep=1)

# -------------------------
# Update function
# -------------------------
def update(val):
    k = int(k_slider.val)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    ax.clear()

    ax.scatter(
        X_scaled["petal length (cm)"],
        X_scaled["petal width (cm)"],
        c=clusters,
        cmap="viridis"
    )

    ax.set_xlabel("Petal Length (scaled)")
    ax.set_ylabel("Petal Width (scaled)")
    ax.set_title(f"K-Means Clustering (k = {k})")

    fig.canvas.draw_idle()

k_slider.on_changed(update)

plt.show()