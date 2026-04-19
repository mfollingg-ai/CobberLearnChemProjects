# Linear Regression vs Neural Network Boiling Point Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Step 1: Create DataFrame
# -----------------------------
data = {
    "Compound": [
        "Methane", "Water", "Propane", "Ethanol", "Formic Acid",
        "Acetic Acid", "Butane", "Acetone", "Benzene", "Toluene", "Octane"
    ],
    "MW": [16, 18, 44, 46, 46, 60, 58, 58, 78, 92, 114],
    "BoilingPoint": [-161, 100, -42, 78, 101, 118, -1, 56, 80, 111, 125]
}

df = pd.DataFrame(data)

X = df[["MW"]]
y = df["BoilingPoint"]

print("\nDataset:\n")
print(df)

# -----------------------------
# Step 2: Linear Regression
# -----------------------------
linear_model = LinearRegression()
linear_model.fit(X, y)

linear_predictions = linear_model.predict(X)

lin_mae = mean_absolute_error(y, linear_predictions)
lin_mse = mean_squared_error(y, linear_predictions)
lin_r2 = r2_score(y, linear_predictions)

# -----------------------------
# Step 3: Neural Network
# -----------------------------
nn_model = MLPRegressor(
    hidden_layer_sizes=(10,10),
    activation="relu",
    max_iter=5000,
    early_stopping=False,
    random_state=42
)

nn_model.fit(X, y)

nn_predictions = nn_model.predict(X)

nn_mae = mean_absolute_error(y, nn_predictions)
nn_mse = mean_squared_error(y, nn_predictions)
nn_r2 = r2_score(y, nn_predictions)

# -----------------------------
# Display Metrics
# -----------------------------
print("\nModel Performance Metrics\n")

print("Linear Regression")
print("-----------------")
print("MAE:", lin_mae)
print("MSE:", lin_mse)
print("R2:", lin_r2)

print("\nNeural Network")
print("-----------------")
print("MAE:", nn_mae)
print("MSE:", nn_mse)
print("R2:", nn_r2)

print("\nNeural Network Epochs Used:", nn_model.n_iter_)

# -----------------------------
# Plot Data and Both Models
# -----------------------------
sorted_df = df.sort_values("MW")
X_sorted = sorted_df[["MW"]]

lin_line = linear_model.predict(X_sorted)
nn_line = nn_model.predict(X_sorted)

plt.scatter(df["MW"], df["BoilingPoint"], label="Actual Data")
plt.plot(X_sorted, lin_line, label="Linear Regression")
plt.plot(X_sorted, nn_line, label="Neural Network")

plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point")
plt.title("Model Comparison")
plt.legend()

plt.savefig("model_comparison.png")
plt.show()
