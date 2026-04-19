import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Create NumPy arrays
# -----------------------------
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# -----------------------------
# 2. Compute residuals
# -----------------------------
residuals = predicted - actual

# -----------------------------
# 3. Error metrics (from scratch)
# -----------------------------
mae = np.mean(np.abs(residuals))
mse = np.mean(residuals ** 2)

# R² calculation from scratch
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((actual - np.mean(actual)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("Manual Calculations:")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R²: {r2:.3f}")

# -----------------------------
# 4. Validate with scikit-learn
# -----------------------------
mae_lib = mean_absolute_error(actual, predicted)
mse_lib = mean_squared_error(actual, predicted)
r2_lib = r2_score(actual, predicted)

print("\nscikit-learn Calculations:")
print(f"MAE: {mae_lib:.3f}")
print(f"MSE: {mse_lib:.3f}")
print(f"R²: {r2_lib:.3f}")

# -----------------------------
# 5. Column display (readability improvement)
# -----------------------------
print("\nActual | Predicted | Residual")
print("-----------------------------")
for a, p, r in zip(actual, predicted, residuals):
    print(f"{a:6} | {p:9} | {r:8.2f}")

# -----------------------------
# 6. Predicted vs Actual Plot
# -----------------------------
plt.figure()
plt.scatter(actual, predicted, s=80)

# Highlight worst prediction (largest absolute residual)
worst_index = np.argmax(np.abs(residuals))
plt.scatter(actual[worst_index], predicted[worst_index], s=120)

plt.plot([actual.min(), actual.max()],
         [actual.min(), actual.max()])

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.close()

# -----------------------------
# 7. Residual Plot
# -----------------------------
plt.figure()
plt.scatter(actual, residuals, s=80)
plt.axhline(0)

plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.close()
