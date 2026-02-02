import numpy as np

# Step 2: Create NumPy arrays
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Step 3: Calculate residuals
residuals = predicted - actual

# Step 4: Error metrics

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(residuals))

# Mean Squared Error (MSE)
mse = np.mean(residuals ** 2)

# R-squared (R^2)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((actual - np.mean(actual)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Print results
print("Actual values:", actual)
print("Predicted values:", predicted)
print("Residuals:", residuals)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R^2):", r_squared)
