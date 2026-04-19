import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Generate Noisy Data
# -----------------------------

np.random.seed(42)

# x values from 0 to 10
X = np.linspace(0, 10, 50)

# True relationship
true_slope = 2
true_intercept = 5

# Add noise
noise = np.random.normal(0, 2, size=X.shape)
y = true_slope * X + true_intercept + noise

# Plot noisy data + true line
plt.scatter(X, y, label="Noisy Data")
plt.plot(X, true_slope * X + true_intercept, label="True Line", linewidth=3)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Noisy Data with True Line")
plt.legend()
plt.savefig("true_vs_noisy_data.png")
plt.show()


# -----------------------------
# 2. Train Linear Regression Model
# -----------------------------

# reshape X for sklearn
X_reshaped = X.reshape(-1, 1)

model = LinearRegression()

model.fit(X_reshaped, y)

# predictions
y_pred = model.predict(X_reshaped)

# Plot model prediction
plt.scatter(X, y, label="Noisy Data")
plt.plot(X, true_slope * X + true_intercept, label="True Line", linewidth=3)
plt.plot(X, y_pred, label="Model Best Fit", linestyle="--")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Fit")
plt.legend()
plt.savefig("regression_fit.png")
plt.show()


# -----------------------------
# 3. Print Model Results
# -----------------------------

print("True slope:", true_slope)
print("True intercept:", true_intercept)

print("Learned slope:", model.coef_[0])
print("Learned intercept:", model.intercept_)


# -----------------------------
# 4. MSE Function
# -----------------------------

def calculate_mse(slope, intercept):

    predictions = slope * X + intercept
    mse = np.mean((y - predictions) ** 2)

    print(f"Slope={slope}, Intercept={intercept}, MSE={mse}")
    return mse


# Example test
calculate_mse(1.5, 4)
calculate_mse(2, 5)


# -----------------------------
# 5. Build Loss Landscape
# -----------------------------

slope_values = np.linspace(0, 4, 100)
intercept_values = np.linspace(0, 10, 100)

loss = np.zeros((len(slope_values), len(intercept_values)))

for i, m in enumerate(slope_values):
    for j, b in enumerate(intercept_values):

        predictions = m * X + b
        loss[i, j] = np.mean((y - predictions) ** 2)


# -----------------------------
# 6. Plot Loss Landscape
# -----------------------------

plt.figure()

plt.imshow(
    loss,
    extent=[intercept_values.min(), intercept_values.max(),
            slope_values.min(), slope_values.max()],
    origin="lower",
    aspect="auto",
    cmap="plasma"   # yellow = low error, purple = high error
)

plt.colorbar(label="MSE Loss")

plt.xlabel("Intercept")
plt.ylabel("Slope")

plt.title("Loss Landscape (MSE)")

plt.savefig("loss_landscape.png")

plt.show()