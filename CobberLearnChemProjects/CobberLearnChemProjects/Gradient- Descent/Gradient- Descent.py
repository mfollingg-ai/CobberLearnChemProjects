import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("\n--- Gradient Descent Interactive Lab ---\n")

# -----------------------------
# USER INPUT SECTION
# -----------------------------

true_slope = float(input("Enter the TRUE slope (ex: 2): "))
true_intercept = float(input("Enter the TRUE intercept (ex: 5): "))
noise_level = float(input("Enter noise level (ex: 2): "))

# -----------------------------
# DATA GENERATION
# -----------------------------

np.random.seed(42)

X = np.linspace(0, 10, 50)

noise = np.random.normal(0, noise_level, size=X.shape)

y = true_slope * X + true_intercept + noise

# Plot noisy data
plt.scatter(X, y, label="Noisy Data")

true_line = true_slope * X + true_intercept
plt.plot(X, true_line, label="True Line", linewidth=3)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Noisy Data vs True Line")

plt.legend()

plt.savefig("true_line_plot.png")

plt.show()

# -----------------------------
# TRAIN MODEL
# -----------------------------

X_reshaped = X.reshape(-1, 1)

model = LinearRegression()

model.fit(X_reshaped, y)

y_pred = model.predict(X_reshaped)

# Plot regression result
plt.scatter(X, y, label="Noisy Data")

plt.plot(X, true_line, label="True Line", linewidth=3)

plt.plot(X, y_pred, label="Model Prediction", linestyle="--")

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Model vs True Line")

plt.legend()

plt.savefig("regression_result.png")

plt.show()

# -----------------------------
# PRINT MODEL RESULTS
# -----------------------------

print("\nModel Results")
print("----------------------")

print("True slope:", true_slope)
print("True intercept:", true_intercept)

print("Learned slope:", model.coef_[0])
print("Learned intercept:", model.intercept_)

# -----------------------------
# MSE FUNCTION
# -----------------------------

def calculate_mse(slope, intercept):

    predictions = slope * X + intercept
    mse = np.mean((y - predictions) ** 2)

    return mse


# -----------------------------
# INTERACTIVE MSE TEST
# -----------------------------

print("\nTest different slope/intercept guesses!")

while True:

    choice = input("\nEnter 'test' to try values or 'quit' to continue: ")

    if choice.lower() == "quit":
        break

    guess_slope = float(input("Enter slope guess: "))
    guess_intercept = float(input("Enter intercept guess: "))

    mse = calculate_mse(guess_slope, guess_intercept)

    print("MSE:", mse)

# -----------------------------
# LOSS LANDSCAPE
# -----------------------------

print("\nGenerating Loss Landscape...")

slope_range = np.linspace(true_slope - 3, true_slope + 3, 100)
intercept_range = np.linspace(true_intercept - 5, true_intercept + 5, 100)

loss = np.zeros((len(slope_range), len(intercept_range)))

for i, m in enumerate(slope_range):
    for j, b in enumerate(intercept_range):

        predictions = m * X + b
        loss[i, j] = np.mean((y - predictions) ** 2)

plt.figure()

plt.imshow(
    loss,
    extent=[intercept_range.min(), intercept_range.max(),
            slope_range.min(), slope_range.max()],
    origin="lower",
    aspect="auto",
    cmap="plasma"
)

plt.colorbar(label="MSE")

plt.xlabel("Intercept")
plt.ylabel("Slope")

plt.title("Loss Landscape")

plt.savefig("loss_landscape.png")

plt.show()

print("\nAll plots saved to your project folder.")
print("Experiment again by rerunning the program!")
