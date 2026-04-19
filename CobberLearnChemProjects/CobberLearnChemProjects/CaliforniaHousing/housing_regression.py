import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# =========================
# LOAD DATA
# =========================
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODELS
# =========================
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(random_state=42),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])
}

results = {}

print("\n==============================")
print(" MODEL PERFORMANCE (R² SCORES)")
print("==============================")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    results[name] = score
    print(f"{name}: {score:.4f}")

# =========================
# BEST MODEL
# =========================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n==============================")
print(" BEST MODEL RESULT")
print("==============================")
print(f"Best model: {best_model_name}")
print(f"R² Score: {results[best_model_name]:.4f}")

# =========================
# FEATURE INSIGHT
# =========================
print("\n==============================")
print(" FEATURE INSIGHT")
print("==============================")

corr = df.corr()["MedHouseVal"].sort_values(ascending=False)
print(corr)

print("\nINTERPRETATION:")
print("The strongest predictor of house value is Median Income.")
print("This matches real estate expectations because income strongly affects housing demand and affordability.")

# =========================
# HISTOGRAMS (SAVE + SHOW)
# =========================
print("\nGenerating histograms...")

df.hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("California Housing Feature Distributions")
plt.tight_layout()

plt.savefig("feature_histograms.png")
plt.show()

print("Saved: feature_histograms.png")

# =========================
# CORRELATION HEATMAP (SAVE + SHOW)
# =========================
print("\nGenerating correlation heatmap...")

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=False,
    cmap="coolwarm"
)

plt.title("California Housing Correlation Heatmap")
plt.tight_layout()

plt.savefig("correlation_heatmap.png")
plt.show()

print("Saved: correlation_heatmap.png")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_house(model):
    print("\nEnter house details:")

    values = []

    for col in X.columns:
        val = float(input(f"{col}: "))
        values.append(val)

    values = np.array(values).reshape(1, -1)
    prediction = model.predict(values)[0]

    print(f"\nPredicted house value: ${prediction * 100000:.2f}")

# =========================
# INTERACTIVE LOOP
# =========================
while True:
    choice = input("\nMake a prediction? (yes/no): ").lower()

    if choice == "yes":
        predict_house(best_model)
    else:
        break

print("\nProgram finished successfully.")
