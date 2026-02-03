import os
import seaborn as sns        # Load Titanic dataset
import pandas as pd          # DataFrame manipulation
import numpy as np           # Numerical calculations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns        # For heatmap plotting
from sklearn.impute import KNNImputer  # KNN imputation
from sklearn.metrics import mean_absolute_error  # MAE metric
from sklearn.ensemble import RandomForestRegressor  # Random forest for imputation

# Create folder for saving plots if it doesn't exist
if not os.path.exists('MakingDataWhole'):
    os.makedirs('MakingDataWhole')

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Preview first 10 rows
print(titanic.head(10))

# Check missing values in 'age'
print("\nMissing values in 'age':", titanic['age'].isnull().sum())

# Mean imputation
mean_age = titanic['age'].mean()
print(f"\nMean age (excluding missing): {mean_age:.2f}")

titanic['age_filled_mean'] = titanic['age'].fillna(mean_age)
print("Missing values after mean imputation:", titanic['age_filled_mean'].isnull().sum())

# Correlation matrix
numeric_data = titanic.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

print("\nCorrelations with Age:")
print(corr_matrix['age'].sort_values(ascending=False))

top_two_features = corr_matrix['age'].drop('age').abs().sort_values(ascending=False).head(2).index.tolist()
print("\nTop two features correlated with age:", top_two_features)

# Save correlation heatmap with improved styling
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            cbar_kws={'shrink': 0.8}, linewidths=0.5)
plt.title("Correlation Matrix - Titanic Dataset", fontsize=16)
plt.tight_layout()
plt.savefig("MakingDataWhole/correlation_matrix.png")
plt.show()  # Show plot window
plt.close()

# KNN imputation
knn_features = numeric_data[top_two_features + ['age']]
imputer = KNNImputer(n_neighbors=3)
knn_imputed = imputer.fit_transform(knn_features)
imputed_ages = knn_imputed[:, -1]
titanic['age_knn_imputed'] = imputed_ages

known_mask = ~titanic['age'].isnull()

plt.figure(figsize=(8, 6))
plt.scatter(titanic.loc[known_mask, 'age'], titanic.loc[known_mask, 'age_knn_imputed'],
            alpha=0.7, label='Data points')
plt.plot([0, 80], [0, 80], 'r--', label='Ideal fit (y = x)')
plt.xlabel("Actual Age", fontsize=14)
plt.ylabel("KNN Imputed Age", fontsize=14)
plt.title("Actual vs KNN-Imputed Age (Known Ages)", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("MakingDataWhole/actual_vs_knn_imputed_age.png")
plt.show()  # Show plot window
plt.close()

mae_knn = mean_absolute_error(titanic.loc[known_mask, 'age'], titanic.loc[known_mask, 'age_knn_imputed'])

print("\n--- Titanic Dataset Missing Ages and Imputation Summary ---")
print(f"Total missing values in 'age': {titanic['age'].isnull().sum()}")
print(f"Mean age (computed from known values): {mean_age:.2f}")
print(f"Missing values after mean imputation: {titanic['age_filled_mean'].isnull().sum()}")
print(f"MAE of KNN imputation on known ages: {mae_knn:.3f}")
print(f"Average age before KNN imputation: {titanic['age'].mean():.2f}")
print(f"Average age after KNN imputation: {titanic['age_knn_imputed'].mean():.2f}")

# Random Forest imputation
rf_data = titanic[['age', 'pclass', 'fare', 'sibsp', 'parch']].copy()
known_ages = rf_data[rf_data['age'].notnull()]
unknown_ages = rf_data[rf_data['age'].isnull()]
X_train = known_ages.drop('age', axis=1)
y_train = known_ages['age']

rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

X_missing = unknown_ages.drop('age', axis=1)
predicted_ages = rf.predict(X_missing)
titanic.loc[titanic['age'].isnull(), 'age_rf_imputed'] = predicted_ages
titanic.loc[titanic['age'].notnull(), 'age_rf_imputed'] = titanic.loc[titanic['age'].notnull(), 'age']

mae_rf = mean_absolute_error(y_train, rf.predict(X_train))

print(f"MAE of Random Forest on training data: {mae_rf:.3f}")
print(f"Average age before RF imputation: {titanic['age'].mean():.2f}")
print(f"Average age after RF imputation: {titanic['age_rf_imputed'].mean():.2f}")
print("--------------------------------------------------------------\n")

plt.figure(figsize=(8, 6))
plt.scatter(y_train, rf.predict(X_train), alpha=0.7, label='Data points')
plt.plot([0, 80], [0, 80], 'r--', label='Ideal fit (y = x)')
plt.xlabel("Actual Age", fontsize=14)
plt.ylabel("RF Predicted Age", fontsize=14)
plt.title("Random Forest: Actual vs Predicted Age", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("MakingDataWhole/rf_actual_vs_predicted_age.png")
plt.show()  # Show plot window
plt.close()

