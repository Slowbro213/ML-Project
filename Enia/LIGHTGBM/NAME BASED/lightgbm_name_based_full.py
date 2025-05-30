# -*- coding: utf-8 -*-
"""LIGHTGBM-NAME BASED-FULL

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zIWovsJI1d7nODb8EyYhWmE5tQLCcWsx
"""

# 📌 1. Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from google.colab import files

# 📌 2. Load dataset
uploaded = files.upload()
df = pd.read_csv("wfp_food_prices_database.csv", low_memory=False)

# 📌 3. Initial cleanup
df = df.drop_duplicates()
df = df.dropna(subset=['mp_price'])

# 📌 4. Outlier removal on target (mp_price)
Q1 = df['mp_price'].quantile(0.25)
Q3 = df['mp_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.0 * IQR
upper_bound = Q3 + 1.0 * IQR
df = df[(df['mp_price'] >= lower_bound) & (df['mp_price'] <= upper_bound)]

# 📌 5. Cyclical encoding for month
df['mp_month'] = df['mp_month'].astype(int)
df['mp_month_sin'] = np.sin(2 * np.pi * (df['mp_month'] - 1) / 12)
df['mp_month_cos'] = np.cos(2 * np.pi * (df['mp_month'] - 1) / 12)

# 📌 6. Optional: Region mean price feature
df["adm1_mean_price"] = df.groupby("adm1_id")["mp_price"].transform("mean")

# 📌 7. Define features and target
features = [
    'cm_name', 'adm0_name', 'adm1_name', 'cur_name', 'pt_name', 'um_name',
    'mp_month_sin', 'mp_month_cos', 'mp_year', 'adm1_mean_price'
]
target = 'mp_price'

# 📌 8. Drop rows with missing in final feature set
df = df.dropna(subset=features + [target])

# 📌 9. Encode categorical features
X = df[features].copy()
y = df[target]

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 📌 10. Train-test split (full dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 11. Train LightGBM using best setup (Setup B)
model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# 📌 12. Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("🏁 FINAL EVALUATION — LightGBM (100% dataset, Setup B):")
print("MAE: ", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²:  ", round(r2, 3))

import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Scatter Plot: True vs Predicted Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y = x line
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("True vs Predicted Prices (LightGBM Model)")
plt.grid(True)
plt.show()

# 2️⃣ Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=50, color='orange')
plt.title("Distribution of Residuals (LightGBM Model)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 3️⃣ Feature Importance Plot
importances = model.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance (LightGBM Model)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.hexbin(y_test, y_pred, gridsize=60, cmap="Blues", bins='log')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Hexbin Plot: True vs Predicted Prices (LightGBM Model)")
plt.colorbar(label="log10(N points)")
plt.xlim(0, 1200)
plt.ylim(0, 1200)
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

errors = np.abs(y_test - y_pred)  # Absolute error

plt.figure(figsize=(8, 6))
sc = plt.scatter(y_test, y_pred, c=errors, cmap='coolwarm', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect Prediction')

plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Prediction Error Coloring: Closer to Line = More Accurate")
plt.colorbar(sc, label="Absolute Error")
plt.legend()
plt.grid(True)
plt.show()