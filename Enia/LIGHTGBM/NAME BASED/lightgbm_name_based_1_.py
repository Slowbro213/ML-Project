# -*- coding: utf-8 -*-
"""LIGHTGBM-ID BASED-1%

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KqlmT5EKIjDQ_3Lv60t9od78PS-e-66o
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

# 📌 10. 1% sample for quick tuning
X_sample = X.sample(frac=0.01, random_state=42)
y_sample = y.loc[X_sample.index]
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# 📌 11. Try multiple parameter sets for LightGBM
param_sets = [
    {"label": "A", "n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    {"label": "B", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 6},
    {"label": "C", "n_estimators": 200, "learning_rate": 0.1, "max_depth": 4},
    {"label": "D", "n_estimators": 100, "learning_rate": 0.1, "max_depth": 8},
    {"label": "E", "n_estimators": 150, "learning_rate": 0.07, "max_depth": 5}
]

results = []

for params in param_sets:
    model = lgb.LGBMRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Setup": params["label"],
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3)
    })

# 📌 12. Show comparison
results_df = pd.DataFrame(results)
print("\n🔍 Comparison of LightGBM Parameter Setups on 1% Sample:")
print(results_df.sort_values(by="R2", ascending=False))