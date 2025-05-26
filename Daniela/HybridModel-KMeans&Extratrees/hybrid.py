
#!pip install category_encoders

# ========== 1. Kaggle Setup ==========
#!mkdir -p ~/.kaggle
#!cp kaggle.json ~/.kaggle/
#!chmod 600 ~/.kaggle/kaggle.json

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# Download and unzip dataset
api.dataset_download_files('salehahmedrony/global-food-prices', path='.', unzip=True)

# ========== 2. Imports ==========
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from category_encoders import TargetEncoder

# ========== 3. Load and Clean Data ==========
df = pd.read_csv("wfp_food_prices_database.csv", low_memory=False)

# Drop missing prices
df = df.dropna(subset=['mp_price'])

# Drop irrelevant columns
drop_cols = ["mp_commoditysource", "adm0_name", "adm1_name", "mkt_name", "cur_name", "pt_name", "um_name"]
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# Remove invalid prices
df = df[df['mp_price'] > 0].reset_index(drop=True)

# Remove outliers
Q1 = df['mp_price'].quantile(0.25)
Q3 = df['mp_price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['mp_price'] >= Q1 - 1.5 * IQR) & (df['mp_price'] <= Q3 + 1.5 * IQR)]

# ========== 4. Feature Engineering ==========
df['mp_month_sin'] = np.sin(2 * np.pi * df['mp_month'] / 12)
df['mp_month_cos'] = np.cos(2 * np.pi * df['mp_month'] / 12)
df['adm1_mean_price'] = df.groupby('adm1_id')['mp_price'].transform('mean')

features = ['adm1_id', 'cm_id', 'cur_id', 'pt_id', 'um_id',
            'mp_month_sin', 'mp_month_cos', 'mp_year', 'adm1_mean_price']
target = 'mp_price'

X_raw = df[features].copy()
y_raw = df[target]

# Convert categorical IDs to string for TargetEncoder
id_cols = ['adm1_id', 'cm_id', 'cur_id', 'pt_id', 'um_id']
for col in id_cols:
    X_raw[col] = X_raw[col].astype(str)

# Log-transform target with clipping
clip_val = y_raw.quantile(0.99)
y_log = np.log1p(y_raw.clip(lower=1e-6, upper=clip_val))

# ========== 5. Preprocessing ==========
numeric_feats = ['mp_month_sin', 'mp_month_cos', 'mp_year', 'adm1_mean_price']
cat_feats = id_cols

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_feats),
    ('cat', TargetEncoder(min_samples_leaf=10, smoothing=5), cat_feats)
])
pipeline = Pipeline([('pre', preprocessor)])
X_processed = pipeline.fit_transform(X_raw, y_log)

# ========== 6. Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_log, test_size=0.2, random_state=42
)

# ========== 7. KMeans Clustering ==========
N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
train_clusters = kmeans.fit_predict(X_train)
test_clusters = kmeans.predict(X_test)

# ========== 8. ExtraTrees per Cluster ==========
param_grid = {'n_estimators': [100], 'max_depth': [10, 20]}
cluster_models = {}
best_params = {}

for cid in np.unique(train_clusters):
    mask = train_clusters == cid
    X_c, y_c = X_train[mask], y_train[mask]
    gs = GridSearchCV(ExtraTreesRegressor(random_state=42, n_jobs=-1),
                      param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(X_c, y_c)
    cluster_models[cid] = gs.best_estimator_
    best_params[cid] = gs.best_params_

# ========== 9. Predict ==========
y_pred_log = np.zeros_like(y_test, dtype=float)
for cid in np.unique(test_clusters):
    mask = test_clusters == cid
    y_pred_log[mask] = cluster_models[cid].predict(X_test[mask])

y_true = np.expm1(y_test)
y_pred = np.expm1(y_pred_log)

# ========== 10. Evaluation ==========
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print('\n Hybrid Model (KMeans + ExtraTrees)')
print(f'MAE  = {mae:.2f}')
print(f'RMSE = {rmse:.2f}')
print(f'R²   = {r2:.4f}')

# ========== 11. Visualization ==========
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Market Price (Hybrid Model)")
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.text(
    0.05, 0.95,
    f"RMSE: {rmse:.2f}\nMAE : {mae:.2f}\nR²   : {r2:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)
plt.tight_layout()
plt.show()
