# ========== 1. Kaggle Setup and Download ==========
import os, zipfile
from sklearn.metrics import mean_squared_error
import numpy as np

os.makedirs("/root/.kaggle", exist_ok=True)
#!mv kaggle.json /root/.kaggle/
#!chmod 600 /root/.kaggle/kaggle.json
#!kaggle datasets download -d salehahmedrony/global-food-prices

with zipfile.ZipFile("global-food-prices.zip", 'r') as zip_ref:
    zip_ref.extractall("food_prices_data")

# ========== 2. Install Dependencies ==========
#!pip install scikit-learn

# ========== 3. Import Libraries ==========
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ========== 4. Load and Preprocess Data with Outlier Removal ==========
df = pd.read_csv("food_prices_data/wfp_food_prices_database.csv", low_memory=False)

# Drop rows with missing target
df = df.dropna(subset=['mp_price'])

# Outlier removal using IQR method on original target
Q1 = df['mp_price'].quantile(0.25)
Q3 = df['mp_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['mp_price'] >= lower_bound) & (df['mp_price'] <= upper_bound)]

# Select relevant features
df = df[[  
    'adm0_name', 'adm1_name', 'mkt_name', 'cm_name',
    'cur_name', 'pt_name', 'um_name', 'mp_month', 'mp_year', 'mp_price'
]].dropna()

# Encode categorical features
cat_cols = ['adm0_name', 'adm1_name', 'mkt_name', 'cm_name', 'cur_name', 'pt_name', 'um_name']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])


# ========== 5. Prepare Features & Target ==========
X = df.drop(columns=['mp_price'])
y = df['mp_price']
y_log = np.log1p(y)  # log-transform the target

# ========== 6. Scaling & PCA ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_scaled)

# ========== 7. Train-Test Split ==========
X_train, X_test, y_train_log, y_test_log = train_test_split(X_pca, y_log, test_size=0.2, random_state=42)
y_test = np.expm1(y_test_log)  # unlog for evaluation

# ========== 8. Train ExtraTreesRegressor with GridSearch ==========
etr = ExtraTreesRegressor(random_state=42)
param_grid = {
    'n_estimators': [100],
    'max_depth': [10, None]
}

# Use a subset if needed
X_train_small = X_train[:200000]
y_train_log_small = y_train_log[:200000]

grid_search = GridSearchCV(etr, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=1)
grid_search.fit(X_train_small, y_train_log_small)

# ========== 9. Evaluate ==========
y_pred_log = grid_search.predict(X_test)
y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nExtraTrees Regressor (Hybrid Model: PCA + log target)")
print(f"Best Params: {grid_search.best_params_}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ========== 10. Predict & Evaluate (Hybrid Model) ==========
y_pred_log_hybrid = grid_search.predict(X_test)
y_pred_hybrid = np.expm1(y_pred_log_hybrid)

rmse_hybrid = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))
mae_hybrid = mean_absolute_error(y_test, y_pred_hybrid)
r2_hybrid = r2_score(y_test, y_pred_hybrid)

print("\n🌲 ExtraTrees Regressor (Hybrid Model: PCA + log target)")
print(f"✅ Best Params: {grid_search.best_params_}")
print(f"✅ RMSE: {rmse_hybrid:.2f}")
print(f"✅ MAE : {mae_hybrid:.2f}")
print(f"✅ R²  : {r2_hybrid:.4f}")

# ========== 11. Log-Scale Scatter Plot ==========
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_hybrid, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Market Price (Hybrid Model, Log Scale)")
plt.grid(True)

# Apply log scale for better visibility of distribution
plt.xscale('log')
plt.yscale('log')

# Metrics box
plt.text(
    0.05, 0.95,
    f"RMSE: {rmse_hybrid:.2f}\nMAE : {mae_hybrid:.2f}\nR²   : {r2_hybrid:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.show()
