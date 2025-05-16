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

# ========== 2. Install TabNet Dependencies ==========
#!pip install pytorch-tabnet category_encoders

# ========== 3. Load and Preprocess Data with Outlier Removal ==========
import pandas as pd

df = pd.read_csv("/content/food_prices_data/wfp_food_prices_database.csv", low_memory=False)

# Drop rows with missing target
df = df.dropna(subset=['mp_price'])

# Remove outliers in target using IQR
Q1 = df['mp_price'].quantile(0.25)
Q3 = df['mp_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['mp_price'] >= lower_bound) & (df['mp_price'] <= upper_bound)]

# Keep only relevant columns
df = df[[  
    'adm0_name', 'adm1_name', 'mkt_name', 'cm_name',
    'cur_name', 'pt_name', 'um_name', 'mp_month', 'mp_year', 'mp_price'
]].dropna()


# ========== 4. Encode Categoricals with OrdinalEncoder ==========
from category_encoders import OrdinalEncoder

cat_cols = ['adm0_name', 'adm1_name', 'mkt_name', 'cm_name', 'cur_name', 'pt_name', 'um_name']
encoder = OrdinalEncoder(cols=cat_cols)
df = encoder.fit_transform(df)

# ========== 5. Train-Test Split ==========
from sklearn.model_selection import train_test_split

X = df.drop(columns=['mp_price'])
y = df['mp_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to NumPy for TabNet
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

# ========== 6. TabNet Regressor ==========
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

tabnet = TabNetRegressor(
    verbose=1,
    seed=42,
    device_name='cuda' if torch.cuda.is_available() else 'cpu'
)

tabnet.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_test_np, y_test_np)],
    eval_metric=['rmse'],
    max_epochs=100,
    patience=10,
    batch_size=1024, virtual_batch_size=128
)




import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 1. Predict & Evaluate ==========
y_pred = tabnet.predict(X_test_np).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 TabNet Evaluation")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE : {mae:.2f}")
print(f"✅ R²  : {r2:.4f}")

# ========== 2. Log Scale Scatter Plot with Metrics ==========
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Market Price (TabNet, Log Scale)")
plt.grid(True)

plt.xscale('log')
plt.yscale('log')

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
