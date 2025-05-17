import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
DATA_PATH = 'wfp_food_prices_database.csv'
N_CLUSTERS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load and clean data (same as before) ...
# 1. Load data
print('Loading data...')
df = pd.read_csv(DATA_PATH, low_memory=False)

# 2. Preprocessing
# Drop irrelevant columns
drop_cols = ["mp_commoditysource", "adm0_name", "adm1_name", "mkt_name", "cur_name", "pt_name", "um_name"]
df = df.drop(columns=drop_cols, errors='ignore')
# Filter valid prices
df = df[df['mp_price'] > 0].reset_index(drop=True)
# Remove outliers using IQR
Q1 = df['mp_price'].quantile(0.25)
Q3 = df['mp_price'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.0 * IQR, Q3 + 1.0 * IQR
df = df[(df['mp_price'] >= lower) & (df['mp_price'] <= upper)].reset_index(drop=True)

# Feature engineering
# Cyclical month encoding
df['mp_month_sin'] = np.sin(2 * np.pi * df['mp_month'] / 12)
df['mp_month_cos'] = np.cos(2 * np.pi * df['mp_month'] / 12)
# Regional mean price
df['adm1_mean_price'] = df.groupby('adm1_id')['mp_price'].transform('mean')

# 3. Prepare features and target
features = ['adm1_id', 'cm_id', 'cur_id', 'pt_id', 'um_id',
            'mp_month_sin', 'mp_month_cos', 'mp_year', 'adm1_mean_price']
target = 'mp_price'
X_raw = df[features].copy()
y_raw = df[target]
# Convert ID columns to string for TargetEncoder
id_cols = ['adm1_id', 'cm_id', 'cur_id', 'pt_id', 'um_id']
for col in id_cols:
    X_raw[col] = X_raw[col].astype(str)
# Log1p transform with clipping to 99th percentile
clip_val = y_raw.quantile(0.99)
y_log = np.log1p(y_raw.clip(lower=1e-6, upper=clip_val))

# 4. Preprocessing pipeline
numeric_feats = ['mp_month_sin', 'mp_month_cos', 'mp_year', 'adm1_mean_price']
cat_feats = id_cols
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_feats),
    ('cat', TargetEncoder(min_samples_leaf=10, smoothing=5), cat_feats)
])
pipeline = Pipeline([('pre', preprocessor)])
print('Fitting preprocessing...')
X_processed = pipeline.fit_transform(X_raw, y_log)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')

# 6. Clustering on train set
dist_clusters = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
train_clusters = dist_clusters.fit_predict(X_train)
test_clusters = dist_clusters.predict(X_test)

# Assume all preprocessing and X_processed, y_log, train/test split, and clustering are done
# Now we define the DL model and training loop

def build_model(input_dim):
    model = Sequential([
         tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='gelu'),
    tf.keras.layers.Dense(128, activation='gelu'),
    tf.keras.layers.Dense(64, activation='gelu'),
    tf.keras.layers.Dense(1, dtype='float32')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model_for_cluster(args):
    cid, X_c, y_c, input_dim, epochs, batch_size = args
    print(f'Training DL model for cluster {cid} with {X_c.shape[0]} samples')

    # Rebuild the model in the subprocess
    model = build_model(input_dim)
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_c, y_c, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2, callbacks=[es])
    return cid, model

def train_dl_cluster_models(X, y, clusters, epochs=50, batch_size=64):
    models = {}
    cluster_ids = np.unique(clusters)
    input_dim = X.shape[1]

    tasks = []
    for cid in cluster_ids:
        mask = clusters == cid
        X_c, y_c = X[mask], y[mask]
        tasks.append((cid, X_c, y_c, input_dim, epochs, batch_size))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(train_model_for_cluster, task) for task in tasks]
        for future in as_completed(futures):
            cid, model = future.result()
            models[cid] = model

    return models

print('Training deep learning models per cluster...')
cluster_models = train_dl_cluster_models(X_train, y_train.values, train_clusters)

# Predicting
print('Predicting on test set...')
y_pred_log = np.zeros_like(y_test, dtype=float)
for cid in np.unique(test_clusters):
    mask = test_clusters == cid
    model = cluster_models[cid]
    y_pred_log[mask] = model.predict(X_test[mask]).flatten()

# Inverse transform
y_true = np.expm1(y_test)
y_pred = np.expm1(y_pred_log)

# Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print('Deep Learning Hybrid Model Performance:')
print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}')

# Save results
with open("results.txt", "w") as f:
    f.write("Deep Learning Hybrid Model Performance:\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Deep Learning)")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
