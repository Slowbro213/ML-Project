# CNN Model for Tabular Forecasting Using TensorFlow

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
# Install category_encoders if not present

from category_encoders import TargetEncoder

# Configuration
load_processed = False
n_components = 0.99
batch_size = 512
epochs = 50
validation_split = 0.2
sample_fraction = 1
random_state = 42

# Helper functions
def preprocess_target(y, clip_percentile=99.0):
    y_clipped = np.clip(y, 1e-6, np.percentile(y, clip_percentile))
    return np.log1p(y_clipped)

def inverse_preprocess_target(y):
    return np.expm1(y)

def cyclical_encode_month(df, col='mp_month'):
    df[f'{col}_sin'] = np.sin(2 * np.pi * (df[col].astype(int) - 1) / 12)
    df[f'{col}_cos'] = np.cos(2 * np.pi * (df[col].astype(int) - 1) / 12)
    return df

def remove_outliers(df, column='mp_price', iqr_multiplier=1.0):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    print(f"Outlier bounds for {column}: [{lower_bound:.2f}, {upper_bound:.2f}]")
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df.reset_index(drop=True)

# Load and preprocess
if not load_processed:
    df = pd.read_csv("wfp_food_prices_database.csv", low_memory=False)
    print("🔎 Raw Dataset Info:")
    print(df.info())

    df = df[df["mp_price"] > 0].reset_index(drop=True)
    df = remove_outliers(df, column='mp_price', iqr_multiplier=1.0)
    df = df.sample(frac=sample_fraction, random_state=random_state).reset_index(drop=True)

    df = cyclical_encode_month(df, col='mp_month')
    df["adm1_mean_price"] = df.groupby("adm1_id")["mp_price"].transform("mean")

    id_columns = ["adm1_id", "cm_id", "cur_id", "pt_id", "um_id"]
    for col in id_columns:
        df[col] = df[col].astype(str)

    target = "mp_price"
    features = [
        "adm1_id", "cm_id", "cm_name", "cur_id", "pt_id", "um_id",
        "mp_month_sin", "mp_month_cos", "mp_year", "adm1_mean_price"
    ]

    X = df[features]
    y = preprocess_target(df[target])

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", TargetEncoder(min_samples_leaf=5, smoothing=10.0), categorical_features)
    ])

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=n_components, random_state=random_state))
    ])

    start_time = time.time()
    X_processed = full_pipeline.fit_transform(X, y)
    print(f"Preprocessing time: {time.time() - start_time:.2f} seconds")

    explained_variance = full_pipeline.named_steps['pca'].explained_variance_ratio_.cumsum()
    print(f"Cumulative explained variance: {explained_variance[-1]:.4f}")

    np.savez_compressed("X_processed.npz", X_processed=X_processed, y=y)
else:
    data = np.load("X_processed.npz")
    X_processed = data["X_processed"]
    y = data["y"]
    print("Loaded processed data.")

print("Processed data shape:", X_processed.shape)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=random_state)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, dtype='float32')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

start_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_split=validation_split,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)
training_time = time.time() - start_time
print(f"Total training time: {training_time:.2f} seconds")
print(f"Average time per epoch: {training_time / len(history.history['loss']):.2f} seconds")

y_pred_test = model.predict(X_test)
y_pred_test_orig = inverse_preprocess_target(y_pred_test.flatten())
y_test_orig = inverse_preprocess_target(y_test)

print(f"\nTest Results:")
print(f"Test MAE: {np.mean(np.abs(y_pred_test_orig - y_test_orig)):.2f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig)):.2f}")
print(f"Test R²: {r2_score(y_test_orig, y_pred_test_orig):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred_test_orig - y_test_orig, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
