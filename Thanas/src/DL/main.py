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
from category_encoders import TargetEncoder

# Configuration
load_processed = False  # Set to True to load preprocessed data
n_components = 0.99  # Fixed for PCA
batch_size = 512  # Optimized for CPU speed
epochs = 50
validation_split = 0.2
sample_fraction = 1  # Full dataset
random_state = 42

# Function to preprocess target
def preprocess_target(y, clip_percentile=99.0):
    y_clipped = np.clip(y, 1e-6, np.percentile(y, clip_percentile))
    return np.log1p(y_clipped)

# Function to inverse-transform predictions
def inverse_preprocess_target(y):
    return np.expm1(y)

# Cyclical encoding for month
def cyclical_encode_month(df, col='mp_month'):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] - 1 / 12)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] - 1 / 12)
    return df

# Outlier removal using IQR
def remove_outliers(df, column='mp_price', iqr_multiplier=1.0):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    print(f"Outlier bounds for {column}: [{lower_bound:.2f}, {upper_bound:.2f}]")
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df.reset_index(drop=True)

# Load and preprocess data
if not load_processed:
    # Load data
    df = pd.read_csv("wfp_food_prices_database.csv", low_memory=False)
    print("🔎 Raw Dataset Info:")
    print(df.info())
    print("\n🔢 Target Value Counts:")
    print(df["mp_price"].describe())

    # Drop irrelevant columns
    df = df.drop(columns=["mp_commoditysource", "adm0_name", "adm1_name", "mkt_name", "cur_name", "pt_name", "um_name"], errors='ignore')
    
    # Filter out invalid prices
    df = df[df["mp_price"] > 0].reset_index(drop=True)
    
    # Remove outliers in mp_price
    df = remove_outliers(df, column='mp_price', iqr_multiplier=1.0)
    print(f"Data shape after outlier removal: {df.shape}")
    
    # Sample the dataset
    df = df.sample(frac=sample_fraction, random_state=random_state).reset_index(drop=True)
    print(f"Sampled data shape: {df.shape}")
    
    # Cyclical encoding for mp_month
    df = cyclical_encode_month(df, col='mp_month')
    
    # Feature engineering: Add mean price per adm1_id
    df["adm1_mean_price"] = df.groupby("adm1_id")["mp_price"].transform("mean")
    
    # Convert ID columns to categorical
    id_columns = ["adm1_id", "cm_id", "cur_id", "pt_id", "um_id"]
    for col in id_columns:
        df[col] = df[col].astype(str)
    
    # Select target and features
    target = "mp_price"
    features = [
        "adm1_id", "cm_id", "cm_name", "cur_id", "pt_id", "um_id",
        "mp_month_sin", "mp_month_cos", "mp_year", "adm1_mean_price"
    ]
    
    X = df[features]
    y = preprocess_target(df[target])
    
    # Verify shapes
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", TargetEncoder(min_samples_leaf=5, smoothing=10.0), categorical_features)
        ]
    )
    
    # Full pipeline with PCA
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=n_components, random_state=random_state))
    ])
    
    # Fit and transform
    start_time = time.time()
    X_processed = full_pipeline.fit_transform(X, y)
    print(f"Preprocessing time: {time.time() - start_time:.2f} seconds")
    
    # Check explained variance
    explained_variance = full_pipeline.named_steps['pca'].explained_variance_ratio_.cumsum()
    print(f"Cumulative explained variance: {explained_variance[-1]:.4f}")
    
    # Save preprocessed data
    np.savez_compressed("X_processed.npz", X_processed=X_processed, y=y)
else:
    data = np.load("X_processed.npz")
    X_processed = data["X_processed"]
    y = data["y"]
    print("Loaded processed data.")

print("Processed data shape:", X_processed.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=random_state)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(2**12, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2**11, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2**10, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2**9, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2**8, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2**7, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, dtype='float32')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
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

# Evaluate on test data
y_pred_test = model.predict(X_test)
y_pred_test_orig = inverse_preprocess_target(y_pred_test.flatten())
y_test_orig = inverse_preprocess_target(y_test)

test_mae = np.mean(np.abs(y_pred_test_orig - y_test_orig))
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
test_r2 = r2_score(y_test_orig, y_pred_test_orig)

print(f"\nTest Results:")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test R²: {test_r2:.4f}")

# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred_test_orig - y_test_orig, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
