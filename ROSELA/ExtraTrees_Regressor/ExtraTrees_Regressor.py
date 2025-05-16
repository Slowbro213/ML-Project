import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("wfp_food_prices_database.csv")
df.dropna(subset=["mp_price"], inplace=True)

df = df[[
    'adm0_name', 'adm1_name', 'mkt_name', 'cm_name',
    'cur_name', 'pt_name', 'um_name', 'mp_month', 'mp_year', 'mp_price'
]]
df.dropna(inplace=True)

# Convert time to numeric timestamp
df['mp_month'] = df['mp_month'].astype(int).astype(str).str.zfill(2)
df['mp_year'] = df['mp_year'].astype(int).astype(str)
df['date'] = pd.to_datetime(df['mp_year'] + '-' + df['mp_month'] + '-01')
df['date_int'] = df['date'].astype('int64') // 10**9
df.drop(columns=['mp_month', 'mp_year', 'date'], inplace=True)

# Label encode categorical columns
cat_cols = ['adm0_name', 'adm1_name', 'mkt_name', 'cm_name', 'cur_name', 'pt_name', 'um_name']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Remove outliers
Q1, Q3 = df['mp_price'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['mp_price'] >= Q1 - 1.5 * IQR) & (df['mp_price'] <= Q3 + 1.5 * IQR)]

# Features and target
X = df.drop(columns=['mp_price'])
y = df['mp_price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train ExtraTrees model
etr = ExtraTreesRegressor(n_estimators=100, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=42)
etr.fit(X_train, y_train)

# Predictions
y_pred = etr.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n[ExtraTrees Regressor]")
print(f"R2 Score : {r2:.4f}")
print(f"MAE       : {mae:.4f}")
print(f"RMSE      : {rmse:.4f}")

# Visualizations for ExtraTrees
residuals = y_test - y_pred
sns.histplot(residuals, bins=40, kde=True, color="teal")
plt.title("Error Distribution (Residuals)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predictions vs Actual Prices")
plt.tight_layout()
plt.savefig("predictions_vs_actual.png")
plt.close()

sns.histplot(y, bins=50, kde=True, color="cornflowerblue")
plt.title("Overall Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.close()

trend_df = pd.read_csv("wfp_food_prices_database.csv")
trend_df.dropna(subset=["mp_price", "mp_month", "mp_year"], inplace=True)
trend_df['mp_month'] = trend_df['mp_month'].astype(int).astype(str).str.zfill(2)
trend_df['mp_year'] = trend_df['mp_year'].astype(int).astype(str)
trend_df['date'] = pd.to_datetime(trend_df['mp_year'] + '-' + trend_df['mp_month'] + '-01')
trend_data = trend_df.groupby('date')['mp_price'].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(trend_data['date'], trend_data['mp_price'], color='darkgreen')
plt.title("Average Price Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Average Price")
plt.tight_layout()
plt.savefig("price_trends.png")
plt.close()

#######Results########
# [ExtraTrees Regressor]
# R2 Score : 0.9730
# MAE       : 30.7200
# RMSE      : 78.5652