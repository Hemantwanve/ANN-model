import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==============================
# GLOBAL TITLE
# ==============================
MAIN_TITLE = "ANN Model for Concrete Strength Prediction"

# ==============================
# CREATE RESULTS FOLDER
# ==============================
os.makedirs("results", exist_ok=True)

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("concrete_strength_dataset.csv")
#X = df[["Age_days", "UPV_m_per_s", "Rebound_Number"]]
X = df.drop("Strength_MPa", axis=1)
y = df["Strength_MPa"]

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# ANN MODEL
# ==============================
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ==============================
# TRAIN MODEL
# ==============================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)

# ==============================
# ANN PREDICTION
# ==============================
y_pred = model.predict(X_test).flatten()

# ==============================
# ANN METRICS
# ==============================
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n===== ANN MODEL PERFORMANCE =====")
print(f"R² Score  : {r2:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"MAE       : {mae:.4f}")

# ==============================
# SONREB MODEL
# ==============================
X_sonreb = df[["UPV_m_per_s", "Rebound_Number"]]
y_sonreb = df["Strength_MPa"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sonreb, y_sonreb, test_size=0.2, random_state=42
)

sonreb_model = LinearRegression()
sonreb_model.fit(X_train_s, y_train_s)

y_sonreb_pred = sonreb_model.predict(X_test_s)

# SONREB METRICS
r2_s = r2_score(y_test_s, y_sonreb_pred)
rmse_s = np.sqrt(mean_squared_error(y_test_s, y_sonreb_pred))
mae_s = mean_absolute_error(y_test_s, y_sonreb_pred)

print("\n===== SONREB MODEL PERFORMANCE =====")
print(f"R² Score  : {r2_s:.4f}")
print(f"RMSE      : {rmse_s:.4f}")
print(f"MAE       : {mae_s:.4f}")

# ==============================
# ERROR ACCURACY
# ==============================
def accuracy_within(y_true, y_pred, threshold):
    return np.mean(np.abs(y_true - y_pred) <= threshold) * 100

acc_1 = accuracy_within(y_test, y_pred, 1)
acc_3 = accuracy_within(y_test, y_pred, 3)
acc_5 = accuracy_within(y_test, y_pred, 5)

# ==============================
# SAVE MODEL & DATA
# ==============================
model.save("results/ann_model.keras")

pd.DataFrame({
    "Actual": y_test.values,
    "ANN_Predicted": y_pred
}).to_csv("results/ann_predictions.csv", index=False)

pd.DataFrame({
    "Actual": y_test_s.values,
    "SONREB_Predicted": y_sonreb_pred
}).to_csv("results/sonreb_predictions.csv", index=False)

# ==============================
# GRAPHS
# ==============================

# Loss Curve
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.title(f"{MAIN_TITLE}\nLoss Curve")
plt.savefig("results/loss_curve.png")
plt.close()

# MAE Curve
plt.figure()
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend()
plt.title(f"{MAIN_TITLE}\nMAE Curve")
plt.savefig("results/mae_curve.png")
plt.close()

# Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"{MAIN_TITLE}\nANN Actual vs Predicted")
plt.savefig("results/ann_actual_vs_predicted.png")
plt.close()

# Residual Plot
residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red')
plt.title(f"{MAIN_TITLE}\nResidual Plot")
plt.savefig("results/residual_plot.png")
plt.close()

# Residual vs Actual
plt.figure()
plt.scatter(y_test, residuals)
plt.axhline(0, color='red')
plt.title(f"{MAIN_TITLE}\nResidual vs Actual")
plt.savefig("results/residual_vs_actual.png")
plt.close()

# ANN vs SONREB
plt.figure()
plt.scatter(y_test, y_pred, label="ANN")
plt.scatter(y_test_s, y_sonreb_pred, label="SONREB")
plt.legend()
plt.title(f"{MAIN_TITLE}\nANN vs SONREB Comparison")
plt.savefig("results/ann_vs_sonreb.png")
plt.close()

# Performance Bar
plt.figure()
plt.bar(["ANN", "SONREB"], [r2, r2_s])
plt.title(f"{MAIN_TITLE}\nR² Comparison")
plt.savefig("results/r2_comparison.png")
plt.close()

# Error Distribution
plt.figure()
plt.hist(y_test - y_pred, alpha=0.6, label="ANN")
plt.hist(y_test_s - y_sonreb_pred, alpha=0.6, label="SONREB")
plt.legend()
plt.title(f"{MAIN_TITLE}\nError Distribution")
plt.savefig("results/error_distribution.png")
plt.close()

# Error Band Accuracy
plt.figure()
plt.bar(["±1", "±3", "±5"], [acc_1, acc_3, acc_5])
plt.title(f"{MAIN_TITLE}\nError Band Accuracy")
plt.savefig("results/error_band_accuracy.png")
plt.close()

# Line Plot
plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title(f"{MAIN_TITLE}\nLine Comparison")
plt.savefig("results/line_plot.png")
plt.close()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title(f"{MAIN_TITLE}\nCorrelation Heatmap")
plt.savefig("results/heatmap.png")
plt.close()

# Boxplot
plt.figure()
plt.boxplot(residuals)
plt.title(f"{MAIN_TITLE}\nError Boxplot")
plt.savefig("results/error_boxplot.png")
plt.close()

import joblib
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df.drop("Strength_MPa", axis=1)
scaler.fit(X)
joblib.dump(scaler, "results/scaler.save")
print("\n✅ ALL RESULTS SAVED IN 'results/' FOLDER")
