# ============================================================
# train_model.py — Weather Prediction Model Training Script
# ============================================================
# This script:
#   1. Generates a synthetic weather dataset (or loads your CSV)
#   2. Preprocesses the data (handle missing values, select features)
#   3. Trains a Random Forest classifier
#   4. Prints accuracy metrics
#   5. Saves the model to model/weather_model.pkl
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# ── Step 1: Load or Generate Dataset ────────────────────────
CSV_PATH = "weatherAUS.csv\weatherAUS.csv"   # ← Replace with your own CSV path if you have one

if os.path.exists(CSV_PATH):
    print(f"[INFO] Loading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
else:
    print("[INFO] No CSV found. Generating synthetic weather dataset...")

    np.random.seed(42)
    n = 2000

    temperature = np.random.uniform(-5, 45, n)      # °C
    humidity    = np.random.uniform(10, 100, n)     # %
    pressure    = np.random.uniform(980, 1040, n)   # hPa
    wind_speed  = np.random.uniform(0, 30, n)       # m/s

    # Rain logic: high humidity + low pressure + moderate temp → rain
    rain_score = (
        (humidity > 70).astype(int) * 2 +
        (pressure < 1010).astype(int) * 2 +
        (temperature > 5).astype(int) +
        (wind_speed > 10).astype(int)
    )
    rain = (rain_score >= 4).astype(int)   # 1 = Rain, 0 = No Rain

    df = pd.DataFrame({
        "temperature": temperature,
        "humidity":    humidity,
        "pressure":    pressure,
        "wind_speed":  wind_speed,
        "rain":        rain
    })

    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Synthetic dataset saved to {CSV_PATH} ({n} rows)")

# ── Step 2: Preprocessing ────────────────────────────────────
print("\n[INFO] Preprocessing data...")

# Show first few rows
print(df.head())

# Handle missing values — fill numeric columns with their median
df.fillna(df.median(numeric_only=True), inplace=True)

# Select features and label
FEATURES = ["temperature", "humidity", "pressure", "wind_speed"]
TARGET   = "rain"   # 1 = Rain, 0 = No Rain

# Make sure all required columns exist
for col in FEATURES + [TARGET]:
    if col not in df.columns:
        raise ValueError(f"[ERROR] Column '{col}' not found in dataset. "
                         f"Available columns: {list(df.columns)}")

X = df[FEATURES]
y = df[TARGET]

print(f"\n[INFO] Dataset shape: {X.shape}")
print(f"[INFO] Rain distribution:\n{y.value_counts().rename({0:'No Rain', 1:'Rain'})}")

# ── Step 3: Train / Test Split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[INFO] Training samples : {len(X_train)}")
print(f"[INFO] Testing  samples : {len(X_test)}")

# ── Step 4: Scale Features ───────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Step 5: Train Random Forest Model ───────────────────────
print("\n[INFO] Training Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    max_depth=10,        # prevent overfitting
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ── Step 6: Evaluate Model ───────────────────────────────────
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n[INFO] Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES)
print("\n[INFO] Feature Importances:")
print(importances.sort_values(ascending=False).to_string())

# ── Step 7: Save Model & Scaler ──────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(model,  "model/weather_model.pkl")
joblib.dump(scaler, "model/weather_scaler.pkl")

print("\n✅ Model saved  → model/weather_model.pkl")
print("✅ Scaler saved → model/weather_scaler.pkl")
print("\n[INFO] Training complete! You can now run: python app.py")
