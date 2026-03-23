# ============================================================
# app.py — Flask Backend for Weather Prediction App
# ============================================================
# Routes:
#   GET  /           → Serves the main HTML page
#   POST /predict    → Fetches live weather data + returns ML prediction
# ============================================================

import os
import requests
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# ── Load environment variables from .env file ────────────────
load_dotenv()

# ── App Setup ────────────────────────────────────────────────
app = Flask(__name__)

# 🔑 Loaded from .env file — never hardcode secrets in source code
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_URL     = "https://api.openweathermap.org/data/2.5/weather"

# ── Load Model & Scaler ──────────────────────────────────────
MODEL_PATH  = os.path.join("model", "weather_model.pkl")
SCALER_PATH = os.path.join("model", "weather_scaler.pkl")

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    print("❌ Model not found! Please run: python train_model.py first.")
    model  = None
    scaler = None

# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "city": "London" }
    1. Calls OpenWeather API for live data
    2. Extracts features: temperature, humidity, pressure, wind_speed
    3. Scales & feeds into ML model
    4. Returns prediction + weather details as JSON
    """

    # ── Validate API key is set ──────────────────────────────
    if not OPENWEATHER_API_KEY:
        return jsonify({
            "error": "OPENWEATHER_API_KEY not set. Add it to your .env file."
        }), 500

    # ── Validate model is loaded ─────────────────────────────
    if model is None or scaler is None:
        return jsonify({
            "error": "Model not loaded. Please run train_model.py first."
        }), 500

    # ── Get city from request body ───────────────────────────
    data = request.get_json()
    city = data.get("city", "").strip()

    if not city:
        return jsonify({"error": "City name is required."}), 400

    # ── Fetch Live Weather from OpenWeather API ───────────────
    try:
        params = {
            "q":     city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"   # °C, m/s
        }
        response = requests.get(OPENWEATHER_URL, params=params, timeout=10)

        # Handle non-200 responses
        if response.status_code == 401:
            return jsonify({"error": "Invalid API key. Check your OpenWeather API key."}), 401
        if response.status_code == 404:
            return jsonify({"error": f"City '{city}' not found. Try a different name."}), 404
        if response.status_code != 200:
            return jsonify({"error": f"Weather API error: {response.status_code}"}), 502

        weather_data = response.json()

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "No internet connection. Check your network."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Weather API timed out. Please try again."}), 504
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    # ── Extract Features from API Response ───────────────────
    try:
        temperature = weather_data["main"]["temp"]        # °C
        humidity    = weather_data["main"]["humidity"]    # %
        pressure    = weather_data["main"]["pressure"]    # hPa
        wind_speed  = weather_data["wind"]["speed"]       # m/s
        description = weather_data["weather"][0]["description"].title()
        icon_code   = weather_data["weather"][0]["icon"]
        city_name   = weather_data["name"]
        country     = weather_data["sys"]["country"]

    except (KeyError, IndexError) as e:
        return jsonify({"error": f"Unexpected API response format: {str(e)}"}), 500

    # ── Prepare Input for Model ───────────────────────────────
    # Must match the exact order used during training:
    # [temperature, humidity, pressure, wind_speed]
    features = np.array([[temperature, humidity, pressure, wind_speed]])
    features_scaled = scaler.transform(features)

    # ── Make Prediction ───────────────────────────────────────
    prediction_code = model.predict(features_scaled)[0]          # 0 or 1
    prediction_proba = model.predict_proba(features_scaled)[0]   # [prob_no_rain, prob_rain]

    prediction_label = "🌧 Rain Expected"   if prediction_code == 1 else "☀️ No Rain Expected"
    confidence       = round(float(max(prediction_proba)) * 100, 1)

    # ── Return JSON Response ──────────────────────────────────
    return jsonify({
        "city":        f"{city_name}, {country}",
        "temperature": round(temperature, 1),
        "humidity":    humidity,
        "pressure":    pressure,
        "wind_speed":  round(wind_speed, 1),
        "description": description,
        "icon":        f"https://openweathermap.org/img/wn/{icon_code}@2x.png",
        "prediction":  prediction_label,
        "confidence":  confidence,
    })


# ── Run App ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌤  Weather Prediction App starting...")
    print("   Open: http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)