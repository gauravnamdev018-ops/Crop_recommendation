from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import requests
import os
import random
from dotenv import load_dotenv

# =============================
# INITIAL SETUP
# =============================
load_dotenv()

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Crop_recommendation1.csv")
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")

# =============================
# LOAD DATASET
# =============================
df = pd.read_csv(CSV_PATH)
X = df.drop("label", axis=1)
y = df["label"]

# =============================
# TRAIN / LOAD MODEL
# =============================
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print("Model saved.")

model = joblib.load(MODEL_PATH)

# =============================
# WEATHER FUNCTION
# =============================
def get_weather(lat, lon):
    api_key = os.getenv("OPENWEATHER_API_KEY")

    defaults = (
        random.uniform(20, 35),
        random.uniform(40, 90),
        random.uniform(50, 300)
    )

    if not api_key or lat is None or lon is None:
        return defaults

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )

        res = requests.get(url, timeout=5)
        if res.status_code != 200:
            return defaults

        data = res.json()
        temp = data["main"].get("temp", defaults[0])
        humidity = data["main"].get("humidity", defaults[1])
        rainfall = data.get("rain", {}).get("1h", defaults[2])

        return temp, humidity, rainfall

    except Exception as e:
        print("Weather error:", e)
        return defaults

# =============================
# API ENDPOINT
# =============================
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        soil_ph = float(data["soil_ph"])
        if not 4 <= soil_ph <= 9:
            return jsonify({"error": "Soil pH must be between 4 and 9"}), 400
    except:
        return jsonify({"error": "Invalid soil pH"}), 400

    lat = data.get("latitude")
    lon = data.get("longitude")
    season = data.get("season", 0)

    K = data.get("K", 50)
    N = data.get("N", 50)
    P = data.get("P", 50)

    temperature, humidity, rainfall = get_weather(lat, lon)

    if season == 0:
        temperature -= 4
        rainfall *= 0.6
    elif season == 1:
        temperature += 4
        humidity -= 8
    elif season == 2:
        rainfall *= 1.6
        humidity += 10

    input_df = pd.DataFrame(
        [[N, P, K, temperature, humidity, soil_ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    top = probs.argsort()[-3:][::-1]

    result = [
        {"crop": classes[i], "confidence": round(probs[i] * 100, 2)}
        for i in top
    ]

    return jsonify({
        "recommendations": result,
        "weather_used": {
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 2),
            "rainfall": round(rainfall, 2)
        },
        "soil_nutrients": {"N": N, "P": P, "K": K}
    })

# =============================
# RUN SERVER
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
