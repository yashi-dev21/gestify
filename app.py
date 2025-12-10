# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model once at startup
try:
    with open("gesture_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Loaded gesture_model.pkl")
except Exception as e:
    print("⚠️ Could not load gesture_model.pkl:", e)
    model = None

@app.route("/")
def index():
    # Renders templates/index.html
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "landmarks": [x1, y1, z1, x2, y2, z2, ...] }
    Returns: { "prediction": "A" }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "landmarks" not in data:
        return jsonify({"error": "No landmarks provided"}), 400

    landmarks = data["landmarks"]

    # Expect 63 values (21 landmarks * 3 coordinates)
    if len(landmarks) != 63:
        return jsonify({"error": f"Expected 63 values, got {len(landmarks)}"}), 400

    try:
        arr = np.array(landmarks, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        return jsonify({"prediction": str(pred)})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    # Production-safe: no debug mode
    app.run(host="0.0.0.0", port=5000)
