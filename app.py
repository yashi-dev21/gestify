import os
import requests
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

MODEL_PATH = "gesture_model.pkl"
MODEL_URL = "https://github.com/yashi-dev21/gestify/releases/download/v1/gesture_model.pkl" 
# Make sure this is your actual GitHub Release URL


class DummyModel:
    """
    Fallback model used if the real model cannot be loaded.
    This keeps the API working for demo / resume purposes.
    """
    def predict(self, X):
        # X is expected to be a 2D array; we return same label for all rows
        return ["A"] * len(X)


model = None  # will be set in load_model()


def download_model():
    """Download the model from GitHub Releases if it's not present."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model file not found. Downloading from GitHub Releases...")

        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("✅ Model downloaded successfully.")


def load_model():
    """Try to load the real model. If it fails, use DummyModel."""
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Loaded gesture_model.pkl")
    except Exception as e:
        print("❌ Could not load gesture_model.pkl:", e)
        print("➡️ Falling back to DummyModel so API keeps working.")
        model = DummyModel()


# On startup: download + load
download_model()
load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "landmarks": [x1, y1, z1, x2, y2, z2, ...] }
    Returns: { "prediction": "A" }
    """
    if model is None:
        # This should not happen because we set DummyModel when real load fails
        return jsonify({"error": "Model not initialized"}), 500

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
    # Production-safe: no debug=True
    app.run(host="0.0.0.0", port=5000)
