from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load labels
labels = {}
with open("labels.txt") as f:
    for line in f:
        idx, name = line.strip().split(",")
        labels[int(idx)] = name

# Config
EXPECTED_FRAMES = 9
FEATURE_DIM = 106

# Initialize Flask
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load CSV file from form-data
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Read CSV with header and skip it
        df = pd.read_csv(file, header=0)  # ✅ Skip the column names

        # Fill missing and validate shape
        df = df.fillna(0.0)

        if df.shape[1] != FEATURE_DIM:
            return jsonify({"error": f"Invalid feature dimension: expected {FEATURE_DIM}, got {df.shape[1]}"}), 400

        if df.shape[0] < EXPECTED_FRAMES:
            padding = pd.DataFrame(np.zeros((EXPECTED_FRAMES - len(df), FEATURE_DIM)))
            df = pd.concat([df, padding], ignore_index=True)
        else:
            df = df.iloc[:EXPECTED_FRAMES]

        # Convert to model input
        input_tensor = np.expand_dims(df.to_numpy(dtype=np.float32), axis=0)

        # Predict
        prediction = model.predict(input_tensor)[0]
        class_idx = int(np.argmax(prediction))
        class_name = labels[class_idx]
        confidence = float(prediction[class_idx])

        return jsonify({
            "prediction": class_name,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Entry point for local dev
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
