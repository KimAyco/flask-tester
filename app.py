from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import json

app = Flask(__name__)

# Configuration
FEATURE_DIM = 106
EXPECTED_FRAMES = 9
LABEL_FILE = 'labels.txt'
MODEL_PATH = 'model.h5'

# Load Label Map
label_map = {}
with open(LABEL_FILE, 'r') as f:
    for line in f:
        idx, label = line.strip().split(',')
        label_map[int(idx)] = label

# Load the Model
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_json(data):
    features = data.get('features')
    if not features:
        raise ValueError("Missing 'features' in JSON payload.")

    df = pd.DataFrame(features).fillna(0.0)
    if df.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {df.shape[1]}.")

    if len(df) < EXPECTED_FRAMES:
        pad = pd.DataFrame(np.zeros((EXPECTED_FRAMES - len(df), FEATURE_DIM)))
        df = pd.concat([df, pad], ignore_index=True)
    else:
        df = df.iloc[:EXPECTED_FRAMES]

    return df.to_numpy(dtype=np.float32).reshape(1, EXPECTED_FRAMES, FEATURE_DIM)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json()
        input_data = preprocess_json(payload)
        prediction = model.predict(input_data, verbose=0)
        predicted_idx = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_idx])
        predicted_label = label_map.get(predicted_idx, "Unknown")

        return jsonify({
            "label": predicted_label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Health check route
@app.route('/')
def home():
    return "🧠 Gesture Prediction API is running."

if __name__ == '__main__':
    app.run(debug=True)
