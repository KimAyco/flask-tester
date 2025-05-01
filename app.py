from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf

# Constants
EXPECTED_FRAMES = 9
FEATURE_DIM = 106
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5500"}}, supports_credentials=True)


# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load label map
labels = {}
with open(LABELS_PATH) as f:
    for line in f:
        idx, name = line.strip().split(",")
        labels[int(idx)] = name

@app.route("/")
def index():
    return "🧠 Flask model server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Accept file upload (CSV)
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        df = pd.read_csv(file, header=None)

        # ✅ Fill missing values
        df = df.fillna(0.0)

        # ✅ Validate shape
        if df.shape[1] != FEATURE_DIM:
            return jsonify({"error": f"Expected {FEATURE_DIM} features per row"}), 400

        # ✅ Pad/truncate to EXPECTED_FRAMES
        if df.shape[0] < EXPECTED_FRAMES:
            pad = pd.DataFrame(np.zeros((EXPECTED_FRAMES - len(df), FEATURE_DIM)))
            df = pd.concat([df, pad], ignore_index=True)
        else:
            df = df.iloc[:EXPECTED_FRAMES]

        # ✅ Prepare tensor
        input_tensor = np.expand_dims(df.to_numpy(dtype=np.float32), axis=0)  # Shape: [1, 9, 106]

        # ✅ Run prediction
        prediction = model.predict(input_tensor)[0]
        class_idx = int(np.argmax(prediction))
        class_name = labels.get(class_idx, "Unknown")
        confidence = float(prediction[class_idx])

        return jsonify({
            "prediction": class_name,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
