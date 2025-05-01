from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd

# 🧠 Model + label config
MODEL_PATH = "model_tfjs/model.h5"  # exported as .h5 instead of tfjs for backend
LABELS_PATH = "model_tfjs/labels.txt"
EXPECTED_FRAMES = 9
FEATURE_DIM = 106

# 🚀 Init app
app = Flask(__name__)
CORS(app)

# 📦 Load model + labels
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

with open(LABELS_PATH, "r") as f:
    label_map = [line.strip().split(",")[1] for line in f.readlines()]
print("✅ Labels:", label_map)

@app.route("/")
def home():
    return "🎯 API Ready for Inference", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 📥 JSON file from frontend
        file = request.files["file"]
        df = pd.read_csv(file, header=None)

        # 🧹 Clean and format input
        df = df.fillna(0.0)

        # Pad/truncate to 9 frames
        if df.shape[0] < EXPECTED_FRAMES:
            pad = pd.DataFrame(np.zeros((EXPECTED_FRAMES - df.shape[0], FEATURE_DIM)))
            df = pd.concat([df, pad], ignore_index=True)
        else:
            df = df.iloc[:EXPECTED_FRAMES]

        # 🚀 Run prediction
        input_tensor = np.expand_dims(df.to_numpy(dtype=np.float32), axis=0)  # shape: [1, 9, 106]
        pred = model.predict(input_tensor)
        class_index = np.argmax(pred)
        confidence = float(np.max(pred))
        predicted_label = label_map[class_index]

        return jsonify({
            "class": predicted_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🧪 Local run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
