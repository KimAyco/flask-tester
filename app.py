from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf

# 🔄 Load model + labels
model = tf.keras.models.load_model("model.h5")
labels = {}
with open("labels.txt") as f:
    for line in f:
        idx, name = line.strip().split(",")
        labels[int(idx)] = name

EXPECTED_FRAMES = 9
FEATURE_DIM = 106

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load CSV or JSON data from request
        data = request.get_json()
        df = pd.DataFrame(data)

        # ✅ Preprocess: fill NaN, pad/truncate
        df = df.fillna(0.0)
        if df.shape[1] != FEATURE_DIM:
            return jsonify({"error": "Invalid feature dimension"}), 400
        if df.shape[0] < EXPECTED_FRAMES:
            padding = pd.DataFrame(np.zeros((EXPECTED_FRAMES - len(df), FEATURE_DIM)))
            df = pd.concat([df, padding], ignore_index=True)
        else:
            df = df.iloc[:EXPECTED_FRAMES]

        # ➕ Add batch dim
        input_tensor = np.expand_dims(df.to_numpy(), axis=0)

        # 🔍 Predict
        prediction = model.predict(input_tensor)[0]
        class_idx = int(np.argmax(prediction))
        class_name = labels[class_idx]
        confidence = float(prediction[class_idx])

        return jsonify({"prediction": class_name, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
