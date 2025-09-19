from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing CORS
import numpy as np
from tflite_runtime.interpreter import Interpreter
import json

app = Flask(__name__)

# Enabling CORS for all origins (or specify the allowed origins)
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5500",
    "http://localhost",  # <-- ADD THIS
    "https://kimayco.github.io",
    "https://kimayco.github.io/mediapipetest1",
    "http://localhost/capstone/signspeak2.6/translate.php"
]}})

  # Allow requests from your local HTML page

# Label map
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',  

}


# Load the TFLite model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Expecting 9 frames x 24 features (wrist + 5 fingertips per hand, x/y)
        arr = np.array(data["data"], dtype=np.float32)
        if arr.shape != (9, 24):
            return jsonify({
                "error": f"Invalid input shape {arr.shape}, expected (9, 24)"
            }), 400

        # Adapt to model's expected input shape if different (e.g., 9x106)
        expected_shape = tuple(input_details[0]["shape"])  # (1, T, F)
        if len(expected_shape) != 3:
            return jsonify({
                "error": f"Unexpected model input shape {expected_shape}"
            }), 500

        batch_size, expected_time, expected_features = expected_shape

        # Time dimension adjust (pad/truncate along frames)
        if expected_time == 9:
            time_aligned = arr
        elif expected_time > 9:
            pad_frames = expected_time - 9
            time_aligned = np.pad(arr, ((0, pad_frames), (0, 0)), mode='constant')
        else:  # expected_time < 9
            time_aligned = arr[:expected_time, :]

        # Feature dimension adjust (pad/truncate along features)
        if expected_features == 24:
            feature_aligned = time_aligned
        elif expected_features > 24:
            pad_feats = expected_features - 24
            feature_aligned = np.pad(time_aligned, ((0, 0), (0, pad_feats)), mode='constant')
        else:  # expected_features < 24
            feature_aligned = time_aligned[:, :expected_features]

        input_data = feature_aligned.reshape(1, expected_time, expected_features).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_idx = int(np.argmax(output_data[0]))
        confidence = float(output_data[0][predicted_idx])

        return jsonify({
            "prediction": label_map.get(predicted_idx, "Unknown"),
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🧠 TFLite Model Server Running"

if __name__ == '__main__':
    app.run(debug=True)

