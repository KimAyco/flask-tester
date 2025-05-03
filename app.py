from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf  # or use tflite_runtime.interpreter
import json

app = Flask(__name__)

# Load labels
label_map = {0: "Hello", 1: "j", 2: "z"}  # Modify as per your labels

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data["data"], dtype=np.float32).reshape(1, 9, 106)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_idx = np.argmax(output_data[0])
        confidence = float(output_data[0][predicted_idx])

        return jsonify({
            "prediction": label_map.get(predicted_idx, "Unknown"),
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional root route for testing
@app.route("/", methods=["GET"])
def home():
    return "🧠 TFLite Model Server Running"

if __name__ == '__main__':
    app.run(debug=True)
