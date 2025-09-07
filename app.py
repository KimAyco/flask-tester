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
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'Big': 12, 'Book': 13, 'C': 14, 'D': 15, 'Drink': 16, 'E': 17, 'Eat': 18, 'F': 19, 'Friend': 20, 'G': 21, 'Go': 22, 'Good': 23, 'H': 24, 'Happy': 25, 'Help': 26, 'Home': 27, 'How': 28, 'I': 29, 'K': 30, 'L': 31, 'M': 32, 'N': 33, 'P': 34, 'Please': 35, 'Q': 36, 'R': 37, 'S': 38, 'Sad': 39, 'Small': 40, 'Sorry': 41, 'Stop': 42, 'T': 43, 'Teacher': 44, 'Thank you': 45, 'U': 46, 'V': 47, 'W': 48, 'Water': 49, 'What': 50, 'When': 51, 'Where': 52, 'Who': 53, 'X': 54, 'Y': 55, 'hello': 56, 'j': 57, 'why': 58, 'z': 59}  # Adjust as needed

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
        input_data = np.array(data["data"], dtype=np.float32).reshape(1, 9, 106)

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

