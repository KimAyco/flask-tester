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
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 
    8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'Big', 13: 'Book', 14: 'C', 
    15: 'D', 16: 'Drink', 17: 'E', 18: 'Eat', 19: 'F', 20: 'Friend', 
    21: 'G', 22: 'Go', 23: 'Good', 24: 'H', 25: 'Happy', 26: 'Help', 
    27: 'Home', 28: 'How', 29: 'I', 30: 'K', 31: 'L', 32: 'M', 33: 'N', 
    34: 'P', 35: 'Please', 36: 'Q', 37: 'R', 38: 'S', 39: 'Sad', 
    40: 'Small', 41: 'Sorry', 42: 'Stop', 43: 'T', 44: 'Teacher', 
    45: 'Thank you', 46: 'U', 47: 'V', 48: 'W', 49: 'Water', 50: 'What', 
    51: 'When', 52: 'Where', 53: 'Who', 54: 'X', 55: 'Y', 56: 'hello', 
    57: 'j', 58: 'why', 59: 'z'
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


