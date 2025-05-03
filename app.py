from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (make sure the model is in the same directory as app.py or provide the correct path)
model = tf.keras.models.load_model('model.h5')  # Change this if your model path is different

@app.route('/')
def hello_world():
    return 'Welcome to the Gesture Tester API!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the incoming data from the POST request
        data = request.get_json()

        # Assuming the incoming data is a list of features (make sure your features match the model's input)
        features = np.array(data['features'])  # Ensure your features are in a proper format for prediction
        
        # If necessary, reshape the input data to match the expected input shape (e.g., (1, frames, features))
        features = features.reshape((1, 9, 106))  # Adjust this based on your model's expected shape

        # Make prediction
        prediction = model.predict(features)

        # Assuming your model's output is a vector of class probabilities (change if your model outputs differently)
        predicted_idx = np.argmax(prediction)
        predicted_label = ['Hello', 'j', 'z'][predicted_idx]  # Modify this with your actual labels
        confidence = prediction[0][predicted_idx] * 100  # Confidence of prediction

        # Return the prediction in JSON format
        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
