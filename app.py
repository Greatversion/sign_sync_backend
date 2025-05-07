# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
import base64
from utils import extract_hand_landmarks

app = Flask(__name__)

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: "hello", 1: "no", 2: "yes"}

@app.route('/')
def index():
    return "Sign Language API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    landmarks = extract_hand_landmarks(image)

    if landmarks:
        prediction = model.predict([np.asarray(landmarks)])
        label = labels_dict[int(prediction[0])]
        return jsonify({'prediction': label})
    else:
        return jsonify({'error': 'No hand detected'}), 200

if __name__ == '__main__':
    app.run(debug=True)
