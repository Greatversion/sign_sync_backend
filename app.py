from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
import os
import psutil
from utils import extract_hand_landmarks

app = Flask(__name__)

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: "hello", 1: "no", 2: "yes"}

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 ** 2} MB")

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

    # Resize the image to 640x480 to reduce memory and processing time
    image = cv2.resize(image, (640, 480))
    
    log_memory_usage()  # Log memory usage after resizing

    # Extract hand landmarks
    landmarks = extract_hand_landmarks(image)

    if landmarks:
        prediction = model.predict([np.asarray(landmarks)])
        label = labels_dict[int(prediction[0])]
        return jsonify({'prediction': label})
    else:
        return jsonify({'error': 'No hand detected'}), 200

if __name__ == '__main__':
    app.run(debug=True)
