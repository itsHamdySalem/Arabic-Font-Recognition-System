from flask import Flask, request, jsonify
from model_training import load_model
from test_model import preprocess_image, extract_sift_features, quantize_features
import cv2
import numpy as np
import time

app = Flask(__name__)

model_path = "models/trained_model.joblib"
kmeans_path = "models/kmeans_model.joblib"
loaded_clf, loaded_kmeans = load_model(model_path, kmeans_path)

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)
    
    features = extract_sift_features(preprocessed_image)
    if len(features) == 0:
        return jsonify({'error': 'No features found in the image'}), 400
    
    quantized_features = quantize_features(features, loaded_kmeans)
    hist, _ = np.histogram(quantized_features, bins=np.arange(loaded_kmeans.n_clusters + 1))
    hist = np.array(hist).reshape(1, -1)
    
    # Predict using the loaded model and measure time taken
    prediction = loaded_clf.predict(hist)
    print(prediction[0])
 
    return jsonify({'prediction': int(prediction[0])}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)