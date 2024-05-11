import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from feature_extraction import extract_sift_features, quantize_features
from preprocessing import preprocess_image
import joblib

def train_model(data_folder, num_clusters=100):
    X = []
    y = []

    i = 0
    for name in ["Scheherazade New", "Marhey", "Lemonada", "IBM Plex Sans Arabic"]:
        folder_path = os.path.join(data_folder, name)

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)

            image = cv2.imread(image_path)
            preprocessed_image = preprocess_image(image)
            X.append(preprocessed_image)
            y.append(i)
        i += 1

    X_features = [extract_sift_features(image) for image in X]
    X_stack = np.vstack([feat for feat in X_features if len(feat) > 0])

    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(X_stack)

    X_quantized = [quantize_features(features, kmeans) for features in X_features]

    histograms = []
    for quantized_features in X_quantized:
        hist, _ = np.histogram(quantized_features, bins=np.arange(num_clusters + 1))
        histograms.append(hist)

    X_hist = np.array(histograms)

    X_train, X_val, y_train, y_val = train_test_split(X_hist, y, test_size=0.2, random_state=42)

    clf = SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print("Validation Accuracy:", 100 * accuracy, "%")

    return clf, kmeans

def save_model(clf, kmeans, model_path, kmeans_path):
    joblib.dump(clf, model_path)
    joblib.dump(kmeans, kmeans_path)

def load_model(model_path, kmeans_path):
    clf = joblib.load(model_path)
    kmeans = joblib.load(kmeans_path)
    return clf, kmeans
