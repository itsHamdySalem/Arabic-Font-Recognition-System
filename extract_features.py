import cv2
import os
import numpy as np
from preprocess import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors if descriptors is not None else np.array([])

def quantize_features(features, kmeans):
    return kmeans.predict(features)


X = []
y = []

i = 0
for name in ["Scheherazade New", "Marhey", "Lemonada", "IBM Plex Sans Arabic"]:
    folder_path = "../fonts-dataset/"+name+" Reduced"

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        preprocessed_image = preprocess_image(image_path)
        X.append(preprocessed_image)
        y.append(i)
    i+=1


X_features = [extract_sift_features(image) for image in X]

X_stack = np.vstack([feat for feat in X_features if len(feat) > 0])

num_clusters = 100  # You can adjust this parameter
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X_stack)

X_quantized = [quantize_features(features, kmeans) for features in X_features]

histograms = []
for quantized_features in X_quantized:
    hist, _ = np.histogram(quantized_features, bins=np.arange(num_clusters+1))
    histograms.append(hist)

X_hist = np.array(histograms)

X_train, X_val, y_train, y_val = train_test_split(X_hist, y, test_size=0.2, random_state=42)

clf = SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", 100*accuracy, "%")
