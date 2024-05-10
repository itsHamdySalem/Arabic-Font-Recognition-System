import cv2
import os
import numpy as np
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors if descriptors is not None else np.array([])

X = []
y = []

folder_path = "../fonts-dataset/IBM Plex Sans Arabic"

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)

    preprocessed_image = preprocess_image(image_path)
    X.append(preprocessed_image)
    y.append(3)


folder_path = "../fonts-dataset/Lemonada"

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)

    preprocessed_image = preprocess_image(image_path)
    X.append(preprocessed_image)
    y.append(2)


folder_path = "../fonts-dataset/Marhey"

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)

    preprocessed_image = preprocess_image(image_path)
    X.append(preprocessed_image)
    y.append(1)



folder_path = "../fonts-dataset/Scheherazade New"

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)

    preprocessed_image = preprocess_image(image_path)
    X.append(preprocessed_image)
    y.append(0)



X_features = [extract_sift_features(image) for image in X]

X_features = np.array(X_features)

X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=42)

clf = SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
