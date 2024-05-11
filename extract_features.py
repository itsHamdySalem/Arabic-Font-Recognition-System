import cv2
import os
import numpy as np
from preprocess import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors if descriptors is not None else np.array([])

def quantize_features(features, kmeans):
    return kmeans.predict(features)


y = []
X_features = []

i = 0
for name in ["Scheherazade New", "Marhey", "Lemonada", "IBM Plex Sans Arabic"]:
    folder_path = "../fonts-dataset/"+name

    for filename in os.listdir(folder_path)[:1000]:
        image_path = os.path.join(folder_path, filename)

        preprocessed_image = preprocess_image(image_path)

        featured_image = extract_sift_features(preprocessed_image)

        if len(featured_image) > 0:
            X_features.append(featured_image)
            y.append(i)
    i+=1


X_stack = np.vstack(X_features)

num_clusters = 200  # You can adjust this parameter
kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
kmeans.fit(X_stack)

X_quantized = [quantize_features(features, kmeans) for features in X_features]

histograms = []
for quantized_features in X_quantized:
    hist, _ = np.histogram(quantized_features, bins=np.arange(num_clusters+1))
    histograms.append(hist)

X_hist = np.array(histograms)

X_train, X_val, y_train, y_val = train_test_split(X_hist, y, test_size=0.2, random_state=42)


# param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': ['scale', 'auto']
# }

# svm = SVC()

# grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

clf = SVC(C=10, kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
