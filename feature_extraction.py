import cv2
import numpy as np

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return descriptors if descriptors is not None else np.array([])

def quantize_features(features, kmeans):
    return kmeans.predict(features)
