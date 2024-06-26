import os
import cv2
import time
import numpy as np
from preprocessing import preprocess_image
from feature_extraction import extract_sift_features, quantize_features

def test_model(test_folder, clf, kmeans):
    predictions = []
    times = []

    filenames = sorted(os.listdir(test_folder), key=lambda x: int(x.split('.')[0]))
    for filename in filenames:
        if filename.endswith('.jpeg'):
            image_path = os.path.join(test_folder, filename)
            test_image = cv2.imread(image_path)
            
            start_time = time.time()
            preprocessed_image = preprocess_image(test_image)

            features = extract_sift_features(preprocessed_image)
            if len(features) == 0:
                predictions.append(-1)
            else:
                quantized_features = quantize_features(features, kmeans)
                hist, _ = np.histogram(quantized_features, bins=np.arange(kmeans.n_clusters + 1))
                hist = np.array(hist).reshape(1, -1)

                prediction = clf.predict(hist)[0]
                predictions.append(prediction)

            end_time = time.time()
            elapsed_time = round(end_time - start_time, 3)
            times.append(elapsed_time)

    return predictions, times

def write_results(predictions, times):
    with open('./results/results.txt', 'w') as results_file:
        results_file.write('\n'.join(map(str, predictions)))

    with open('./results/time.txt', 'w') as time_file:
        time_file.write('\n'.join(map(str, times)))    
