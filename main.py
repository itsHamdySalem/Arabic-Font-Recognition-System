from model_training import train_model, save_model, load_model
from performance_analysis import performance_analysis
from test_model import test_model, write_results

test_folder = './data/'
data_folder = "./fonts-dataset/"
model_path = "./models/trained_model.joblib"
kmeans_path = "./models/kmeans_model.joblib"

clf, kmeans = train_model(data_folder)
save_model(clf, kmeans, model_path, kmeans_path)

# Later, for testing on a new dataset
loaded_clf, loaded_kmeans = load_model(model_path, kmeans_path)

# Use loaded_clf and loaded_kmeans for prediction
predictions, times = test_model(test_folder, loaded_clf, loaded_kmeans)
write_results(predictions, times)

# Load actual results from file
actual_results = None
with open('./results/actual_results.txt', 'r') as file:
    actual_results = [int(line.strip()) for line in file]

# Apply performance analysis
accuracy, precision, recall, f1, avg_time = performance_analysis(actual_results, predictions, times)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Average Time per Test Iteration:", avg_time)