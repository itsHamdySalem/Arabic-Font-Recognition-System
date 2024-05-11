import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def performance_analysis(actual_results, predicted_results, times):
    accuracy = accuracy_score(actual_results, predicted_results)
    precision = precision_score(actual_results, predicted_results, average='macro')
    recall = recall_score(actual_results, predicted_results, average='macro')
    f1 = f1_score(actual_results, predicted_results, average='macro')
    avg_time = sum(times) / len(times)

    cm = confusion_matrix(actual_results, predicted_results)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Scheherazade New', 'Marhey', 'Lemonada', 'IBM Plex Sans Arabic']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.show()

    return accuracy, precision, recall, f1, avg_time