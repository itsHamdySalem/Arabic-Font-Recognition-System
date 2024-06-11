# Arabic Font Recognition System

## Overview

The Arabic Font Recognition System is designed to classify Arabic text in an image into one of four specific fonts. The fonts are:

0. Scheherazade New
1. Marhey
2. Lemonada
3. IBM Plex Sans Arabic

This system handles various image variations to ensure accurate font recognition. It comprises a complete machine learning pipeline, including preprocessing, feature extraction/selection, model selection and training, and performance analysis.

## Project Modules

### Preprocessing Module
The preprocessing module addresses different paragraph formats in images. It ensures that the text is correctly aligned and standardized for the following stages in the pipeline. It handles:
- Text alignment
- Text size/weight
- Image blur
- Brightness variation
- Text color
- Text rotation
- Salt and pepper noise

### Feature Extraction/Selection Module
In this module, significant features from the preprocessed images are extracted and selected for model training. These features are crucial for distinguishing between different fonts.

### Model Selection and Training Module
This module involves selecting the appropriate machine learning model and training it using the extracted features. It includes:
- Choosing the right algorithm
- Training the model on the training dataset
- Fine-tuning hyperparameters

### Performance Analysis Module
The performance analysis module evaluates the trained model to ensure its effectiveness and accuracy. It involves:
- Measuring accuracy
- Analyzing the model’s performance on validation and test datasets
- Fine-tuning based on performance metrics

## Server

The system includes a server that accepts POST requests at the endpoint `/predict`. An image is attached to the request, and a prediction is returned.

## Dataset

The dataset used to train and evaluate the model is available on Kaggle: [Arabic Fonts Dataset](https://www.kaggle.com/datasets/breathemath/fonts-dataset-cmp).

The dataset is divided into:
- **Training set**: Used to train the model.
- **Validation set**: Used to tune hyperparameters and choose between models.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- Necessary libraries: openCV, numpy, sklearn, joblib, matplotlib, flask

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/itsHamdySalem/Arabic-Font-Recognition-System.git

2. Navigate to the project directory:
    ```sh
    cd Arabic-Font-Recognition-System

## Results

The system classifies the text into one of the four fonts with high accuracy, handling various image distortions and text variations effectively. The model achieves an accuracy of **98.7%**.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

- Kaggle for the dataset: [Arabic Fonts Dataset](https://www.kaggle.com/datasets/breathemath/fonts-dataset-cmp).
