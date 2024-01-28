# Digit Recognition using Convolutional Neural Networks (CNN)

This repository contains a Python implementation of a Convolutional Neural Network (CNN) for digit recognition. The project utilizes the Keras library to build and train the model on a dataset of handwritten digits. This type of neural network is well-suited for image classification tasks such as recognizing handwritten digits.

## Overview

The project involves the use of a CNN model to classify handwritten digits from the MNIST dataset. The code includes preprocessing of data, creation of a CNN model, training the model with augmented data, and evaluating its performance. The model's predictions are saved and visualized to validate its effectiveness.

## Requirements

- Python 3
- Libraries: 
  - numpy
  - pandas
  - keras
  - matplotlib
  - scikit-learn

## Dataset

The dataset used is the MNIST dataset of handwritten digits. It contains 60,000 training images and 10,000 testing images, each of size 28x28 pixels.

## Implementation Details

1. **Data Preparation**: The training and test datasets are loaded. The data is reshaped and normalized to aid in the training process.

2. **CNN Model**: 
   - The CNN model consists of convolutional layers, max pooling, dropout, and dense layers.
   - Activation functions used include ReLU and softmax for multi-class classification.
   - The model is compiled with the RMSprop optimizer and categorical cross-entropy loss function.

3. **Training**: 
   - The model is trained using data augmentation techniques like rotation, zoom, and width/height shifts.
   - A callback for reducing learning rate on plateaus is used to optimize training.
   - The training process includes validation to monitor the model's performance.

4. **Evaluation and Prediction**: 
   - The trained model is used to make predictions on the test dataset.
   - Predictions are saved to a CSV file.
   - A visual inspection of predictions is provided for sample test images.

## Usage

1. Clone the repository.
2. Ensure all required libraries are installed.
3. Run the script. The model will train and predictions will be saved in `result.csv`.

## Code Structure

- Data loading and preprocessing
- Model building
- Data augmentation
- Model training and callbacks
- Predictions and result saving
- Visualization of predictions

## Results

The model's performance can be evaluated based on its accuracy on the validation set during training and by visually inspecting the predictions made on the test set.

## License

This project is open-sourced under the MIT license.
