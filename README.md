
# MNIST Digit Recognition Project

## Overview
This project implements and compares different machine learning approaches for the MNIST handwritten digit recognition task. Three different models are implemented and evaluated: a simple Neural Network, a Convolutional Neural Network (CNN), and a Support Vector Machine (SVM).

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)

## Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Seaborn

## Installation
```bash
pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib seaborn
```

## Dataset
The project uses the MNIST dataset, which contains:
- 60,000 training images
- 10,000 test images
- 28x28 pixel grayscale images
- 10 classes (digits 0-9)

The dataset is automatically loaded using Keras' built-in dataset loader.
![Pixel Intensity Correlation](https://github.com/omkarnitturkar/Handwritten-digit-recognition/blob/main/Pixel%20Intensity%20Correlation.png)

## Models

### 1. Simple Neural Network
- Architecture:
  - Flatten layer (input)
  - Dense layer (128 units, ReLU activation)
  - Dense layer (10 units, Softmax activation)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

### 2. Convolutional Neural Network (CNN)
- Architecture:
  - Conv2D layer (32 filters, 3x3 kernel)
  - MaxPooling2D layer (2x2)
  - Flatten layer
  - Dense layer (64 units, ReLU activation)
  - Dense layer (10 units, Softmax activation)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy


![Confusion Matrix Neural Network](https://github.com/omkarnitturkar/Handwritten-digit-recognition/blob/main/Confusion%20Matrix%20Neural%20Network.png)

![Confusion Matrix CNN](https://github.com/omkarnitturkar/Handwritten-digit-recognition/blob/main/Confusion%20Matrix%20CNN.png)
### 3. Support Vector Machine (SVM)
- Kernel: Linear and RBF
- Input: Flattened pixel values (784 features)
- Trained on various dataset sizes to analyze performance scaling

## Results
The project includes comprehensive evaluation metrics:
- Accuracy scores for all models
- Confusion matrices
- Classification reports (precision, recall, F1-score)
- Training history plots for neural networks
- SVM performance scaling analysis

## Challenges and Solutions

### Data Preprocessing
- Challenge: Raw pixel value ranges
- Solution: Normalized pixel values to [0,1] range

### Model Overfitting
- Challenge: Gap between training and test accuracy
- Solution: Implemented dropout and proper regularization

### Computational Efficiency
- Challenge: SVM scaling with large datasets
- Solution: Implemented progressive training size analysis

### Memory Management
- Challenge: Large model memory requirements
- Solution: Optimized batch sizes and model architectures

## Usage
1. Load and preprocess the data:
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

2. Train and evaluate models:
```python
# Neural Network
nn_model = build_nn_model()
nn_model.fit(x_train, y_train, epochs=5)



# CNN
cnn_model = build_cnn_model()
cnn_model.fit(x_train[..., np.newaxis], y_train, epochs=5)

# SVM
svm = SVC(kernel="linear")
svm.fit(x_train_flat, y_train)
```
![NNA accuracy and loss](https://github.com/omkarnitturkar/Handwritten-digit-recognition/blob/main/NNA%20accuracy%20and%20loss.png)
![CNN accuracy and loss](https://github.com/omkarnitturkar/Handwritten-digit-recognition/blob/main/CNN%20accuracy%20and%20loss.png)

## Performance Monitoring
The project includes visualization tools for:
- Training/validation accuracy curves
- Loss curves
- Confusion matrices
- Pixel intensity distributions
- Correlation heatmaps

![SVC Test Accuracy vs Training Size](https://github.com/omkarnitturkar/Handwritten-digit-recognition/blob/main/SVC%20Test%20Accuracy%20vs%20Training%20Size.png)
## Future Improvements
- Implement data augmentation
- Explore deeper CNN architectures
- Add ensemble methods
- Implement cross-validation
- Optimize hyperparameters using grid search
