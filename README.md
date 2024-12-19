# MNIST Classification Project

## Overview
This project implements and evaluates various machine learning and deep learning models on the MNIST dataset. The goal is to classify handwritten digits using Neural Networks (NN), Convolutional Neural Networks (CNN), and Support Vector Machines (SVM).

---

## Features
1. **Data Preprocessing**:
   - Normalized pixel values (scaled to [0,1]).
   - Flattened images for SVM compatibility.

2. **Model Implementations**:
   - **Neural Network (NN)**: A feedforward network with dense layers.
   - **Convolutional Neural Network (CNN)**: Designed for better image classification performance.
   - **Support Vector Machine (SVM)**: Trained with linear and RBF kernels.

3. **Evaluation Metrics**:
   - Confusion Matrices
   - Precision, Recall, and F1-Score
   - Accuracy Tracking

4. **Visualizations**:
   - Training accuracy/loss curves for NN and CNN.
   - Confusion matrices for all models.
   - SVM accuracy trends with varying training sizes.

---

## Results

### Model Comparison
| Model   | Accuracy | Strengths                  | Weaknesses                       |
|---------|----------|----------------------------|-----------------------------------|
| NN      | ~96%     | Fast training              | Limited feature extraction       |
| CNN     | ~99%     | Superior for image data    | Computationally expensive        |
| SVM     | ~97%     | Effective for small data   | Struggles with large dimensions  |


![Model Comparison](https://github.com/omkarnitturkar/Handwritten-digit-recognition/blob/main/Model%20accuray%20Comparision.png)
### Visualizations
- Confusion matrices for all models.
- Training and validation loss/accuracy plots for NN and CNN.
- SVM performance trends with increasing training sizes.

---

## Challenges
1. **Overfitting**: Addressed by tuning regularization and using validation data.
2. **Computational Limitations**: CNNs required significant memory and computation time.
3. **Hyperparameter Tuning**: Experimented with learning rates, kernels, and architectures.
4. **SVM Scalability**: Managed dimensionality issues using preprocessing and kernel selection.

---

## Future Work
1. Implement data augmentation for CNN to improve generalization.
2. Experiment with transfer learning for better performance.
3. Optimize SVM hyperparameters for scalability.
4. Explore ensemble methods combining NN, CNN, and SVM.

---

## Installation

### Prerequisites
Ensure you have Python 3.x installed with the following libraries:
- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Install Dependencies
To install all dependencies, run:
```bash
pip install -r requirements.txt
