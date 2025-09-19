# CNN-MNIST: Convolutional Neural Network for Handwritten Digit Recognition

This repository contains a Jupyter Notebook that demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch.

## Overview

- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 grayscale images of digits (0–9), each 28x28 pixels.
- **Model:** A simple CNN with two convolutional layers, max pooling, and a fully connected output layer.
- **Goal:** Achieve high accuracy in digit classification and visualize the learning process.

## Features

- Data preprocessing: Resize images to 16x16 for faster training.
- Custom CNN architecture explained step-by-step.
- Training and validation loops with loss and accuracy tracking.
- Visualization of training progress (loss and accuracy curves).
- Saving the trained model for future use.
- Example code for visualizing kernels, activations, and misclassified samples.

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yasinpurraisi/CNN-MNIST.git
    cd CNN-MNIST
    ```

2. **Install dependencies:**
    - Python 3.8+
    - PyTorch
    - torchvision
    - matplotlib
    - numpy

    You can install them with:
    ```bash
    pip install torch torchvision matplotlib numpy
    ```

3. **Run the notebook:**
    - Open `CNN-MNIST.ipynb` in Jupyter Notebook or VS Code.
    - Execute cells step by step to train and analyze the model.

## Results

- The model achieves high validation accuracy (typically >97%).
- Loss decreases steadily, indicating effective learning.
- Most misclassifications occur on ambiguous digits.
- The trained model is saved as `cnn_mnist_model.pth`.

## File Structure

- `CNN-MNIST.ipynb` — Main notebook with code and explanations.
- `cnn_mnist_model.pth` — Saved PyTorch model (after training).
- `Readme.md` — Project documentation.

## Credits

Notebook created by [Yasin Pourraisi], September 2025.

**Contact:**
- GitHub: [yasinpurraisi](https://github.com/yasinpurraisi)
- Email: yasinpourraisi@gmail.com
- Telegram: [yasinprsy](https://t.me/yasinprsy)

---

**License:** MIT
