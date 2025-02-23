# DeepDigitor: MLP from Scratch

DeepDigitor is a neural network project designed to classify handwritten digits using the MNIST dataset. This project implements a fully manual Multilayer Perceptron (MLP) without using any pre-built deep learning frameworks like TensorFlow, Keras, PyTorch, or Scikit-learn. Instead, the MLP architecture is constructed from scratch using basic Python libraries such as CuPy to perform operations like matrix multiplication, activation functions, and backpropagation. The goal is to understand and demonstrate the inner workings of neural networks in a transparent and educational manner.

## Features

We implemented and explained the following components from scratch, with full transparency:
- **Perceptron**: The basic building block of the network.
- **Layer**: Fully connected layers built manually.
- **Loss Function**: Custom implementation to measure prediction error.
- **Activation Function**: Hand-coded functions like sigmoid, ReLU, etc.
- **MLP**: The complete multilayer perceptron architecture.
- **Batch**: Mini-batch processing for efficient training.
- **Epoch**: Iterative training over the dataset.
- **Dropout**: Regularization technique to prevent overfitting.
- **L1 and L2 Regularization**: Manual implementation for model regularization.
- **Forward and Backward Pass**: Core algorithms for training the network.
- **Metrics**: Accuracy and F1 score, calculated from scratch.

Each component is fully explained to provide an educational deep dive into how neural networks function under the hood.

## Purpose

This project is intended for anyone looking to:
- Learn the fundamentals of neural networks without relying on black-box frameworks.
- Understand the mathematics and programming behind MLPs.
- Experiment with a lightweight, transparent implementation of a digit classifier.

## Contributing

Feel free to contribute by submitting pull requests, suggesting improvements, or adding new features.

## Usage

1. Clone the repository.
2. Install the required dependencies (e.g., CuPy, Matplotlib).
3. Run the main script to train the MLP on the MNIST dataset and evaluate its performance.

If you find this project useful, please give it a star to show your support!⭐

Happy coding, and enjoy exploring the world of neural networks!
