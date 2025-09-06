# Neural Nets From Scratch

This repository implements a simple feedforward neural network framework in Python using only NumPy. It is designed for educational purposes, demonstrating the core mechanics of neural networks, including forward and backward propagation, custom activation functions, and training on the MNIST dataset.

## Features

- Modular layer and activation function design
- Custom implementations of Sigmoid, Softmax, and ReLU activations
- Batch training with backpropagation
- Model saving and loading via pickle
- One-hot encoding and normalization for MNIST data
- Training and evaluation with cross-tabulation of predictions

## Getting Started

### Prerequisites

- Python 3.8+
- Poetry for dependency management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-nets-from-scratch.git
cd neural-nets-from-scratch
```
2. Install dependencies
```bash
poetry install
```
3. Download or place the MNIST CSV files in the `mnist-in-csv` directory.

### Usage

Train and evaluate the model on MNIST:
```bash
poetry run python neural_nets_from_scratch/main.py
```
This will:

- Load and preprocess the MNIST data
- Train a neural network with two hidden layers
- Save the trained model to model.pickle
- Print a confusion matrix of predictions vs. actual labels

### Customization

- Modify `main.py` to change the network architecture, learning rate, batch size, or number of epochs.
- Implement additional activation functions or layers in `activation_function.py` and `layer.py`.

## Code Overiew

- `DenseLayer`: Implements a fully connected layer with customizable activation.
- `SequentialModel`: Manages layers, training, forward/backward passes, and model persistence.
- `ActivationFunction`: Abstract base for activation functions; includes Sigmoid, Softmax, and ReLU.
- `utils.py`: Contains data shuffling utilities for batch training.
