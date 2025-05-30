# MNIST CNN Classification

A PyTorch implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset.

## Overview

This project implements a CNN architecture to classify handwritten digits (0-9) from the MNIST dataset. The model achieves high accuracy through a combination of convolutional layers, max pooling, and fully connected layers.

## Model Architecture

The CNN consists of:
- 2 Convolutional layers with ReLU activation
- 2 Max Pooling layers
- 2 Fully Connected layers
- Final output layer with 10 classes (digits 0-9)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- CUDA (optional, for GPU acceleration)
- Jupyter Notebook (for running the analysis notebook)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cnn-mnist.git
cd cnn-mnist
```

2. Install the required packages:
```bash
pip install torch torchvision jupyter
```

## Usage

### Training the Model

Run the training script:
```bash
python run_experiments.py
```

The script will:
1. Download the MNIST dataset (if not already present)
2. Train the CNN model
3. Evaluate the model on the test set
4. Display training progress and final accuracy

### Model Analysis

The project includes a Jupyter notebook (`notebook.ipynb`) that provides:
- Detailed analysis of the model's architecture
- Visualization of tensor dimensions throughout the network
- Step-by-step inspection of how data flows through each layer

To run the notebook:
```bash
jupyter notebook notebook.ipynb
```

## Project Structure

- `run_experiments.py`: Main script containing the model definition and training loop
- `notebook.ipynb`: Interactive notebook for model analysis and dimension inspection
- `data/`: Directory where MNIST dataset is stored (automatically created)
- `.gitignore`: Specifies files to be ignored by git

## Model Performance

The model is trained with the following hyperparameters:
- Batch size: 64
- Learning rate: 0.001
- Epochs: 5
- Optimizer: Adam

## License

This project is open source and available under the MIT License.