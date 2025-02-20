# Linear Regression from Scratch with NumPy

This repository contains an implementation of linear regression from scratch using NumPy. The goal is to provide a clear and concise explanation of the mathematical concepts behind linear regression and how to implement it using basic numerical operations.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Concepts](#mathematical-concepts)
3. [Implementation](#implementation)
4. [Usage](#usage)
5. [Example](#example)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

Linear regression is a fundamental statistical and machine learning technique used for predicting a continuous target variable based on one or more input features. This implementation uses NumPy to perform the necessary computations, providing a clear understanding of the underlying mathematics.

## Mathematical Concepts

### Linear Regression Model

The linear regression model can be represented as:

\[ h_W(x) = W^T x + b \]

where:
- \( h_W(x) \) is the predicted value.
- \( W \) is the vector of weights.
- \( x \) is the input feature vector.
- \( b \) is the bias term.

### Cost Function

The cost function measures the difference between the predicted values and the actual values. For linear regression, the Mean Squared Error (MSE) is commonly used:

\[ J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (h_W(x^{(i)}) - y^{(i)})^2 \]

where:
- \( m \) is the number of training examples.
- \( y^{(i)} \) is the actual value for the \( i \)-th example.

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function. The update rules for the parameters are:

\[ W := W - \alpha \frac{\partial J(W, b)}{\partial W} \]
\[ b := b - \alpha \frac{\partial J(W, b)}{\partial b} \]

where:
- \( \alpha \) is the learning rate.
- \( \frac{\partial J(W, b)}{\partial W} \) and \( \frac{\partial J(W, b)}{\partial b} \) are the partial derivatives of the cost function with respect to \( W \) and \( b \), respectively.

### Partial Derivatives

The partial derivatives are computed as:

\[ \frac{\partial J(W, b)}{\partial W} = \frac{1}{m} X^T (h_W(X) - y) \]
\[ \frac{\partial J(W, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h_W(x^{(i)}) - y^{(i)}) \]

## Implementation

The implementation includes the following functions:

- `compute_cost(X, y, W, b)`: Computes the cost function.
- `gradient_descent(X, y, W, b, learning_rate, iterations)`: Performs gradient descent to optimize the parameters.
- `predict(X, W, b)`: Makes predictions using the learned parameters.

## Usage

To use the linear regression implementation, follow these steps:

1. Prepare your dataset as NumPy arrays `X` (features) and `y` (target).
2. Initialize the parameters `W` and `b`.
3. Call the `gradient_descent` function to train the model.
4. Use the `predict` function to make predictions on new data.

## Example

```python
import numpy as np

# Example dataset
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])

# Initialize parameters
W = np.zeros((2, 1))
b = 0

# Train the model
W, b, cost_history = gradient_descent(X, y, W, b, learning_rate=0.01, iterations=1000)

# Make predictions
X_test = np.array([[3, 5]])
predictions = predict(X_test, W, b)
print("Predictions:", predictions)
