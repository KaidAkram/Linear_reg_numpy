# Linear Regression from Scratch using NumPy

This project implements Linear Regression using only NumPy, without relying on machine learning libraries like Scikit-Learn. Below is an explanation of the mathematical concepts used.

## 1. Problem Definition
Linear regression models the relationship between a dependent variable \( y \) and an independent variable \( X \) using the equation:
\[
y = W X + b\]
where:
- \( W \) (weight) is the coefficient
- \( b \) is the bias (intercept)
- \( X \) is the input feature
- \( y \) is the predicted output

## 2. Generating Data
We generate data using:
\[
y = 4 + 3X + \varepsilon\]
where \( \varepsilon \) is Gaussian noise.

## 3. Cost Function
The **Mean Squared Error (MSE)** is used to measure the difference between predicted and actual values:
\[
J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
\]
where \( m \) is the number of data points, and \( \hat{y} = WX + b \) is the predicted output.

## 4. Gradient Descent
To minimize the cost function, we use **gradient descent**, updating \( W \) and \( b \) iteratively:
\[
W := W - \alpha \frac{\partial J}{\partial W}
\]
\[
b := b - \alpha \frac{\partial J}{\partial b}
\]
where the gradients are computed as:
\[
\frac{\partial J}{\partial W} = \frac{1}{m} X^T (\hat{y} - y)
\]
\[
\frac{\partial J}{\partial b} = \frac{1}{m} \sum (\hat{y} - y)
\]
and \( \alpha \) is the learning rate.

## 5. Implementation
- **`cost_func(W, b, X, y)`** computes the cost function.
- **`gradient_descent(W, b, alpha, iterations, X, y)`** updates parameters iteratively.

## 6. Running the Model
1. Generate synthetic data.
2. Initialize weights randomly.
3. Compute cost and optimize parameters using gradient descent.
4. Print final values of \( W \) and \( b \).

## 7. Visualization
You can plot the cost history to verify convergence:
```python
plt.plot(cost_his)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()
```

## 8. Conclusion
This implementation demonstrates how linear regression works mathematically and how gradient descent optimizes the parameters efficiently.
