
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data = np.loadtxt('hw2data1.txt', delimiter=',')

# Separate into X (feature) and y (target)
X = data[:, 0]  # Population of the city
y = data[:, 1]

# Extend feature vector to include polynomial terms up to degree 4
X_poly = np.c_[X, X**2, X**3, X**4]  # Polynomial features up to x^4

# Normalize the full polynomial feature set
X_poly_mean = np.mean(X_poly, axis=0)
X_poly_std = np.std(X_poly, axis=0)
X_poly = (X_poly - X_poly_mean) / X_poly_std  # Standardization

# Add intercept term (column of ones)
X_poly_intercept = np.c_[np.ones(X.shape[0]), X_poly]

# Hyperparameters
alpha = 0.0001  # Further reduced learning rate for stability
epochs = 1000  # Number of passes over dataset
lambda_values = [0.01, 0.1, 1, 10]  # Different regularization strengths
m, n = X_poly_intercept.shape  # Number of samples, number of features

# Dictionary to store results for different lambda values
theta_results = {}

# Train polynomial regression model for different regularization values
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', marker='x', label='Training data')

for lambda_reg in lambda_values:
    # Initialize theta with small random values instead of zeros
    theta = np.random.randn(n) * 0.01

    # Stochastic Gradient Descent (SGD) with L2 Regularization (Ridge)
    for epoch in range(epochs):
        for i in range(m):  # Iterate through each data point
            xi = X_poly_intercept[i, :]  # Single training example
            yi = y[i]  # Corresponding target value
            error = (xi @ theta) - yi  # Compute error
            gradient = error * xi + (lambda_reg / m) * theta  # Regularized gradient
            theta -= alpha * gradient  # Update theta

    # Store theta values for this lambda
    theta_results[lambda_reg] = theta

    # Generate predictions using learned theta
    x_range = np.linspace(X.min(), X.max(), 100)
    X_range_poly = np.c_[x_range, x_range**2, x_range**3, x_range**4]  # Polynomial features
    X_range_poly = (X_range_poly - X_poly_mean) / X_poly_std  # Apply the same normalization
    X_range_poly_intercept = np.c_[np.ones(x_range.shape[0]), X_range_poly]  # Add intercept
    y_pred = X_range_poly_intercept @ theta  # Predicted values

    # Plot the polynomial regression fit for this lambda
    plt.plot(x_range, y_pred, label=f'λ={lambda_reg}')

# Plot settings
plt.xlabel('Population of City')
plt.ylabel('Profit')
plt.title('Polynomial Regression with Regularization (SGD)')
plt.legend()
plt.grid(True)

# Save plot as an image for LaTeX
# plt.savefig('polynomial_regression_plot.png', dpi=300)
plt.show()

# # Print final theta values for each lambda
# for lambda_reg, theta_vals in theta_results.items():
#     print(f"\nTheta values for λ = {lambda_reg}:")
#     for i, val in enumerate(theta_vals):
#         print(f"θ_{i}: {val:.4f}")