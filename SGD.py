import numpy as np
import matplotlib.pyplot as plt

# Load data from file (Ensure hw2data1.txt is in the same directory)
data = np.loadtxt('Linear_regression_sgd_dataset.txt', delimiter=',')

# Separate into X (features) and y (targets)
X = data[:, 0]  # Population of the city
y = data[:, 1]  # Profit of the food truck

# Add intercept term (column of ones)
X_intercept = np.c_[np.ones(X.shape[0]), X]

# Hyperparameters for SGD
alpha = 0.01  # Learning rate
epochs = 1000  # Number of passes over dataset
m = X.shape[0]  # Number of training samples

# Initialize theta (parameters) as zeros
theta = np.zeros(2)

# Stochastic Gradient Descent (SGD)
for epoch in range(epochs):
    for i in range(m):  # Loop over each training example
        xi = X_intercept[i, :]  # Single training example
        yi = y[i]  # Corresponding target value
        error = (xi @ theta) - yi  # Compute error
        theta -= alpha * error * xi  # Update theta

# Generate predictions using learned theta
x_range = np.linspace(X.min(), X.max(), 100)
x_range_intercept = np.c_[np.ones(x_range.shape[0]), x_range]
y_pred = x_range_intercept @ theta

# Plot results
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', marker='x', label='Training data')
plt.plot(x_range, y_pred, label='SGD Linear Regression Fit', color='blue')
plt.xlabel('Population of City')
plt.ylabel('Profit')
plt.title('Stochastic Gradient Descent - Linear Regression')
plt.legend()
plt.grid(True)

# plt.savefig('sgd_linear_regression_plot.png', dpi=300)
plt.show()

# # Print final theta values
# print("Theta values after SGD (θ):")
# print(f"Intercept (θ₀): {theta[0]:.4f}")
# print(f"Slope (θ₁): {theta[1]:.4f}")