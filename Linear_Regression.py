import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data = np.loadtxt('hw2data1.txt', delimiter=',')

X = data[:, 0]
y = data[:, 1]

# Add intercept term (column of ones)
X_intercept = np.c_[np.ones(X.shape[0]), X]

# using normal equation to get the val of theta
theta = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y


x_range = np.linspace(X.min(), X.max(), 100)
x_range_intercept = np.c_[np.ones(x_range.shape[0]), x_range]
y_pred = x_range_intercept @ theta


plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', marker='x', label='Training data')
plt.plot(x_range, y_pred, label='Linear Regression Fit', color='blue')
plt.xlabel('Population of City')
plt.ylabel('Profit')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()

# Print final theta values
print("Theta values after SGD (θ):")
print(f"Intercept (θ₀): {theta[0]:.4f}")
print(f"Slope (θ₁): {theta[1]:.4f}")