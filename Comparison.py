import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data = np.loadtxt('hw2data1.txt', delimiter=',')

X = data[:, 0]
y = data[:, 1]

# Add column of ones
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

# # Print final theta values
# print(f"Intercept (θ₀): {theta[0]:.4f}")
# print(f"Slope (θ₁): {theta[1]:.4f}")

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################


#CODE FOR Q1.2
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('hw2data1.txt', delimiter=',')

X = data[:, 0]  
y = data[:, 1]  

X_intercept = np.c_[np.ones(X.shape[0]), X]

# Hyperparameters for SGD
alpha = 0.01  
epochs = 1000  
m = X.shape[0] 

theta = np.zeros(2)

# Stochastic Gradient Descent (SGD)
for epoch in range(epochs):
    for i in range(m):  
        xi = X_intercept[i, :]  
        yi = y[i] 
        error = (xi @ theta) - yi  
        theta -= alpha * error * xi

# predictions using learned theta
x_range = np.linspace(X.min(), X.max(), 100)
x_range_intercept = np.c_[np.ones(x_range.shape[0]), x_range]
y_pred = x_range_intercept @ theta



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

# print(f"Intercept (θ₀): {theta[0]:.4f}")
# print(f"Slope (θ₁): {theta[1]:.4f}")

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

#CODE FOR Q1.3
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('hw2data1.txt', delimiter=',')

X = data[:, 0]  
y = data[:, 1]

# Extend feature vector to include polynomial terms up to degree 4
# I decided to do this manually instead of using the polyfeatures function from the pdf.
X_poly = np.c_[X, X**2, X**3, X**4]  # Polynomial features up to x^4

# Normalizing the full poly feature set
X_poly_mean = np.mean(X_poly, axis=0)
X_poly_std = np.std(X_poly, axis=0)
X_poly = (X_poly - X_poly_mean) / X_poly_std  # Standardization

X_poly_intercept = np.c_[np.ones(X.shape[0]), X_poly]

#Hyperparameters
alpha = 0.0001  
epochs = 1000  
lambda_values = [0.01, 0.1, 1, 10]  
m, n = X_poly_intercept.shape  

#hashmap to store results for diff lambda vals
theta_results = {}

#Training the model for diff regularization vals 
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', marker='x', label='Training data')

for lambda_reg in lambda_values:
    theta = np.random.randn(n) * 0.01

    #SGD using L2 Regularization (Ridgge Regression)
    for epoch in range(epochs):
        for i in range(m):  
            xi = X_poly_intercept[i, :]  
            yi = y[i]  
            error = (xi @ theta) - yi  
            gradient = error * xi + (lambda_reg / m) * theta  
            theta -= alpha * gradient  

    theta_results[lambda_reg] = theta

    #predictions using learned theta
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

# plt.savefig('polynomial_regression_plot.png', dpi=300)
plt.show()