import numpy as np
import matplotlib.pyplot as plt

# Basic logistic regression practice
# Goal is to generate some basic randomized data around a
# logistic function of the form y = a/ (1 + e^-bx)
# and then fit it

# Parameters
aReal, bReal = 3, 6  # true parameters
numDataPoints = 500 # number of randomly generated data points in fake data
alpha = 0.09 # learning rate
np.random.seed(0) # Set seed for reproducibility
numIterations = 10000 # number of iterations of gradient descent
xDomain = 10
yRandomNoise = 0.1

#define helper sigmoid function
def sigmoid(z, a, b):
    """
    sigmoid function for logistic regression
    """
    return 1 / (1 + np.exp(-(a*z + b)))

def binary_cross_entropy(y_true, y_pred):
    """
    Calculate finary cross-entropy loss
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient_descent_step(x, y, a, b, learning_rate):
    """
    Performs a single step of gradient descent for the logistic regression problem.

    Args:
        x (numpy.ndarray): Input data.
        y (numpy.ndarray): Target data.
        a (float): Current estimate of the amplitude parameter.
        b (float): Current estimate of the shift parameter.
        learning_rate (float): Learning rate for gradient descent.

    Returns:
        tuple: Updated values of a and b.
    """
    # Calculate the predictions
    m = len(x)
    predictions = sigmoid(x, a, b)

    # Compuete gradients
    da = (1/m) * np.sum(x * (predictions - y))
    db = (1/m) * np.sum(predictions - y)

    a_new = a - alpha * da
    b_new = b - alpha * db

    return a_new, b_new

# Initialize model parameters
aGuess, bGuess = np.random.uniform(0, 10, 2)

# Generate data points
x = np.random.uniform(-xDomain, xDomain, numDataPoints)
y = sigmoid(x, aReal, bReal) + np.random.uniform(-yRandomNoise, yRandomNoise, numDataPoints)

# Initialize arrays to store parameter values
a_values = np.zeros(numIterations)
b_values = np.zeros(numIterations)
a_values[0], b_values[0] = aGuess, bGuess

print(f"True a, b: {aReal}, {bReal}")
print(f"Starting guess a, b: {aGuess:.6f}, {bGuess:.6f}")

# loop numIterations times using gradient descent
# given the ith iteration of a, b, update to the next iteration of a, b using gradient descent
# start by writing partial derivFalseative helper functions for m and b


for i in range(1, numIterations):
    a_values[i], b_values[i] = gradient_descent_step(x, y, a_values[i-1], b_values[i-1], alpha)

    #print every 100 iterations  a, b values
    if i % 100 == 0:
        predictions = sigmoid(x, a_values[i], b_values[i])
        cost = binary_cross_entropy(y, predictions)
        print(f"Iteration {i}, Cost: {cost:.6f}, a: {a_values[i]:.6f}, b: {b_values[i]:.6f}")

# print final a, b values
print(f"Final a, b values: {a_values[-1]:.6f}, {b_values[-1]:.6f}")

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(x, y, alpha=0.5)
plt.title('Data Points')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.scatter(a_values, b_values, alpha=0.5)
plt.title('Parameter Trajectory')
plt.xlabel('a')
plt.ylabel('b')

plt.tight_layout()
plt.savefig('logistic_regression_plots.png')
plt.close()