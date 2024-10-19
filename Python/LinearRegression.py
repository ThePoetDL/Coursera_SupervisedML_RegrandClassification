import random
import matplotlib.pyplot as plt

# Basic linear regression
# Goal is to generate some basic randomized data around a linear function of form y = mx + b
# and then fit it

# Step 1: Generate N = 100 data points around function y= mx +b
m = 2 # trueM
b = 3 # trueB
N = 100 # number of randomly generated data points
alpha_m = 0.000001 # learning rate alpha
alpha_b = 0.000001 # learning rate alpha
# random.seed(0) # Set seed for reproducibility
numIterations = 1000 # number of iterations of gradient descent

# Helper function to calculate regression cost as
# squared error for one data point x[i], y[i]
def calculate_costMSE(x, y, m, b):
    return sum(((yi - (m * xi + b)) ** 2) for xi, yi in zip(x, y)) / (2 * N)

# Personal note, given a loss function of the form
# L(m,b) = calculate_costSSE = sum ( (yi - (m*xi + b))^2 )
# where xi, yi ith x and y data points
# the partial derivative of L(m, b) with respect to m is
# 1/2N * sum(-2x_i*(y_i - (m*x_i + b))) = -1/N * sum(x_i*(y_i - (m*x_i + b)))
# and the partial derivative of L(m, b) with respect to b is
# 1/2N * sum(-2*(y_i - (m*x_i + b))) = -1/N * sum((y_i - (m*x_i + b)))

# Pick random starting m, b guess values between -50 and +50
mGuess = random.uniform(-50, 50)
bGuess = random.uniform(-50, 50)

# Generate 100 data points around function y = mx + b, with domain (0,500) with randomness around +/-20% of y
x = [random.uniform(0, 500) for i in range(N)]
y = [m * xi + b + random.uniform(-100, 100)    for xi in x]

# Let's implement gradient descent, saving each m, b combination
# along the way, in a key:value pair
# add another parallel list that stores the SSE for that mb combination
mbs = {}
mbs[0] = [mGuess, bGuess]

# loop 50 times using gradient descent
# given the ith iteration of m, b, update to the next iteration of m, b using gradient descent
# start by writing partial derivative helper functions for m and b

print("true m, b: ", m, ", ", b)
print("starting m, b: ", mbs[0])

for i in range(1, numIterations+1):
    currentm = mbs[i-1][0]
    currentb = mbs[i-1][1]
    dm = -1/N * sum(x_i*(y_i - (currentm*x_i + currentb)) for x_i, y_i in zip(x,y)) # partial derivative of loss function with respect to m
    db = -1/N * sum((y_i - (currentm*x_i + currentb)) for x_i, y_i in zip(x,y)) # partial derivative of loss function with respect to m

    # update m, b values
    mbs[i] = [mbs[i-1][0] - alpha_m * dm, mbs[i-1][1] - alpha_b * db]

    #print newest m, b values
    # print("Iteration ", i, " m, b: ", mbs[i])

# print final m, b values
print("Final m, b values: ", mbs[numIterations])

# graph [x] and [y] generated numbers
plt.scatter(x, y)
plt.savefig('scatter_plot.png')
plt.close()
# plt.show()
#scatter plot of m, b trajectory
mValues = [mbs[i][0] for i in range(numIterations+1)]
bValues = [mbs[i][1] for i in range(numIterations+1)]
plt.scatter(mValues, bValues)
plt.savefig('m_b_trajectory.png')