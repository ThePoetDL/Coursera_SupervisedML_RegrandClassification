import numpy as np
import time

#set seed for reproducibility = 50
np.random.seed(50)

n = 100000 # length of array

# Create two integer arrays of length 1000 populated with random numbers
arr1 = np.random.randint(0, 1000, size=n)
arr2 = np.random.randint(0, 1000, size=n)

# For loop to add up arrays
t0_forLoop = time.time()
sumForLoop = 0
for i in range(n):
    sumForLoop = + sumForLoop + arr1[i] * arr2[i]
t1_forLoop = time.time()

# Vectorized approach
t0_vectorized = time.time()
sumVectorized = np.dot(arr1, arr2)
t1_vectorized = time.time()

#Print results
print("For loop time: ", t1_forLoop - t0_forLoop)
print("vectorized approach time: ", t1_vectorized - t0_vectorized)
print("Ratio of time: ", (t1_forLoop - t0_forLoop) / (t1_vectorized - t0_vectorized))


