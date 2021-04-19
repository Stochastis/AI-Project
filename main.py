import numpy as np

inputVector = [1.72, 1.23]
weights1 = [1.26, 0]
weights2 = [2.17, 0.32]

# Computing the dot product of inputVector and weights1
dotProduct1 = np.dot(inputVector, weights1)
dotProduct2 = np.dot(inputVector, weights2)

print(f"The first dot product is: {dotProduct1}")
print(f"The second dot product is: {dotProduct2}")
