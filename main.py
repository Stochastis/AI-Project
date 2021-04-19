import numpy as np

# Wrapping the vectors in NumPy arrays
inputVector = np.array([1.66, 1.56])
weights1 = np.array([1.45, -0.66])
bias = np.array([0.0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_prediction(input_vector, weights, bias_param):
    layer1 = np.dot(input_vector, weights) + bias_param
    layer2 = sigmoid(layer1)
    return layer2


prediction = make_prediction(inputVector, weights1, bias)

print(f"The prediction result is: {prediction}")
