import numpy as np

# Wrapping the vectors in NumPy arrays
inputVector = np.array([2, 1.5])
weights = np.array([1.45, -0.66])
bias = np.array([0.0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_prediction(input_vector, weights_param, bias_param):
    layer1 = np.dot(input_vector, weights_param) + bias_param
    layer2 = sigmoid(layer1)
    return layer2


prediction = make_prediction(inputVector, weights, bias)

target = 0  # The target output
mse = np.square(prediction - target)  # Calculate the Mean Squared Error(MSE)
print(f"Prediction before adjustment: {prediction}; Error: {mse}")

# Derivative will be used to determine how to adjust the weights.
derivative = 2 * (prediction - target)
print(f"The derivative is {derivative}")

# Change the weights according to the derivative.
weights -= derivative
prediction = make_prediction(inputVector, weights, bias)
mse = np.square(prediction - target)
print(f"Prediction after adjustment: {prediction}; Error: {mse}")
