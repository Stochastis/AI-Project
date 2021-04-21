import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _sigmoid_derivative(x):
    return _sigmoid(x) * (1 - _sigmoid(x))


class NeuralNetwork:
    def __init__(self, learning_rate_p):
        # Initialize random values for the weights and bias and set the learning rate.
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate_p

    def predict(self, input_vector):
        # Layer 1 is the difference between the input and the weights plus the bias.
        # Layer 2 is the activation layer. It runs layer 1 through a sigmoid function.
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        print(f'DEBUG: input_vector = {input_vector}')
        print(f'DEBUG: self.weights = {self.weights}')
        print(f'DEBUG: np.dot(input_vector, self.weights) = {np.dot(input_vector, self.weights)}')
        print(f'DEBUG: self.bias = {self.bias}')
        print(f'DEBUG: layer_1 = {layer_1}')
        layer_2 = _sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        # Makes a prediction and calculates the derivatives.
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = _sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)  # Derivative of error function.
        dprediction_dlayer1 = _sigmoid_derivative(layer_1)  # Derivative of sigmoid function.
        dlayer1_dbias = 1  # Derivative of bias.
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
                derror_dweights * self.learning_rate
        )


learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
print(f'DEBUG: prediction = {neural_network.predict([1, 2])}')
