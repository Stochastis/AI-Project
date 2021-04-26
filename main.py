import matplotlib.pyplot as plt
import numpy as np
import xlrd


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

    def train(self, _input_vectors, _targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random.
            random_data_index = np.random.randint(len(_input_vectors))

            input_vector = _input_vectors[random_data_index]
            target = _targets[random_data_index]

            # Compute the gradients and update the weights.
            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances.
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error.
                for data_instance_index in range(len(_input_vectors)):
                    data_point = _input_vectors[data_instance_index]
                    target = _targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error += error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors


book = xlrd.open_workbook("dataWorkbook.xlsx")
input_vectors = [[4, 5], [4, 6], [5, 4], [3, 7], [7, 3], [8, 4], [6, 4], [0, 9], [8, 0], [4, 6], [8, 4], [8, 9], [7, 3],
                 [5, 4], [7, 8], [8, 9], [9, 6], [0, 4], [7, 0], [0, 0], [5, 5]]
targets = [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 100000)
print(neural_network.predict([6, 5]))

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")
