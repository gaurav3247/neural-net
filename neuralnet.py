import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, weights, bias):
        weights_ = np.array(weights)
        self.weights = weights_.reshape((len(weights), 1))
        # print(self.weights.shape)
        self.bias = bias
        self.output = None

    def activation_function(self, x):
        return x

    def activation_function_derivative(self, x):
        return 1

    def calculate_output(self, inputs):
        self.output = self.activation_function(np.dot(inputs, self.weights) + self.bias)

    def get_output(self):
        return self.output


class SigmoidNeuron(Neuron):
    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_function_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))


class ReLUNeuron(Neuron):
    def activation_function(self, x):
        return max(0, x)

    def activation_function_derivative(self, x):
        if x > 0:
            return 1
        return 0

class HiddenLayer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.rand(num_inputs, num_neurons)
        self.biases = np.random.rand(1, num_neurons)

    def activation_function(self, x):
        return x

    def assign_inputs(self, inputs):
        self.inputs = np.array(inputs)

    def calculate_outputs(self):
        self.outputs = self.activation_function(np.dot(self.inputs, self.weights) + self.biases)

    def get_outputs(self):
        return self.outputs


class SigmoidLayer(HiddenLayer):
    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))


class ReLULayer(HiddenLayer):
    def activation_function(self, x):
        return np.maximum(0, x)


class NeuralNetwork:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.output = None

    def assign_inputs(self, inputs):
        self.inputs = np.array(inputs)

    def forward(self):
        X = self.inputs
        for layer in self.hidden_layers:
            layer.assign_inputs(X)
            layer.calculate_outputs()
            X = layer.get_outputs()

        self.output = self.softmax(X)

    def softmax(self, x):
        arr_ = np.sum(np.exp(x), axis=1).reshape((x.shape[0], 1))
        return np.exp(x) / arr_

    def get_outputs(self):
        return self.output

if __name__ == '__main__':
    inputs = [[.001, .002, .003, .0025],
              [2, 5, -1, 2],
              [-1.5, 2.7, 3.3, -0.8]]
    # inputs = [[1, 2, -1.5],
    #           [2, 5, 2.7],
    #           [3, -1, 3.3],
    #           [2.5, 2, -0.8]]

    h1 = HiddenLayer(4, 3)
    h2 = ReLULayer(3, 4)
    h3 = SigmoidLayer(4, 2)

    nn = NeuralNetwork([h1, h2, h3])
    nn.assign_inputs(inputs)
    nn.forward()
    print(nn.get_outputs())