import numpy as np
import matplotlib.pyplot as plt


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
    inputs = [[2, 3, 4, 5],
              [1234, 4000, -2, 8],
              [10, 25, 50, 51]]

    h1 = HiddenLayer(4, 3)
    h2 = ReLULayer(3, 4)
    h3 = SigmoidLayer(4, 2)

    nn = NeuralNetwork([h1, h2, h3])
    nn.assign_inputs(inputs)
    nn.forward()
    print(nn.get_outputs())