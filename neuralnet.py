import numpy as np
import matplotlib.pyplot as plt


class HiddenLayer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.rand(num_neurons, num_inputs)
        self.biases = np.random.rand(num_neurons, 1)
        self.dw = np.zeros(self.weights.shape)
        self.db = np.zeros(self.biases.shape)

    def activation_function(self, x):
        return x

    def activation_function_derivative(self, x):
        return 1

    def assign_inputs(self, inputs):
        self.inputs = np.array(inputs)

    def calculate_outputs(self):
        self.outputs = self.activation_function(np.dot(self.weights, self.inputs) + self.biases)

    def get_outputs(self):
        return self.outputs


class SigmoidLayer(HiddenLayer):
    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_function_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))


class ReLULayer(HiddenLayer):
    def activation_function(self, x):
        return np.maximum(0, x)

    def activation_function_derivative(self, x):
        return np.where(x > 0, 1, 0)


class NeuralNetwork:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.output = None
        self.X_train = None
        self.y_train = None

    def assign_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def assign_inputs(self, inputs):
        self.inputs = np.array(inputs)

    def forward(self, X):
        for layer in self.hidden_layers:
            layer.assign_inputs(X)
            layer.calculate_outputs()
            X = layer.get_outputs()
        
        self.output = self.softmax(X)

    def softmax(self, x):
        arr_ = np.sum(np.exp(x), axis=0)
        return np.exp(x) / arr_

    def get_outputs(self):
        return self.output

    def calculate_cost(self, y):
        m = self.inputs.shape[1]
        total_loss = np.sum(y * np.log(self.output))
        cost = -1/m * total_loss
        return cost

    def backpropagation(self, y):
        m = self.inputs.shape[1]
        dz2 = self.output - y
        dw2 = 1/m * np.dot(dz2, self.hidden_layers[1].outputs.T)
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)

        g1_prime = self.hidden_layers[0].activation_function_derivative(self.hidden_layers[0].outputs)
        dz1 = np.dot(self.hidden_layers[1].weights.T, dz2) * g1_prime
        dw1 = 1/m * np.dot(dz1, self.inputs.T)
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

        self.hidden_layers[1].dw, self.hidden_layers[1].db = dw2, db2
        self.hidden_layers[0].dw, self.hidden_layers[0].db = dw1, db1

    def update_weights(self, learning_rate):
        for layer in self.hidden_layers:
            layer.weights -= learning_rate * layer.dw
            layer.biases -= learning_rate * layer.db

    def gradient_descent(self, learning_rate, num_iterations):
        for i in range(num_iterations):
            self.forward(self.X_train)
            self.backpropagation(self.y_train)
            self.update_weights(learning_rate)

    def make_prediction(self, X): #!! Change/fix code. make_prediction takes single data point, not entire dataset
        self.forward(X) # Do not call self.forward(self.X) here
        return self.output

if __name__ == '__main__':
    inputs = [[2, 3, 4, 5],
              [-1234, 4000, -2, 8],
              [10, 25, 50, 51]]
    inputs = np.array(inputs).T

    h1 = ReLULayer(4, 3)
    h2 = SigmoidLayer(3, 3)
    # h3 = SigmoidLayer(4, 2)

    nn = NeuralNetwork([h1, h2])
    nn.assign_inputs(inputs)
    nn.forward(inputs)
    print(nn.get_outputs())
    nn.backpropagation(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    print(nn.calculate_cost(np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])))