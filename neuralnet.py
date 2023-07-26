import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class HiddenLayer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.randn(num_neurons, num_inputs) * 0.01
        self.biases = np.random.randn(num_neurons, 1) * 0.01
        self.inputs, self.outputs = None, None
        self.dw = np.zeros(self.weights.shape)
        self.db = np.zeros(self.biases.shape)

    def activation_function(self, x):
        return x

    def activation_function_derivative(self, x):
        return 1

    def assign_inputs(self, inputs):
        self.inputs = inputs

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
        self.y_hat = None

    def assign_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def forward(self, X):
        for layer in self.hidden_layers:
            print(X[:10])
            layer.assign_inputs(X)
            layer.calculate_outputs()
            X = layer.get_outputs()

        self.y_hat = self.softmax(X)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def get_outputs(self):
        return self.output

    def calculate_cost(self, y):
        m = self.inputs.shape[1]
        total_loss = np.sum(y * np.log(self.y_hat))
        cost = -1/m * total_loss
        return cost

    def backpropagation(self, y):
        m = self.X_train.shape[1]
        dz2 = self.y_hat - y
        dw2 = 1/m * np.dot(dz2, self.hidden_layers[1].outputs.T)
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)

        g1_prime = self.hidden_layers[0].activation_function_derivative(self.hidden_layers[0].outputs)
        dz1 = np.dot(self.hidden_layers[1].weights.T, dz2) * g1_prime
        dw1 = 1/m * np.dot(dz1, self.X_train.T)
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

        self.hidden_layers[1].dw, self.hidden_layers[1].db = dw2, db2
        self.hidden_layers[0].dw, self.hidden_layers[0].db = dw1, db1

    def update_weights(self, learning_rate):
        for layer in self.hidden_layers:
            layer.weights -= learning_rate * layer.dw
            layer.biases -= learning_rate * layer.db

    def train_model(self, learning_rate, num_iterations):
        for i in range(num_iterations):
            print(i+1)
            self.forward(self.X_train)
            self.backpropagation(self.y_train)
            self.update_weights(learning_rate)

    def make_prediction(self, x): #!! Change/fix code. make_prediction takes single data point, not entire dataset
        # self.forward(x) # Do not call self.forward(self.X) here
        for layer in self.hidden_layers:
            layer.assign_inputs(x)
            layer.calculate_outputs()
            x = layer.get_outputs()

        self.output = self.softmax(x)
        return self.output

def make_one_hot(y):
    one_hot = np.zeros((y.size, y.max()+1))
    one_hot[np.arange(y.size), y] = 1
    return one_hot.T

if __name__ == '__main__':
    data = pd.read_csv('./emnist-mnist-train.csv', header=None)
    data_arr = np.array(data)
    np.random.shuffle(data_arr)

    X_train, X_test = data_arr[:25000].T, data_arr[25000: 30000].T
    y_train_, y_test_ = np.reshape(X_train[0], (X_train[0].shape[0], 1)).T, np.reshape(X_test[0], (X_test[0].shape[0], 1)).T
    y_train, y_test = make_one_hot(y_train_), make_one_hot(y_test_)
    X_train, X_test = X_train[1:], X_test[1:]

    nn = NeuralNetwork([ReLULayer(784, 10), ReLULayer(10, 10)])
    nn.assign_training_data(X_train, y_train)
    nn.train_model(0.01, 250)

    test_pred = nn.make_prediction(X_test)
    print(test_pred.shape, y_test.shape)
    print(test_pred[:10].max(axis=0))
    print(test_pred[:10])
    print(y_test[:10])

    # inputs = [[2, 3, 4, 5],
    #           [-1234, 4000, -2, 8],
    #           [10, 25, 50, 51]]
    # inputs = np.array(inputs).T