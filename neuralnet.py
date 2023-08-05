import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class HiddenLayer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.randn(num_neurons, num_inputs) * 0.01
        self.biases = np.zeros((num_neurons, 1))
        self.A, self.z = None, None
        self.dW = np.zeros(self.weights.shape)
        self.dB = np.zeros(self.biases.shape)
        self.dA, self.dZ = None, None
        self.mt_w, self.mt_b = 0, 0
        self.vt_w, self.vt_b = 0, 0

    def activation_function(self, x):
        return x

    def activation_function_derivative(self, x):
        return 1

    def calculate_outputs(self, X):
        self.z = np.dot(self.weights, X) + self.biases
        self.A = self.activation_function(self.z)

    def get_outputs(self):
        return self.A


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


class OutputLayer(HiddenLayer):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        # shift_x = x - np.max(x)
        # return np.exp(x) / np.sum(np.exp(shift_x), axis=0)

    def activation_function(self, x):
        if self.weights.shape[0] == 1:
            return self.sigmoid(x)
        else:
            return self.softmax(x)

    def activation_function_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))


class NeuralNetwork:
    def __init__(self, hidden_layers: List[HiddenLayer], num_classes: int):
        output_layer = OutputLayer(hidden_layers[-1].weights.shape[0], num_classes)
        hidden_layers.append(output_layer)
        self.hidden_layers = hidden_layers
        self.output = None
        self.X_train = None
        self.y_train = None
        self.y_hat = None

    def assign_training_data(self, X_train, y_train, normalize=False):
        if normalize:
            self.X_train = self.normalize(X_train)
        else:
            self.X_train = X_train
        self.y_train = y_train

    def normalize(self, X):
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        return (X - mean) / std

    def forward(self, X):
        for layer in self.hidden_layers:
            layer.calculate_outputs(X)
            X = layer.get_outputs()

        self.y_hat = X

    def get_outputs(self):
        return self.output

    def calculate_cost(self, X, y):
        m = X.shape[1]
        total_loss = np.sum(y * np.log(self.y_hat))
        cost = -1/m * total_loss
        return cost

    def backpropagation(self, X, y):
        m = X.shape[1]
        j = len(self.hidden_layers)

        self.hidden_layers[j-1].dZ = self.y_hat - y
        self.hidden_layers[j-1].dW = 1/m * np.dot(self.hidden_layers[j-1].dZ, self.hidden_layers[j-1].A.T)
        self.hidden_layers[j-1].dB = 1/m * np.sum(self.hidden_layers[j-1].dZ, axis=1, keepdims=True)
        self.hidden_layers[j-2].dA = np.dot(self.hidden_layers[j-1].weights.T, self.hidden_layers[j-1].dZ)

        for i, layer in enumerate(reversed(self.hidden_layers[1:-1])):
            layer.dZ = layer.dA * layer.activation_function_derivative(layer.z)
            layer.dW = 1/m * np.dot(layer.dZ, self.hidden_layers[j-i-2].A.T)
            layer.dB = 1/m * np.sum(layer.dZ, axis=1, keepdims=True)
            self.hidden_layers[j-i-2].dA = np.dot(layer.weights.T, layer.dZ)

        self.hidden_layers[0].dZ = self.hidden_layers[0].dA * self.hidden_layers[0].activation_function_derivative(self.hidden_layers[0].z)
        self.hidden_layers[0].dW = 1/m * np.dot(self.hidden_layers[0].dZ, X.T)
        self.hidden_layers[0].dB = 1/m * np.sum(self.hidden_layers[0].dZ, axis=1, keepdims=True)

    def update_weights(self, learning_rate, beta_1, beta_2):
        epsilon=10 ** -8
        for layer in self.hidden_layers:
            # layer.vt_w = beta_2 * layer.vt_w + (1 - beta_2) * np.square(layer.dW)
            # layer.vt_b = beta_2 * layer.vt_b + (1 - beta_2) * np.square(layer.dB)
            layer.mt_w = beta_1 * layer.mt_w + learning_rate * layer.dW
            layer.mt_b = beta_1 * layer.mt_b + learning_rate * layer.dB

            layer.weights -= layer.mt_w
            layer.biases -= layer.mt_b
            # layer.weights -= learning_rate * layer.dW / (np.sqrt(layer.vt_w) + epsilon)
            # layer.biases -= learning_rate * layer.dB / (np.sqrt(layer.vt_b) + epsilon)

    def train_model(self, training_type, num_iterations, learning_rate=0.01, momentum=0, ada_grad=0):
        print("Training using", training_type)
        print("Learning rate: ", learning_rate)
        print("Number of iterations: ", num_iterations)
        print("Momentum: ", momentum)
        if training_type == 'gradient descent':
            self.gradient_descent(learning_rate, num_iterations, momentum, ada_grad)
        elif training_type == 'sgd':
            self.sgd(learning_rate, num_iterations, 100, momentum)
        elif training_type == 'mini batch':
            batch_size = int(input("Enter batch size: "))
            self.mini_batch_gd(learning_rate, num_iterations, momentum, ada_grad, batch_size)

    def make_mini_batches(self, X, y, batch_size):
        m = X.shape[1]
        mini_batches = []

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_y = y[:, permutation]

        num_complete_batches = math.floor(m / batch_size)
        for i in range(0, num_complete_batches):
            mini_X = shuffled_X[:, i*batch_size : (i+1)*batch_size]
            mini_y = shuffled_y[:, i*batch_size : (i+1)*batch_size]
            mini_batches.append((mini_X, mini_y))

        if m % batch_size != 0:
            mini_X = shuffled_X[:, num_complete_batches*batch_size : m]
            mini_y = shuffled_y[:, num_complete_batches*batch_size : m]
            mini_batches.append((mini_X, mini_y))

        return mini_batches

    def gradient_descent(self, learning_rate, num_iterations, momentum, ada_grad):
        v_t = 0
        for i in range(num_iterations):
                print("Iteration: ", i+1)
                self.forward(self.X_train)
                self.backpropagation(self.X_train, self.y_train)
                self.update_weights(learning_rate, momentum, ada_grad)
                print("Cost :", self.calculate_cost(self.X_train, self.y_train))
        print("Final cost: ", self.calculate_cost(self.X_train, self.y_train))

    def mini_batch_gd(self, learning_rate, num_iterations, momentum, ada_grad, batch_size=100):
        m = self.X_train.shape[1]

        for i in range(num_iterations):
            print("Iteration: ", i+1)
            mini_batches = self.make_mini_batches(self.X_train, self.y_train, batch_size)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                self.forward(X_mini)
                self.backpropagation(X_mini, y_mini)
                self.update_weights(learning_rate, momentum, ada_grad)
                print("Cost :", self.calculate_cost(X_mini, y_mini))

    def sgd(self, learning_rate, num_iterations, batch_size=100):
        for i in range(num_iterations):
            print("Iterarion: ", i+1)
            for j in range(0, self.X_train.shape[1], batch_size):
                X_batch = self.X_train[:, j: j+batch_size]
                y_batch = self.y_train[:, j: j+batch_size]
                self.forward(X_batch)
                self.backpropagation(y_batch)
                self.update_weights(learning_rate)


def make_one_hot(y):
    one_hot = np.zeros((y.size, y.max()+1))
    one_hot[np.arange(y.size), y] = 1
    return one_hot.T

if __name__ == '__main__':
    # X_train_data = [[1, 2, 3, 2.5],
    #            [2.0, 5.0, -1.0, 2.0],
    #            [-1.5, 2.7, 3.3, -0.8]]

    # X_train = np.array(X_train_data)
    # y_train_ = np.array([2, 8, 5, 1]).T
    # y_train = make_one_hot(y_train_)

    # nn = NeuralNetwork([ReLULayer(3, 2)], 2)
    # nn.assign_training_data(X_train, y_train)

    # nn.train_model('gradient descent', 0.01)
    
    data = pd.read_csv('./emnist-mnist-train.csv', header=None)
    data_arr = np.array(data)
    np.random.shuffle(data_arr)

    X_train, X_test = data_arr[:20000].T, data_arr[20000: 30000].T
    y_train_, y_test_ = np.reshape(X_train[0], (X_train[0].shape[0], 1)).T, np.reshape(X_test[0], (X_test[0].shape[0], 1)).T
    y_train, y_test = make_one_hot(y_train_), make_one_hot(y_test_)
    X_train, X_test = X_train[1:], X_test[1:]
    print(X_train.shape)
    nn = NeuralNetwork([ReLULayer(784, 10)], num_classes=10)
    nn.assign_training_data(X_train, y_train)
    nn.train_model('gradient descent', 125, 0.01, momentum=0.9, ada_grad=0.9)

    nn.forward(X_test)
    test_pred = nn.y_hat
    print(test_pred[0].shape, y_test[0].shape)
    print(test_pred[:10].max(axis=0))
    print("Prediction: ", test_pred[:10].argmax(axis=0))
    print("Actual: ", y_test[:10].argmax(axis=0))

    acc = np.sum(test_pred.argmax(axis=0) == y_test.argmax(axis=0)) / y_test.shape[1]
    print(f"Accuracy: {acc*100}%")
