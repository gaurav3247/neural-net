import math
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt

class HiddenLayer:
    def __init__(self, num_inputs, num_neurons, activation='relu', dropout=0.0):
        self.layer_type = activation
        function_dict = {'relu': self.relu_activation, 'sigmoid': self.sigmoid_activation, 'linear': self.linear_activation, 'tanh': self.tanh_activation, 'softmax': self.softmax_activation}
        derivative_dict = {'relu': self.relu_derivative, 'sigmoid': self.sigmoid_derivative, 'linear': self.linear_derivative, 'tanh': self.tanh_derivative, 'softmax': self.sigmoid_derivative}
        self.activation = function_dict[activation]
        self.activation_derivative = derivative_dict[activation]
        self.weights = np.random.randn(num_neurons, num_inputs) * 0.01
        self.biases = np.zeros((num_neurons, 1))
        self.dropout = dropout
        self.A, self.z = None, None
        self.dW = np.zeros(self.weights.shape)
        self.dB = np.zeros(self.biases.shape)
        self.dA, self.dZ = None, None
        self.mt_w, self.mt_b = 0, 0
        self.vt_w, self.vt_b = 0, 0
        self.best_weights = self.weights
        self.best_biases = self.biases
        self.D = 1

    def activation_function(self, x):
        return self.activation(x)

    def activation_function_derivative(self, x):
        return self.activation_derivative(x)

    def linear_activation(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))

    def softmax_activation(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        # shift_x = x - np.max(x) # Numerically stable softmax
        # return np.exp(x) / np.sum(np.exp(shift_x), axis=0)

    def relu_activation(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def tanh_activation(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def calculate_outputs(self, X):
        self.z = np.dot(self.weights, X) + self.biases
        self.A = self.activation_function(self.z)

    def set_outputs(self, A):
        self.A = A

    def get_outputs(self):
        return self.A


class NeuralNetwork:
    def __init__(self, problem_type, hidden_layers: List[HiddenLayer], num_classes: int, loss='mse'):
        self.problem_type = problem_type
        if problem_type == 'regression':
            output_layer = HiddenLayer(hidden_layers[-1].weights.shape[0], 1, activation='linear')
        elif problem_type == 'classification':
            if num_classes == 1:
                output_layer = HiddenLayer(hidden_layers[-1].weights.shape[0], 1, activation='sigmoid')
            else:
                output_layer = HiddenLayer(hidden_layers[-1].weights.shape[0], num_classes, activation='softmax')
        hidden_layers.append(output_layer)
        self.layers = hidden_layers
        self.output = None
        self.X_train = None
        self.y_train = None
        self.y_hat = None

        self.loss_dict = {'mse': self.mse_cost, 'binary ce': self.binary_ce_cost, 'categorical ce': self.categorical_ce_cost}
        self.loss = self.loss_dict[loss]

    def add_layer(self, layer):
        self.layers.insert(-1, layer)

    def save_weights(self, filename='best_weights.txt'):
        file = open(filename, 'w') #File to store the best weights
        for layer in self.layers:
            file.write(layer.layer_type + ' Layer' + '\n')
            file.write(str(layer.weights) + '\n')
            file.write(str(layer.biases) + '\n\n')
        file.close()

    def assign_training_data(self, X_train, y_train, standardize=False, normalize=False):
        if standardize:
            self.X_train = self.standardize(X_train)
        elif normalize:
            self.X_train = self.normalize(X_train)
        else:
            self.X_train = X_train
        self.y_train = y_train

    def normalize(self, X):
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        return (X - x_min) / (x_max - x_min)

    def standardize(self, X):
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        return (X - mean) / std

    def forward(self, X):
        for layer in self.layers[:-1]:
            layer.calculate_outputs(X)
            X = layer.get_outputs()
            if layer.dropout > 0:
                d = np.random.rand(X.shape[0], X.shape[1])
                layer.D = d < layer.dropout
                layer.set_outputs(np.multiply(X, layer.D) / layer.dropout)
                X = layer.get_outputs()

        self.layers[-1].calculate_outputs(X)
        X = self.layers[-1].get_outputs()
        return X

    def get_outputs(self):
        return self.output

    def calculate_accuracy(self, y_hat, y_train):
        if self.problem_type == 'regression':
            return np.sum(np.abs(y_hat - y_train)) / y_train.shape[1]
        elif self.problem_type == 'classification':
            if y_train.shape[0] == 1:
                return np.sum(y_hat.round() == y_train) / y_train.shape[1]
            else:
                return np.sum(np.argmax(y_hat, axis=0) == np.argmax(y_train, axis=0)) / y_train.shape[1]

    def mse_cost(self, y_hat, y_train, alpha):
        m = y_train.shape[1]
        total_loss = np.sum(np.square(y_hat - y_train))
        cost = 1 / (2 * m) * total_loss
        weights_cost = np.sum([np.sum(np.square(layer.weights)) for layer in self.layers])
        return cost + alpha * weights_cost

    def binary_ce_cost(self, y_hat, y_train, alpha):
        m = y_train.shape[1]
        total_loss = np.sum(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))
        cost = -1/m * total_loss
        weights_cost = np.sum([np.sum(np.square(layer.weights)) for layer in self.layers])
        return cost + alpha * weights_cost

    def categorical_ce_cost(self, y_hat, y_train, alpha):
        m = y_train.shape[1]
        total_loss = np.sum(y_train * np.log(y_hat))
        cost = -1/m * total_loss
        weights_cost = np.sum([np.sum(np.square(layer.weights)) for layer in self.layers])
        return cost + alpha * weights_cost

    def calculate_cost(self, y_hat, y_train, alpha=0):
        return self.loss(y_hat, y_train, alpha)

    def backpropagation(self, X, y, y_hat):
        m = X.shape[1]
        j = len(self.layers)

        self.layers[j-1].dZ = y_hat - y
        self.layers[j-1].dW = 1/m * np.dot(self.layers[j-1].dZ, self.layers[j-2].A.T)
        self.layers[j-1].dB = 1/m * np.sum(self.layers[j-1].dZ, axis=1, keepdims=True)
        self.layers[j-2].dA = np.dot(self.layers[j-1].weights.T, self.layers[j-1].dZ)
        if self.layers[j-2].dropout > 0:
            self.layers[j-2].dA = np.multiply(self.layers[j-2].dA, self.layers[j-2].D)
            self.layers[j-2].dA = self.layers[j-2].dA / self.layers[j-2].dropout

        for i, layer in enumerate(reversed(self.layers[1:-1])):
            layer.dZ = layer.dA * layer.activation_function_derivative(layer.z)
            layer.dW = 1/m * np.dot(layer.dZ, self.layers[j-i-3].A.T)
            layer.dB = 1/m * np.sum(layer.dZ, axis=1, keepdims=True)
            self.layers[j-i-3].dA = np.dot(layer.weights.T, layer.dZ)
            if self.layers[j-i-3].dropout > 0:
                self.layers[j-i-3].dA = np.multiply(self.layers[j-i-3].dA, self.layers[j-i-3].D)
                self.layers[j-i-3].dA = self.layers[j-i-3].dA / self.layers[j-i-3].dropout

        self.layers[0].dZ = self.layers[0].dA * self.layers[0].activation_function_derivative(self.layers[0].z)
        self.layers[0].dW = 1/m * np.dot(self.layers[0].dZ, X.T)
        self.layers[0].dB = 1/m * np.sum(self.layers[0].dZ, axis=1, keepdims=True)

    def update_weights(self, learning_rate, beta_1, beta_2, adam, t):
        epsilon = 10 ** -8
        for layer in self.layers:
            layer.mt_w = beta_1 * layer.mt_w + (1 - beta_1) * layer.dW
            layer.mt_b = beta_1 * layer.mt_b + (1 - beta_1) * layer.dB
            layer.vt_w = beta_2 * layer.vt_w + (1 - beta_2) * np.square(layer.dW)
            layer.vt_b = beta_2 * layer.vt_b + (1 - beta_2) * np.square(layer.dB)
            if adam:
                vt_w_corrected, vt_b_corrected = layer.vt_w/(1-beta_2 ** t), layer.vt_b/(1-beta_2 ** t)
                mt_w_corrected, mt_b_corrected = layer.mt_w/(1-beta_1 ** t), layer.mt_b/(1-beta_1 ** t)
                layer.weights -= learning_rate * mt_w_corrected / (np.sqrt(vt_w_corrected) + epsilon)
                layer.biases -= learning_rate * mt_b_corrected / (np.sqrt(vt_b_corrected) + epsilon)
            elif beta_2:
                layer.weights -= learning_rate * layer.dW / (np.sqrt(layer.vt_w) + epsilon)
                layer.biases -= learning_rate * layer.dB / (np.sqrt(layer.vt_b) + epsilon)
            elif beta_1:
                layer.weights -= learning_rate * layer.mt_w
                layer.biases -= learning_rate * layer.mt_b
            else:
                layer.weights -= learning_rate * layer.dW
                layer.biases -= learning_rate * layer.dB

    def train_model(self, training_type, num_iterations, learning_rate=0.01, batch_size=100, momentum=0, ada_grad=0, adam=False, verbose=True):
        if verbose > 0:
            print("Training using", training_type)
            print("Number of iterations: ", num_iterations)
            print("Learning rate: ", learning_rate)
            print("Momentum: ", momentum)
            print("AdaGrad: ", ada_grad)
            print("Adam: ", adam)
        time.sleep(2)
        if training_type == 'gradient descent':
            self.gradient_descent(learning_rate, num_iterations, momentum, ada_grad, adam, verbose)
        elif training_type == 'sgd':
            self.sgd(learning_rate, num_iterations, momentum, ada_grad, adam, verbose)
        elif training_type == 'mini batch':
            self.mini_batch_gd(learning_rate, num_iterations, batch_size, momentum, ada_grad, adam, verbose)
        print("------------------Training complete------------------")
        y_hat = self.forward(self.X_train)
        final_cost = self.calculate_cost(y_hat, self.y_train)
        print("Final cost: ", round(final_cost, 5))
        acc = self.calculate_accuracy(y_hat, self.y_train)
        print(f"Training accuracy: {round(acc*100, 2)}%")
        print("----------------------------------------------------")

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

    def gradient_descent(self, learning_rate, num_iterations, momentum, ada_grad, adam, verbose):
        min_cost = math.inf
        min_iteration = 0
        for i in range(num_iterations):
                print(f'{verbose*f"Iteration: {i+1}"}', end=verbose*'\n')
                y_hat = self.forward(self.X_train)
                cost_ = self.calculate_cost(y_hat, self.y_train)
                if cost_ < min_cost:
                    min_cost = cost_
                    min_iteration = i+1
                    for layer in self.layers:
                        layer.best_weights = layer.weights.copy()
                        layer.best_biases = layer.biases.copy()
                self.backpropagation(self.X_train, self.y_train, y_hat)
                
                self.update_weights(learning_rate, momentum, ada_grad, adam, i+1)
                print(f'{verbose*f"Cost: {round(cost_, 5)}"}', end=verbose*'\n')

        print(f'{verbose*f"Minimum cost at iteration {min_iteration}"}', end=verbose*'\n')
        for layer in self.layers:
            layer.weights = layer.best_weights
            layer.biases = layer.best_biases

    def mini_batch_gd(self, learning_rate, num_iterations, batch_size, momentum, ada_grad, adam, verbose):
        min_cost = math.inf
        min_iteration = 0
        for i in range(num_iterations):
            print(f'{verbose*f"Iteration: {i+1}"}', end=verbose*'\n')
            cost_ = 0
            mini_batches = self.make_mini_batches(self.X_train, self.y_train, batch_size)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                y_hat = self.forward(X_mini)
                self.backpropagation(X_mini, y_mini, y_hat)
                self.update_weights(learning_rate, momentum, ada_grad, adam, i+1)
                cost_ += self.calculate_cost(y_hat, y_mini)

            if (cost_ / batch_size) < min_cost:
                min_cost = cost_ / batch_size
                min_iteration = i+1
                for layer in self.layers:
                    layer.best_weights = layer.weights.copy()
                    layer.best_biases = layer.biases.copy()
            print(f'{verbose*f"Cost: {round(cost_/batch_size, 5)}"}', end=verbose*'\n')

        print(f'Minimum cost at iteration {min_iteration}')
        for layer in self.layers:
            layer.weights = layer.best_weights
            layer.biases = layer.best_biases


    def sgd(self, learning_rate, num_iterations, momentum, ada_grad, adam, verbose):
        m = self.X_train.shape[1]
        min_cost = math.inf
        min_iteration = 0

        for i in range(num_iterations):
            print(f'{verbose*f"Iteration: {i+1}"}', end=verbose*'\n')
            cost_ = 0
            for j in range(m):
                X_mini = self.X_train[:, j].reshape(-1, 1)
                y_mini = self.y_train[:, j].reshape(-1, 1)
                y_hat = self.forward(X_mini)
                self.backpropagation(X_mini, y_mini, y_hat)
                self.update_weights(learning_rate, momentum, ada_grad, adam, i+1)
                cost_ += self.calculate_cost(y_hat, y_mini)

            if (cost_ / m) < min_cost:
                min_cost = cost_ / m
                min_iteration = i+1
                for layer in self.layers:
                    layer.best_weights = layer.weights.copy()
                    layer.best_biases = layer.biases.copy()
            print(f'{verbose*f"Cost: {round(cost_/m, 5)}"}', end=verbose*'\n')

        print(f'Minimum cost at iteration {min_iteration}')
        for layer in self.layers:
            layer.weights = layer.best_weights
            layer.biases = layer.best_biases

def make_one_hot(y):
    one_hot = np.zeros((y.size, y.max()+1))
    one_hot[np.arange(y.size), y] = 1
    return one_hot.T
