import math
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = open('best_weights.txt', 'w') #File to store the best weights
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
        self.vt_w_corrected, self.vt_b_corrected = 0, 0
        self.mt_w_corrected, self.mt_b_corrected = 0, 0
        self.best_weights = self.weights
        self.best_biases = self.biases

    def change_weights(self, new_weights, new_biases):
        print("Changing weights")
        self.best_weights = new_weights
        self.best_biases = new_biases
        print(self.best_weights, self.best_biases)

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
        # shift_x = x - np.max(x) # Numerically stable softmax
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

        return X

    def get_outputs(self):
        return self.output

    def calculate_accuracy(self, y_hat, y_train):
        return np.sum(y_hat.argmax(axis=0) == y_train.argmax(axis=0)) / y_train.shape[1]

    def calculate_cost(self, y_hat, y_train):
        m = y_train.shape[1]
        total_loss = np.sum(y_train * np.log(y_hat))
        cost = -1/m * total_loss
        return cost

    def backpropagation(self, X, y, y_hat):
        m = X.shape[1]
        j = len(self.hidden_layers)

        self.hidden_layers[j-1].dZ = y_hat - y
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

    def update_weights(self, learning_rate, beta_1, beta_2, adam, t):
        epsilon = 10 ** -8
        for layer in self.hidden_layers:
            layer.mt_w = beta_1 * layer.mt_w + (1 - beta_1) * layer.dW
            layer.mt_b = beta_1 * layer.mt_b + (1 - beta_1) * layer.dB
            layer.vt_w = beta_2 * layer.vt_w + (1 - beta_2) * np.square(layer.dW)
            layer.vt_b = beta_2 * layer.vt_b + (1 - beta_2) * np.square(layer.dB)

            layer.vt_w_corrected, layer.vt_b_corrected = layer.vt_w/(1-beta_2 ** t), layer.vt_b/(1-beta_2 ** t)
            layer.mt_w_corrected, layer.mt_b_corrected = layer.mt_w/(1-beta_1 ** t), layer.mt_b/(1-beta_1 ** t)

            layer.weights -= learning_rate * layer.mt_w / (np.sqrt(layer.vt_w) + epsilon)
            layer.biases -= learning_rate * layer.mt_b / (np.sqrt(layer.vt_b) + epsilon)

    def train_model(self, training_type, num_iterations, learning_rate=0.01, momentum=0, ada_grad=0, adam=False, verbose=True):
        print("Training using", training_type)
        print("Learning rate: ", learning_rate)
        print("Number of iterations: ", num_iterations)
        print("Momentum: ", momentum)
        time.sleep(2)
        if training_type == 'gradient descent':
            self.gradient_descent(learning_rate, num_iterations, momentum, ada_grad, adam, verbose)
        elif training_type == 'sgd':
            self.sgd(learning_rate, num_iterations, momentum, ada_grad, adam, verbose)
        elif training_type == 'mini batch':
            batch_size = int(input("Enter batch size: "))
            self.mini_batch_gd(learning_rate, num_iterations, batch_size, momentum, ada_grad, adam, verbose)
        print("------------------Training complete------------------")
        y_hat = self.forward(self.X_train)
        final_cost = self.calculate_cost(y_hat, self.y_train)
        print("Minimum cost: ", final_cost)
        acc = self.calculate_accuracy(y_hat, self.y_train)
        print(f"Training accuracy: {round(acc*100, 2)}%")

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
                    for layer in self.hidden_layers:
                        layer.best_weights = layer.weights.copy()
                        layer.best_biases = layer.biases.copy()
                self.backpropagation(self.X_train, self.y_train, y_hat)
                
                self.update_weights(learning_rate, momentum, ada_grad, adam, i+1)
                print(f'{verbose*f"Cost: {cost_}"}', end=verbose*'\n')

        # print("Minimum cost: ", min_cost)
        # print("Minimum cost at iteration: ", min_iteration)
        for layer in self.hidden_layers:
            layer.weights = layer.best_weights
            layer.biases = layer.best_biases

    def mini_batch_gd(self, learning_rate, num_iterations, batch_size, momentum, ada_grad, adam, verbose):
        min_cost = math.inf
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
            print(cost_)
            if (cost_ / batch_size) < min_cost:
                min_cost = cost_ / batch_size
                for layer in self.hidden_layers:
                    layer.best_weights = layer.weights.copy()
                    layer.best_biases = layer.biases.copy()
            print(f'{verbose*f"Cost: {cost_/batch_size}"}', end=verbose*'\n')

        for layer in self.hidden_layers:
            layer.weights = layer.best_weights
            layer.biases = layer.best_biases


    def sgd(self, learning_rate, num_iterations, momentum, ada_grad, adam, verbose):
        m = self.X_train.shape[1]

        for i in range(num_iterations):
            print("Iteration: ", i+1)
            cost_ = 0
            for j in range(m):
                X_mini = self.X_train[:, j].reshape(-1, 1)
                y_mini = self.y_train[:, j].reshape(-1, 1)
                y_hat = self.forward(X_mini)
                self.backpropagation(X_mini, y_mini)
                self.update_weights(learning_rate, momentum, ada_grad, adam, i+1)
                cost_ += self.calculate_cost(y_hat, y_mini)
            print("Cost :", cost_ / m)


def make_one_hot(y):
    one_hot = np.zeros((y.size, y.max()+1))
    one_hot[np.arange(y.size), y] = 1
    return one_hot.T
