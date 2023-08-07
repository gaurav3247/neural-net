Implementation of a Neural Network in Python

Optimization techniques already implemented-
- SGD (Stochastic gradient descent)
- Mini batch gradient descent
- Momentum
- RMSProp

Optimization techniques to be implemented-
- NAG
- ADAM

Initialize neural network by providing list of Hidden Layer objects for all the hidden layers and number of clases in classification problem.

Example:
```nn = NeuralNetwork([HiddenLayer(100, 784, 'relu'), HiddenLayer(10, 100, 'softmax')], 10)```

Assign training data using the following function-
```nn.assign_training_data(X_train, y_train)```