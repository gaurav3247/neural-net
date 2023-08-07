Implementation of a Neural Network in Python

Optimization techniques already implemented-
- SGD (Stochastic gradient descent)
- Mini batch gradient descent
- Momentum
- RMSProp
- Adam

Optimization techniques to be implemented-
- NAG

Initialize neural network by providing list of Hidden Layer objects for all the hidden layers and number of clases.

Example:
```nn = NeuralNetwork([HiddenLayer(100, 784, 'relu'), HiddenLayer(10, 100, 'softmax')], num_classes=10)```

Assign training data using the following function-

```nn.assign_training_data(X_train, y_train)```

Train model on training data-
Training parameters-
- training_type: specifies which training algorithm to use; 'gradient descent', 'mini batch', or 'sgd'
- num_iterations: specifies number of iterations to run training algorithm for
- learning_rate: specifies learning rate (default is 0.01)
- momentum: specifies decay parameter for momentum optimization (default is 0)
- ada_grad: specifies decay parameter for RMSProp optimizer (default is 0)
- adam: specifies whether to use Adam optimizer or not (default is False)
- verbose: specifies verbosity (default is True)

Example:

```nn.train_mode('gradient descent', num_iterations=100, learning_rate=0.01, momentum=0.9, ada_grad=0.999, adam=True, verbose=False)```
