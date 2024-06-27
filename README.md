Implementation of a Neural Network framework in Python

Optimization techniques-
- SGD (Stochastic gradient descent)
- Mini batch gradient descent
- Momentum
- RMSProp
- Adam

Loss functions:
- 'mse': mean squared error
- 'binary ce': binary cross entropy loss
- 'categorical ce': categorical cross entropy loss

Initialize neural network by providing list of Hidden Layer objects for all the hidden layers, the number of classes, and the loss function. The neural network automatically adds a softmax or sigmoid output layer depending on number of classes.

Example:
```nn = NeuralNetwork([ReLULayer(100, 784), SigmoidLayer(10, 100)], num_classes=10, loss='categorical ce')```

Assign training data using the following function-

```nn.assign_training_data(X_train, y_train)```

Train model on training data-
Training parameters-
- training_type: specifies which training algorithm to use; 'gradient descent', 'mini batch', or 'sgd'
- num_iterations: specifies number of iterations to run training algorithm for
- learning_rate: specifies learning rate (default is 0.01)
- batch_size: specifies batch size for mini batch gradient descent (default is 100)
- momentum: specifies decay parameter for momentum optimization (default is 0)
- ada_grad: specifies decay parameter for RMSProp optimizer (default is 0)
- adam: specifies whether to use Adam optimizer or not (default is False)
- verbose: specifies verbosity (default is True)

Example:

```nn.train_mode('gradient descent', num_iterations=100, learning_rate=0.01, batch_size=32, momentum=0.9, ada_grad=0.999, adam=True, verbose=False)```

Making predictions: Use the forward method in NeuralNetwork class to make predictions on test data. The forward method returns the predictediction for each test example.

Example:
```y_pred = nn.forward(X_test)```

**Update: Added batch normalization and dropout options for training
