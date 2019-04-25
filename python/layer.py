import numpy as np
from sklearn.metrics import mean_squared_error

class Layer:
    """
    Implementation of a neural network layer
    """

    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"


    def __init__(self, input_dims, output_dims, activation_function):
        """ Initializes the layer's parameters


        """
        self.weights =  np.random.rand(input_dims, output_dims)
        assert activation_function == self.LINEAR or activation_function == self.SIGMOID or activation_function == self.TANH
        self.activation_function = activation_function

    def activation(self, x):
        """ Activation function of each neuron in the layer.

        Args:
            x: input value to the activation function

        Returns: 
            f(x) where f is the function provided in the layer initialization

        """
        if self.activation_function == self.LINEAR:
            return x
        elif self.activation_function == self.SIGMOID:
            return 1.0/(1.0+(np.exp(-x)))
        elif self.activation_function == self.TANH:
            return np.tanh(x)

    def derivative_activation(self, x):
        """ Derivative of the activation function 

        Args:
            x: activated value

        Returns:
            Value of the derivative of the activation function
        """
        if self.activation_function == self.LINEAR:
            return np.ones(x.shape)
        elif self.activation_function == self.SIGMOID:
            return x*(1-x)
        elif self.activation_function == self.TANH:
            return (1-(x**2))

    def feed_forward(self, X):
        """ Feed forward of the preivous layer or input data with layer's weights

        Feed forward forward propogates i.e dot product of the layer's weight with the input data

        Args:
            X: Input data - either training data or output of a layer

        Returns:
            y: Output data of dimensions 1xoutput_dims
        """
        self.X = X
        self.y = self.activation(np.dot(X, self.weights))
        return self.y

    def back_propogate(self, layer, y):
        """ Backpropagtes the error to correct the weights

        Args:
            layer: The next layer in the neural network from which the derivative value is calculated
            y: The target values

        """
        if layer is None:
            y = y.reshape((y.shape[0],1))
            self.d_activation = (y - self.y) * self.derivative_activation(self.y)
        else:
            self.d_activation = np.dot(layer.d_activation, layer.weights.T) * self.derivative_activation(self.y)
        self.d_weights = np.dot(self.X.T, self.d_activation)
    
    def compute_error(self, y):
        """ Calculates the error as per the output
        """
        error = mean_squared_error(self.y, y)**2
        print("Error: ", error)
        return error
        

