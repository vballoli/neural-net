class NeuralNetwork:
    """
    Implementation of an Artificial Neural Network with back propogation.
    """

    def __init__(self, learning_rate, layers):
        """ Initialize Neural Network parameters
    
        Args:
            learning_rate: Learning rate of the neural network
        """
        self.learning_rate = learning_rate
        self.layers = layers

    def feed_forward(self, X):
        """ Feed forward of the neural network across the layers

        Args:
            X: Input training data
            y: Input target data
        """
        self.X = X
        layers = self.layers
        for i in range(len(layers)):
            current_layer = layers[i]
            if i != 0:
                previous_layer = layers[i-1]
                current_layer.feed_forward(previous_layer.y)
            else:
                current_layer.feed_forward(X)
        self.layers = layers

    def back_propogation(self, y):
        """ Back propgation of the error across the layers 

        """
        self.y = y
        layers = self.layers
        for i in reversed(range(len(layers))):
            current_layer = layers[i]
            if i != len(layers)-1:
                current_layer.back_propogate(layers[i+1], None)
            else:
                current_layer.compute_error(y)
                current_layer.back_propogate(None, y)
        self.layers = layers

    def update_weights(self):
        """ Update all the weights in the layers

        """
        layers = self.layers
        for layer in layers:
            layer.weights += self.learning_rate * layer.d_weights
        self.layers = layers
            
