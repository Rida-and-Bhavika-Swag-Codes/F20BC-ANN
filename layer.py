import numpy as np

class Layer:

    def __init__(self, nodes, output_connections, activation):
        self.input = None #input vector
        self.output = None #output vector
        self.num_nodes = nodes

        #switch case for activations here?
        self.activation = activation 

        #number of rows in weight matrix = number of nodes in the layer
        #number of colums = number of nodes in the next layer/number of output connections
        self.weights = np.random.rand(self.num_nodes, output_connections) - 0.5
        self.bias = np.random.rand(output_connections, 1) - 0.5

    def propogate_forward(self):
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def update_param(self, learning_rate, gradients):
        self.weights = self.weights - learning_rate * gradients

    def weighted_sum(self, weights, bias, input):
        self.weights.dot(self.input) + bias

    """debug function"""
    def get_properties(self):
        print("Input vector:", self.input)
        print("Output vector:", self.output)
        print("Number of nodes:", self.num_nodes)
        print("Activation Function:", self.activation)
        print("Weights:", self.weights)
        print("Bias:", self.bias)

#activation functions:
def tanh(weighted_sum):
    return np.tanh(weighted_sum), 1-np.tanh(weighted_sum)**2

