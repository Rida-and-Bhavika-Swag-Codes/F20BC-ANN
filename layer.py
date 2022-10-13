import numpy as np

class Layer:

    def __init__(self, nodes, input, output_connections, activation, weighted_sum):
        self.input = input #input vector
        self.output = None #output vector
        self.num_nodes = nodes
        #switch case for activations
        self.activation , self.activation_derivative = tanh(weighted_sum)

        #number of rows in weight matrix = number of nodes in the layer
        #number of colums = number of nodes in the next layer/number of output connections
        self.weights = np.random.rand(self.num_nodes, output_connections) - 0.5
        self.bias = np.random.rand(output_connections, 1) - 0.5


    def update_param(self, learning_rate, gradients):
        self.weights = self.weights - learning_rate * gradients

    def weighted_sum(weights, bias, input):
        weights.dot(X) + bias


#activation functions:
def tanh(weighted_sum):
    return np.tanh(weighted_sum), 1-np.tanh(weighted_sum)**2

