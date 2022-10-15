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
        self.weights = np.random.rand(output_connections, self.num_nodes) - 0.5
        self.bias = np.random.rand(output_connections, 1) - 0.5

    def propogate_forward(self, input):
        #set input vector
        self.input = input

        # calculate weighted sum
        wsum = np.dot(self.weights, input) + self.bias
        # apply activation
        self.output = relu(wsum)

    """Use with backpropogation"""
    def update_param(self, learning_rate, gradients):
        self.weights = self.weights - learning_rate * gradients

    """debug function"""
    def get_properties(self):
        print("Number of nodes:", self.num_nodes)
        print("Activation Function:", self.activation)
        print("Weights:", self.weights)
        print("Bias:", self.bias)
        print("Input vector:", self.input)
        print("Output vector:", self.output)


#activation functions:
def tanh(weighted_sum):
    return np.tanh(weighted_sum), 1-np.tanh(weighted_sum)**2

def relu(wsum):
    return np.maximum(wsum, 0)

