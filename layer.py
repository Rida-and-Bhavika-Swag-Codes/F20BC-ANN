import numpy as np

class Layer:

    def __init__(self, input, output_connections):
        self.input = input #input vector
        self.output = None #output vector
        self.num_nodes = None
        self.activation = None

        #number of rows = number of nodes in this layer
        #number of colums = number of nodes in the next layer/number of output connections
        self.weights = np.random.rand(self.num_nodes, output_connections) - 0.5
        self.bias = np.random.rand(output_connections, 1) - 0.5


    def update_weights(self, nodes, output):
        pass

