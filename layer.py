import numpy as np

class Layer:

    def __init__(self, nodes, output_connections, activation):
        self.input = None #input vector
        self.output = None #output vector
        self.num_nodes = nodes

        #switch case for activations here?
        self.activation = self.setActivation(activation)

        #number of rows in weight matrix = number of nodes in the layer
        #number of colums = number of nodes in the next layer/number of output connections
        self.weights = np.random.rand(output_connections, self.num_nodes) - 0.5
        self.bias = np.random.rand(output_connections, 1) - 0.5

    def propogate_forward(self, input):
        #set input vector
        self.input = input

        # calculate weighted sum
        wsum = np.dot(self.weights, input) + self.bias
        # convert type to float32 [reference: https://stackoverflow.com/questions/18557337/numpy-attributeerror-float-object-has-no-attribute-exp]
        wsum = np.array(wsum, dtype = np.float32)
        # apply activation
        if self.activation: #if self.activation null then this is an output layer and we don't forward propogate from here
            self.output = self.activation(wsum)

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
    
    """Take a integer as input and match it with an activation function. Then, return matched function"""
    def setActivation(self, activation):
        match activation:
            case 1 : return tanh
            case 2 : return relu
            case 3 : return sigmoid
            case 0 : None
        #add error handling

#activation functions:
def tanh(wsum):
    return np.tanh(wsum)

def relu(wsum):
    return np.maximum(wsum, 0)

def sigmoid(wsum): #aka. logistic activation
    return 1.0 / (1 + np.exp(-wsum))

