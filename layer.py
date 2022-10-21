import numpy as np

class Layer:

    def __init__(self, nodes, output_connections, activation):
        self.input = None #input vector
        self.output = None #output vector
        self.num_nodes = nodes

        #switch case for activations here?
        self.activation, self.activation_prime = self.setActivation(activation)

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
            return self.output

    def propogate_backward(self, loss, learning_rate):
        print("in back prop")
        
        ierror = loss * self.weights.T # input error
        wgrad = np.dot(self.input.T, loss)
        bgrad = loss * 1

        # updating the parameters
        self.weights -= learning_rate * wgrad
        self.bias -= learning_rate * bgrad
        return self.activation_prime(ierror) * loss

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
            case 1 : return tanh, dtanh
            case 2 : return relu, drelu
            case 3 : return sigmoid, dsigmoid
            case 0 : return None, None
        #add error handling

# activation functions:
def tanh(wsum):
    return np.tanh(wsum)

def relu(wsum):
    return np.maximum(wsum, 0)

def sigmoid(wsum): #aka. logistic activation
    return 1.0 / (1 + np.exp(-wsum))

# derivatives of activation functions
def dtanh(wsum):
    th = tanh(wsum)
    return wsum * (1 - np.square(th))

def drelu(wsum):
    rel = relu(wsum)
    return np.int64(rel > 0)

def dsigmoid(wsum):
    sigm = sigmoid(wsum)
    return sigm * (1 - sigm)

