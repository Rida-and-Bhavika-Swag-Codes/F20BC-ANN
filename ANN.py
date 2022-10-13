import numpy as np
import layer

class ANN:

    #currently building a network with 1 hidden layer of 7 nodes and output layer with 2 nodes. Learning rate of 0.1. Loss function - MSE. 
    # Activation function tanh in all layers. 200 training epochs. Stochastic gradient descent. No test-train split. 

    """
    Initialise network with hyperparameters

    Parameters: 

    """
    def __init__(self, learn_rate, activation, loss, input, output):
        self.input = input  #input vector 
        self.layers = [layer.Layer(len(input), self.input, 7, activation, 0)] #assume 7 nodes in the next hidden layer
        self.learning_rate = learn_rate
        self.training_epochs = 50

        #switch case for loss HERE?
        #self.loss_function = loss
        #self.GD_type = None
        #self.dropout_rate = None

    """train ANN"""
    def train(self):
        propogate_forward()
        #propogate_backward()
        #layer.update_weights

    """test ANN"""
    def test(self):
        #cross validation, train-test split or both
        pass

    """
    Append hidden layers

    *nodes_per_layer - nodes per hidden layer 
    activations - list of activations (in the order of the hidden layer)
    
    """
    def setHidden(self, *nodes_per_layer, activations):
        #define number of outgoingn connections from this layer
        
        if len(*nodes_per_layer) < 2: #if only 1 or no hidden layer
            outnodes = self.output
        else:
            outnodes = nodes_per_layer[1:]

        for i in range (len(*nodes_per_layer)):
            self.layers.append[layer.Layer(nodes_per_layer[i], self.input, outnodes[i], activations[i], 0)]
        pass


#loss functions
"""
Mean squared error loss function
Parameters: 
output - output of the ANN
y - true value
"""
def mse(output, y):
    pass

def b_cross_entropy():
    pass

def hinge_loss():
    pass


#
def propogate_forward():
    #1. calculate weighted sum
    #calculate Activations of Hidden Layer
    
    #calculate Activations of output layer
    pass
def propogate_backward():
    # compute loss - find derivative of the weights and biases at each layer
    # update params
    pass











