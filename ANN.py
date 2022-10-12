import numpy as np
import layer

class ANN:

    #currently building a network with 1 hidden layer of 7 nodes and output layer with 2 nodes. Learning rate of 0.1. Loss function - MSE. 
    # Activation function tanh in all layers. 200 training epochs. Stochastic gradient descent. No test-train split. 

    """initialise network with hyperparameters"""
    def __init__(self, learn_rate, activation, loss):
        self.input = None 
        self.layers = [layer.Layer(30, 7, activation, 0)] #assume 7 nodes in the next hidden layer

        #switch case for loss
        self.loss_function = loss
        self.num_hidden_layers = None
        self.learning_rate = learn_rate
        self.training_epochs = 50
        #self.GD_type = None
        #self.dropout_rate = None

    """train ANN"""
    def train(self):
        layer.propogate_forward()
        layer.propogate_backward()
        layer.update_weights

    """test ANN"""
    def test(self):
        #cross validation, train-test split or both
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
    Layer.update_param()












