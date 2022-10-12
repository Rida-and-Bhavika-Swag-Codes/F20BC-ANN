import numpy as np
from layer import Layer

class ANN:

    #currently building a network with 1 hidden layer of 7 nodes and output layer with 2 nodes. Learning rate of 0.2. Loss function - MSE. 
    # Activation function tanh in all layers. 

    """initialise network with hyperparameters"""
    def __init__(self, learn_rate):
        self.input = Layer((30, 7 )) #assume 7 nodes in the next hidden neuron
        self.layers = [] #add input layer by default?
        self.loss_function, self.loss_derivative = mse()
        self.num_hidden_layers = None
        self.learning_rate = learn_rate
        #self.training_epochs = None
        #self.GD_type = None
        #self.dropout_rate = None

    """train ANN"""
    def train(self):
        propogate_forward()
        propogate_backward()

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

#activation functions:
def tanh(weighted_sum):
    return np.tanh(weighted_sum), 1-np.tanh(weighted_sum)**2

#
def propogate_forward():
    pass
def propogate_backward():
    pass












