import numpy as np
import layer

class ANN:

    #currently building a network with 1 hidden layer of 7 nodes and output layer with 2 nodes. Learning rate of 0.1. Loss function - MSE. 
    # Activation function tanh in all layers. 200 training epochs. Stochastic gradient descent. No test-train split. 

    """
    Initialise network with hyperparameters

    Parameters: 

    """
    def __init__(self, learn_rate, loss, input, output):
        self.input = input  #input vector 
        self.layers = [] #assume 7 nodes in the next hidden layer
        self.output = output #Target class

        self.learning_rate = learn_rate
        self.training_epochs = 2

        #switch case for loss HERE?
        #self.loss_function = losss
        #self.GD_type = None
        #self.dropout_rate = None

    """train ANN"""
    def train(self):
        for j in range(self.training_epochs):
            layer.Layer.propogate_forward(self.layers[0],self.layers[0].input)
            for i in range(len(self.layers)-2):
                layer.Layer.propogate_forward(self.layers[i+1], self.layers[i].output)

            #propogate_backward()
            #layer.update_weights

    """test ANN"""
    def test(self):
        #cross validation, train-test split or both
        pass

    """
    Append hidden layers

    Parameters:
    *nodes_per_layer - nodes per hidden layer 
    activations - list of activations (in the order of the hidden layer)
    
    """
    def setLayers(self, activations, *nodes_per_layer):
        print("the activations are", activations)
        print("the variable argument is ", nodes_per_layer)
        print("number of hidden layers", len(nodes_per_layer)-1, "\n")

        #append input layer
        l = layer.Layer(len(self.input[0]), 7, 0)
        l.input = self.input
        self.layers.append(l)
        print("adding input layer")

        #append hidden layers
        for i in range (len(nodes_per_layer)-1):
            self.layers.append(layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i])) #replace propogate forward with the output instead
            l = self.layers[i+1]
            print("added 1 hidden layer")

        #append output layer
        self.layers.append(layer.Layer(nodes_per_layer[-1], 0, 0)) #output layer has no output connections or activation function
        print("added 1 output layer")
    

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













