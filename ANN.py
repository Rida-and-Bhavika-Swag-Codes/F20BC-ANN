import numpy as np
import layer

class ANN:

    """
    Initialise network with hyperparameters

    Parameters: 

    """
    def __init__(self, learn_rate, loss, input, output):
        self.input = input  #input vector 
        self.layers = [] #assume 7 nodes in the next hidden layer
        self.output = output #Target class

        self.learning_rate = learn_rate
        self.training_epochs = 1

        #switch case for loss HERE?
        #self.loss_function = loss
        #self.GD_type = None
        #self.dropout_rate = None

    """train ANN"""
    def train(self):
        for j in range(self.training_epochs):
            print("propogating forward")
            layer.Layer.propogate_forward(self.layers[0],self.layers[0].input)
            for i in range(len(self.layers)-2):
                print("propogating forward")
                layer.Layer.propogate_forward(self.layers[i+1], self.layers[i].output)
            self.layers[-1].input = self.layers[-2].output #set activations of the last layer

            #propogate_backward()
            #layer.update_weights

    """test ANN"""
    def test(self):
        #cross validation, train-test split or both
        pass

    """
    Append hidden layers

    Parameters:
    *nodes_per_layer - nodes per hidden layer and the output layer
    activations - list of activations (in the order of the hidden layer)
    
    """
    def setLayers(self, activations, *nodes_per_layer):
        print("the activations are", activations)
        print("the variable argument is ", nodes_per_layer)
        print("number of hidden layers", len(nodes_per_layer)-1, "\n")

        #append input layer
        l = layer.Layer(len(self.input[0]), 7, activations[0])
        l.input = self.input
        self.layers.append(l)
        print("adding input layer")

        #append hidden layers
        for i in range (len(nodes_per_layer)-1):
            self.layers.append(layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1])) #replace propogate forward with the output instead
            l = self.layers[i+1]
            print("added 1 hidden layer")

        #append output layer
        self.layers.append(layer.Layer(nodes_per_layer[-1], 0, 0)) #output layer has no output connections or activation function
        print("added 1 output layer")
    

#loss functions
"""
Mean squared error loss function

Parameters: 
output - ** predicted ** output of the ANN
y - true value
"""
def mse(output, y):
    return np.mean(np.power(y - output, 2))

def b_cross_entropy(output, y):
    # adding epsilon to the predicted output to avoid log(0) error
    epsilon = 1e-7    
    return -np.mean(np.multiply(y, np.log(output + epsilon)) + np.multiply(1 - y , np.log(1 - output + epsilon)))

def hinge_loss(output, y):
    noutput = np.array([-1 if i == 0 else i for i in output])
    ny = np.array([-1 if i == 0 else i for i in y])

    return np.mean([max(0, 1 - act * pred) for act, pred in zip(ny, noutput)])
    













