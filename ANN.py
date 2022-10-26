from tkinter import Y
import numpy as np
import layer

class ANN:

    """
    Initialise network with hyperparameters

    Parameters: 

    """
    def __init__(self, input, output, loss = 2, epoch = 2, learn_rate = 0.1):
        self.input = input.to_numpy()  #input vector 
        self.layers = [] #assume 7 nodes in the next hidden layer
        self.output = output #Target class

        self.learning_rate = learn_rate
        self.training_epochs = epoch
        
        self.loss_function = None
        self.loss_prime = None
        # 1 = bce, 2 = mse, 3 = mae
        match loss:
            case 1 : self.loss_function, self.loss_prime = bce, dbce
            case 2 : self.loss_function, self.loss_prime = mse, dmse
            case 3 : self.loss_function, self.loss_prime = mae, dmae
            case other: print("no loss function selected")
         
        #self.GD_type = None
        #self.dropout_rate = None

    """train ANN"""
    def train(self):
        for j in range(self.training_epochs):
            print("propogating forward")
            loss_epoch = 0
            for sample in range (len(self.input[0])):
                layer.Layer.propogate_forward(self.layers[0], self.input[:,sample])
                for i in range(len(self.layers)-2):
                    layer.Layer.propogate_forward(self.layers[i+1], self.layers[i].output)
                self.layers[-1].input = self.layers[-2].output #set activations of the last layer

                print("input:", self.input)
                loss = self.sgd(self.output[sample]) 
                loss_epoch += loss
                print("loss:", loss)
            epoch_error = loss_epoch/len(self.input[0])
            print("epoch error:", epoch_error)
    """test ANN
    input: x values given to the model
    returns the predicted labels for input"""
    def test(self, input):
         # predict output for given input
        predictions = []
        input = input.T
        #forward propogate over all samples of the given input
        for j in range(len(self.input)):
            layer.Layer.propogate_forward(self.layers[0],input)
            for i in range(len(self.layers) - 1):
                layer.Layer.propogate_forward(self.layers[i+1], self.layers[i].output)
            predictions.append(self.layers[-1].input)
        return predictions

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
        l = layer.Layer(len(self.input), 7, activations[0])
        self.layers.append(l)
        print("adding input layer")

        #append hidden layers
        for i in range (len(nodes_per_layer)-1):
            self.layers.append(layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1]))
            print("added 1 hidden layer")

        #append output layer
        self.layers.append(layer.Layer(nodes_per_layer[-1], 0, 0))#output layer has no output connections or activation function
        print("added 1 output layer")
    



    """ gradient descent functions """
    # stochastic gradient descent 
    def sgd(self, y): 
        print("in gd")
        
        # the predicted outcome
        pred = self.layers[-1].input

        loss = self.loss_function(pred, y)
        error = self.loss_prime(pred, y)

        for l in reversed(self.layers[1:-1]):
            error, wgrad, bgrad = l.propogate_backward(error)
            l.update_parameters(wgrad, bgrad, self.learning_rate)

        # backpropogate last layer
        self.layers[0].input.resize(1,30) 
        error, wgrad, bgrad = self.layers[0].propogate_backward(error, False) 
        self.layers[0].update_parameters(wgrad, bgrad, self.learning_rate)

        return loss

""" using one_hot_encode as this is binary classification """
def one_hot_encode(y, nclasses):
    y_onehot = np.zeros((y.size, nclasses))
    y_onehot[np.arange(y.size), y] = 1
    return y_onehot

# loss functions
"""
Parameters: 
pred - ** predicted ** output of the ANN
y - true value
""" 

# adding epsilon to the predicted output to avoid log(0) error and for stability
EPSILON = 1e-7  

def bce(pred, y):
    return -np.mean((y * np.log(pred + EPSILON)) + (1 - y ) * np.log(1 - pred + EPSILON))

def dbce(pred, y):
    return -(y / (pred + EPSILON)) + ((1 - y)/(1 - pred + EPSILON))

def mse(pred, y):
    return np.mean(np.power(y - pred, 2))

def dmse(pred, y):
    return 2*(pred-y)/y.size

def mae(pred, y):
    return np.mean(abs(y - pred))

def dmae(pred, y):
    return -((y - pred) / (abs(y - pred) + EPSILON))/y.size