from tkinter import Y
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
        self.training_epochs = 2
        
        self.loss_function = None
        self.loss_prime = None
        # 1 = mse, 2 = bce, 3 = hinge
        match loss:
            case 1 : self.loss_function, self.loss_prime = mse, dmse
            case 2 : self.loss_function, self.loss_prime = bce, dbce
            case 3 : self.loss_function, self.loss_prime = hingeloss, dhingeloss
            case other: print("no loss function selected")
         
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
            # gradient descent
            print("\nmean loss:", self.sgd(self.input, self.output, 2))

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
        l = layer.Layer(len(self.input[0]), 7, activations[0])
        l.input = self.input
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
    def sgd(self, X, Y, nclasses): 
        print("in gd")
        total_loss = 0
        pred = self.layers[-1].input.T
        for p, y in zip(pred, Y):
            total_loss += self.loss_function(p, y)
            error = self.loss_prime(p, y)

            for l in reversed(self.layers[:-1]):
                print("back prop")
                layer.Layer.propogate_backward(l, error, self.learning_rate)

        return total_loss / len(X)

            

""" using one_hot_encode as this is binary classification """
def one_hot_encode(y, nclasses):
    y_onehot = np.zeros((y.shape[0], nclasses))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


# loss functions
"""
Parameters: 
pred - ** predicted ** output of the ANN
y - true value
"""
def mse(pred, y):
    return np.mean(np.power(y - pred, 2))

def dmse(pred, y):
    return 2*(pred-y)/y.shape[0]


# adding epsilon to the predicted output to avoid log(0) error
EPSILON = 1e-7  

def bce(pred, y):
    return -np.mean(np.multiply(y, np.log(pred + EPSILON)) + np.multiply(1 - y , np.log(1 - pred + EPSILON)))

def dbce(pred, y):
    return np.sum(-(y / (pred + EPSILON)), (1 - y)/(1 - pred + EPSILON))

def hingeloss(pred, y):
    # from https://stats.stackexchange.com/questions/539496/how-to-create-hinge-loss-function-in-python-from-scratch
    npred = np.array([-1 if i == 0 else i for i in pred])
    ny = np.array([-1 if i == 0 else i for i in y])

    return np.mean([max(0, 1 - actual * predicted) for actual, predicted in zip(ny, npred)])

def dhingeloss(pred, y):
    pass