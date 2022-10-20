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

        self.weights = []
        self.biases = []

        self.learning_rate = learn_rate
        self.training_epochs = 1
        
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
            current_layer = layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1])
            self.layers.append(current_layer) #replace propogate forward with the output instead
            #l = self.layers[i+1]
            self.layers.append(current_layer)
            self.weights.append(current_layer.weights)
            self.biases.append(current_layer.bias)
            print("added 1 hidden layer")

        #append output layer
        output_layer = layer.Layer(nodes_per_layer[-1], 0, 0)
        self.layers.append(output_layer) #output layer has no output connections or activation function
        self.layers.append(output_layer)
        self.weights.append(output_layer.weights)
        self.biases.append(output_layer.bias)
        print("added 1 output layer")
    


    """ gradient descent functions """
    # stochastic gradient descent 
    def sgd(self, X, Y, nclasses): # w minimises the function and needs to be estimated
        total_loss = 0
        for x, y in zip(X, Y):
            x = x.reshape(1, -1)
            y = y.reshape(1)

            y_onehot = one_hot_encode(y, nclasses) # true y
            pred = layer.Layer.propogate_forward(x)

            total_loss += self.loss_function(pred, y_onehot)
            error = self.loss_prime(pred, y_onehot)
            wgrad, bgrad = layer.Layer.propogate_backward(error, self.learning_rate)
            # incomplete
            

""" using one_hot_encode as this is binary classification """
def one_hot_encode(y, nclasses):
    y_onehot = np.zeros((y.shape[0], nclasses))
    id = [np.arange(y.shape[0]), y]
    y_onehot[id] = 1
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

def bce(pred, y):
    # adding epsilon to the predicted output to avoid log(0) error
    epsilon = 1e-7    
    return -np.mean(np.multiply(y, np.log(pred + epsilon)) + np.multiply(1 - y , np.log(1 - pred + epsilon)))

def dbce(pred, y):
    pass

def hingeloss(pred, y):
    npred = np.array([-1 if i == 0 else i for i in pred])
    ny = np.array([-1 if i == 0 else i for i in y])

    return np.mean([max(0, 1 - actual * predicted) for actual, predicted in zip(ny, npred)])

def dhingeloss(pred, y):
    pass