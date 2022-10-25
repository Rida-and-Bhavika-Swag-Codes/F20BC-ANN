import layer
import numpy as np
class ANN:

    """
    Initialise network with hyperparameters

    Parameters: 

    """
    def __init__(self, learn_rate, loss, input, output):
        self.input = input #input vector 
        self.output = output #Target class
        self.layers = [] #assume 7 nodes in the next hidden layer
        
        self.learning_rate = learn_rate
        self.training_epochs = 100
        
        self.loss_function = None
        self.loss_prime = None
        # 1 = bce, 2 = mse, 3 = mae
        # match loss:
        #     case 1 : self.loss_function, self.loss_prime = bce, dbce
        #     case 2 : self.loss_function, self.loss_prime = mse, dmse
        #     case 3 : self.loss_function, self.loss_prime = mae, dmae
        #     case other: print("no loss function selected")

    def setLayers(self, activations, *nodes_per_layer):
        print("the activations are", activations)
        print("the variable argument is ", nodes_per_layer)
        print("number of hidden layers", len(nodes_per_layer)-1, "\n")
        print("input to the network is", self.input)
        print("the output is", self.output)

        #append input layer
        #l = layer.Layer(len(self.input), nodes_per_layer[0], activations[0])
        l = layer.Layer(30, 10, activations[0])
        l.input = self.input
        self.layers.append(l)
        print("adding input layer")

        #append hidden layers
        for i in range (len(nodes_per_layer)-1):
            #self.layers.append(layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1]))
            self.layers.append(layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1]))
            print("added 1 hidden layer")

        #append output layer
        #self.layers.append(layer.Layer(nodes_per_layer[-1], 0, 0 ))#output layer has no output connections or activation function
        self.layers.append(layer.Layer(nodes_per_layer[-1], 0, 0 ))
        print("added 1 output layer")

    def propogate_forward(self):

        print("the network layers are", self.layers)
        for currlayer, nextlayer in zip(self.layers[:-1], self.layers[1:]):
            print("weights", currlayer.weights)
            #calculate weighted sum
            currlayer.wsum = currlayer.weights.dot(currlayer.input) + currlayer.bias
            #apply activation
            currlayer.output = currlayer.activation(currlayer.wsum)
            #set next layers input as the output frm this layer
            nextlayer.input = currlayer.output


    def propogate_backward(self):
        print("CHECKING")
        wgrad, bgrad = [], []
        one_hot_Y = one_hot(self.output)
        layer = self.layers[-2]
        error = layer.output - one_hot_Y
        m = self.output.size
        
        for i in range(-2, -(len(self.layers)), -1):
           smn = 1/ m * error.dot(self.layers[i].input.T)
           wgrad.append(smn)
           print("the layer is", layer, layer.weights.shape)
           print("the w of the last layer", smn.shape)
           smn = 1/m *np.sum(error)
           bgrad.append(smn)
           print("the b of the last layer", smn.shape)
           error = layer.weights.T.dot(error) * self.layers[i-1].activation_prime(self.layers[i-1].wsum)

        #return wgrad, bgrad
        self.parameter_update(wgrad, bgrad)

    def parameter_update(self, wgrad, bgrad):
        for layer, wupdate, bupdate in zip(reversed(self.layers[:-1]), wgrad, bgrad):
            print("dimensions", layer, layer.weights.shape, wupdate.shape, bupdate.shape)
            layer.weights -= (self.learning_rate * wupdate)
            layer.bias -= (self.learning_rate * bupdate)
            print("new weights", layer.weights.shape)
            print("new bias", layer.bias.shape)
 


    def train_sgd(self):
        for i in range(200):
            self.propogate_forward()
            self.propogate_backward()
            
            if i%10 == 0:
                print("Iteration: ", i)
                predictions = get_predictions(self.layers[-1].input)
                print("Accuracy ", get_accuracy(predictions, self.output))

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

    

    
         