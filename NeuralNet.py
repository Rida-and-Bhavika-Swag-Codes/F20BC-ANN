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
        self.decay = self.learning_rate/self.training_epochs # for learning rate scheduler 

        match loss:
            case 1 : self.loss_function = bce
            case 2 : self.loss_function = mse
            #case 3 : self.loss_function = mae            
            case other: self.loss_function = None
 

    def setLayers(self, activations, *nodes_per_layer):
        print("the activations are", activations)
        print("the variable argument is ", nodes_per_layer)
        print("number of hidden layers", len(nodes_per_layer)-1, "\n")
        print("input to the network is", self.input)
        print("the output is", self.output)

        #append input layer
        #l = layer.Layer(len(self.input), nodes_per_layer[0], activations[0])
        l = layer.Layer(30, nodes_per_layer[0], activations[0])
        l.input = self.input
        self.layers.append(l)
        print("adding input layer")

        #append hidden layers
        for i in range (len(nodes_per_layer)-1):
            self.layers.append(layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1]))
            print("added 1 hidden layer")

        #append output layer
        #output layer has no output connections or activation function
        self.layers.append(layer.Layer(nodes_per_layer[-1], 0, 0 ))
        print("added 1 output layer")

    def propogate_forward(self):
        print("our layers are", self.layers)
        for currlayer, nextlayer in zip(self.layers[:-1], self.layers[1:]):
            print("layers in for prog", currlayer)
            print("weights", currlayer.weights)
            #calculate weighted sum
            currlayer.wsum = currlayer.weights.dot(currlayer.input) + currlayer.bias
            #apply activation
            currlayer.output = currlayer.activation(currlayer.wsum)
            #set next layers input as the output frm this layer
            nextlayer.input = currlayer.output


    def propogate_backward(self):
        wgrad, bgrad = [], []
        #use one hot encoding with 2 output nodes
        if self.layers[-1].num_nodes != 1:
            true_Y = one_hot(self.output)
        else: 
            true_Y = self.output

        layer = self.layers[-2]
        error = layer.output - true_Y
        print("my error function", self.loss_function)
        print("the error was", error.shape, "but the error should be",)

        
        m = self.output.size
        
        for i in range(-2, -(len(self.layers)+1), -1):
           smn = 1/ m * error.dot(self.layers[i].input.T)
           #print("appending the change for wgrad", wgrad)
           wgrad.append(smn)
           smn = 1/m *np.sum(error)
           bgrad.append(smn)
           if i == -(len(self.layers)):
            break
           error = self.layers[i].weights.T.dot(error) * self.layers[i-1].activation_prime(self.layers[i-1].wsum)
        

        self.parameter_update(wgrad, bgrad)

    def parameter_update(self, wgrad, bgrad):
        for layer, wupdate, bupdate in zip(reversed(self.layers[:-1]), wgrad, bgrad):
            print("dimensions", layer, layer.weights.shape, wupdate.shape, bupdate.shape)
            layer.weights = layer.weights - (self.learning_rate * wupdate)
            layer.bias = layer.bias - (self.learning_rate * bupdate)
    
    def lrschedule(self, epoch):
       self.learning_rate *= 1/(1 + self.decay * (epoch))

    def train_sgd(self):
        for j in range(self.training_epochs):
            self.lrschedule(j)
            for i in range(200):
                self.propogate_forward()
                self.propogate_backward()

                if i%10 == 0:
                    print("Iteration: ", i)
                    predictions = get_predictions(self.layers[-1].input)
                    print("Accuracy ", get_accuracy(predictions, self.output))
            predictions = get_predictions(self.layers[-1].input)
        print("Final Accuracy ", get_accuracy(predictions, self.output))


def get_predictions(A2):
    if A2.shape[0] <= 1:
        temp = A2 > 0.5 # 0.5 is threshold value for output node 1
        return temp.astype(int)
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(np.sum(predictions == Y))
    return np.sum(predictions == Y) / Y.size

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y



# adding epsilon to the predicted output to avoid log(0) error and for stability
EPSILON = 1e-7  

def bce(pred, y):
    print("the error is", -np.mean((y * np.log(pred + EPSILON)) + (1 - y ) * np.log(1 - pred + EPSILON)))
    return -np.mean((y * np.log(pred + EPSILON)) + (1 - y ) * np.log(1 - pred + EPSILON))

def mse(pred, y):
    print("first step substract", y - pred)
    print("step 2 square", )
    return np.mean(np.square(np.subtract(y,pred)))

def mae(pred, y):
    return np.mean(abs(y - pred))

    

    
         
