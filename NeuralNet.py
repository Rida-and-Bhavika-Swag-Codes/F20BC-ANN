import layer
import numpy as np
import time
class ANN:

    """
    Initialise network with hyperparameters
    """
    def __init__(self, input, output, learn_rate = 0.1 , epoch = 200, loss = 2, lrschedule = 0, typegd = 3, bsize = 30):

        self.input = input # input vector 
        self.output = output # target class
        self.layers = [] 
        
        self.learning_rate = learn_rate
        self.training_epochs = epoch

        self.lrsched = None 
        self.decay = None

        if lrschedule == 1: # =0 when using constant lr, else =1 with decay
            self.lrsched = 1
            self.decay = self.learning_rate/self.training_epochs # for learning rate scheduler 

        self.typegd = self.train_sgd
        self.batchsize = 1
        
        # set type of gradient descent
        match typegd:
            case 1: #default stochastic gradient descent
                pass

            case 2: #batch gradient descent
                self.batchsize = self.input.shape[1] 
                self.typegd = self.train_mbgd

            case 3: #mini batch gradient descent
                self.batchsize = bsize
                self.typegd = self.train_mbgd

            case other:
                print("incorrect gradient descent option, defaulting to stochastic")

        match loss:
            case 1 : self.loss_function = bce
            case 2 : self.loss_function = hinge_loss
            case 3 : self.loss_function = square_loss
            case other: 
                print("no loss function selected")
                self.loss_function = None





    """Function to set initial activations of all layers"""
    def setLayers(self, activations, nodes_per_layer):
        # append input layer
        l = layer.Layer(30, nodes_per_layer[0], activations[0])
        # assign the dataset features as input to the first layer
        l.input = self.input
        self.layers.append(l)

        # append hidden layers
        for i in range (len(nodes_per_layer)-1):
            l = layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1])
            self.layers.append(l)
        # append output layer
        l = layer.Layer(nodes_per_layer[-1], 0, 0)
        self.layers.append(l)  # output layer has no output connections or activation function
       



    """Function to propgate all layers : finding the activations of all layers"""
    def propogate_forward(self):
        for currlayer, nextlayer in zip(self.layers[:-1], self.layers[1:]):
            #calculate weighted sum
            currlayer.wsum = currlayer.weights.dot(currlayer.input) + currlayer.bias
            #apply activation
            currlayer.output = currlayer.activation(currlayer.wsum)
            #set next layers input as the output frm this layer 
            nextlayer.input = currlayer.output




    """Function to propgate all layers backwards"""
    def propogate_backward(self, truey):

        # hold cache of gradient vector for weights and biases
        wgrad, bgrad = [], []
        # use one hot encoding with 2 output nodes
        if self.layers[-1].num_nodes != 1:
            true_y = one_hot(truey)
        else: # with only one node, no need to use one hot encoding
            true_y = truey

        # find the error between the predicted value of the network and the true value(labels)
        layer = self.layers[-2]
        error = layer.output - true_y
        # no. of samples in the current train/test set
        m = truey.size
        
        # back propogate all other layers
        for i in range(-2, -(len(self.layers) + 1), -1):

            wgrad.append((1/m) * np.dot(error, self.layers[i].input.T))
            bgrad.append((1/m) * np.sum(error))

            if i == -len(self.layers):
                break
            # find the error of the next layer
            error = np.dot((self.layers[i].weights).T, error) * self.layers[i-1].activation_prime(self.layers[i-1].wsum)
        
        #after back prop, update the weights and biases
        self.parameter_update(wgrad, bgrad)




    """Function used to update parameters: use after backprop"""
    def parameter_update(self, wgrad, bgrad):
        # updating weights and bias per layer
        for layer, wupdate, bupdate in zip(reversed(self.layers[:-1]), wgrad, bgrad):
            layer.weights = layer.weights - (self.learning_rate * wupdate)
            layer.bias = layer.bias - (self.learning_rate * bupdate)
    



    """Function to set a new learning rate when using learning rate schedule"""
    def lrschedule(self, epoch):
        # learning rate to decrease with each epoch
        self.learning_rate *= 1/(1 + self.decay * (epoch))



    """Function to train with stochastic gradient descent"""
    def train_sgd(self):
        start_time = time.time() 
        loss = []
        acc = []

        for i in range(self.training_epochs):

            if self.lrsched:
                #set in the learning schedule 
                self.lrschedule(i)
            # update weights after each sample has been propogate forward and backward
            self.propogate_forward()
            #calculate loss/cost
            predictions = get_predictions(self.layers[-1].input)
            loss.append(self.loss_function(predictions, self.output))
            acc.append(get_accuracy(predictions, self.output))
            self.propogate_backward(self.output)

        end_time = time.time() 
        return sum(loss)/len(loss), sum(acc)/len(acc), end_time - start_time
    



    """
    Function to create mini batches for mini-batch gradient descent
    reference: https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
    """
    def create_batches(self):
        # create a list to hold all minibatches
        mini_batches = []
        #concatenate each sample with its corresponding label
        y = np.resize(self.output, (len(self.input.T),1))
        data = np.hstack((self.input.T, y)) 
        #shuffle data
        np.random.shuffle(data)

        for i in range((data.shape[0] // self.batchsize)):
            #slice a minibatch out of the data list
            mini_batch = data[i * self.batchsize:(i + 1)*self.batchsize, :]
            #seperate the previously concatenated X and Y data before training
            X_mini = mini_batch[:, :-1].T
            Y_mini = mini_batch[:, -1]
            #add new mini batch 
            mini_batches.append((X_mini, Y_mini))

        # when batch size doesnt divide the no.of samples perfectly, seperately append the last batch (will be of a different size)
        if data.shape[0] % self.batchsize != 0:
            mini_batch = data[i * self.batchsize:data.shape[0]]
            X_mini = mini_batch[:, :-1].T
            Y_mini = mini_batch[:, -1]
            mini_batches.append((X_mini, Y_mini))
        return mini_batches


    """Function to train with batch and mini-batch gradient descent"""
    def train_mbgd(self):

        start_time = time.time() 
        loss = []
        acc = []
        for i in range(self.training_epochs):

            if self.lrsched:
                #set in the learning schedule 
                self.lrschedule(i)

            mbatches = self.create_batches()
            
            for mbatch in mbatches:
                x, y = mbatch
                #set the first layers input as the current minibatch
                self.layers[0].input = x
                self.propogate_forward()
                #calculate loss/cost
                predictions = get_predictions(self.layers[-1].input)
                loss.append(self.loss_function(predictions, y))
                acc.append(get_accuracy(predictions, y))
                self.propogate_backward(y.flatten())

        end_time = time.time()
        return sum(loss)/len(loss), sum(acc)/len(acc), end_time - start_time




    """"""    
    def train(self):
        return self.typegd()

    def test(self, input, output):
        # setting X_test as input and y_test as output
        self.input = input
        self.output = output

        # updating input layer input
        self.layers[0].input = self.input
        self.propogate_forward()

        predictions = get_predictions(self.layers[-1].input)
        return get_accuracy(predictions, self.output)
    



    """Function to display properties of the network - useful for debugginng"""
    def get_properties(self):
        print("Number of layers:", len(self.layers))
        print("Learning rate:", self.learning_rate)
        print("Training epochs:", self.training_epochs)
        print("Loss function:", self.loss_function)
        print("Decay:", self.decay)


""" HELPER FUNCTIONS """

THRESHOLD = 0.5

def get_predictions(output):
    if output.shape[0] <= 1:
        boolnp = output > THRESHOLD # 0.5 is threshold value for output node 1
        return boolnp[0].astype(int)
    return np.argmax(output, 0)

def get_accuracy(predictions, y_true):
    return np.sum(predictions == y_true) / y_true.size

# one hot encode for output nodes > 1
def one_hot(y):
    y = y.astype(int)
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


""" LOSS FUNCTIONS """

EPSILON = 1e-7

# binary cross entropy
# reference: #https://stackoverflow.com/questions/67615051/implementing-binary-cross-entropy-loss-gives-different-answer-than-tensorflows
def bce(y_pred,y_true): 
    y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
    return -np.mean(y_true * np.log(y_pred + EPSILON)  + (1 - y_true) * np.log(1 - y_pred + EPSILON))

# hinge loss
# reference: https://stats.stackexchange.com/questions/539496/how-to-create-hinge-loss-function-in-python-from-scratch
def hinge_loss(y_pred, y_true):
    npred = np.array([-1 if i == 0 else i for i in y_pred])
    ntrue = np.array([-1 if i == 0 else i for i in y_true])

    return np.mean([max(0, 1 - act * pred) for act, pred in zip(ntrue, npred)])

# square loss - this is for classification, formula is different than for regression
# formula reference: https://math.stackexchange.com/questions/2370977/square-loss-function-in-classification
def square_loss(y_pred, y_true):
    return np.mean(np.square(1 - y_true * y_pred))