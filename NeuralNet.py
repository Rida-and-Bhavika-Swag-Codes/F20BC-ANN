import layer
import numpy as np
class ANN:

    """
    Initialise network with hyperparameters
    Parameters: 
    """
    def __init__(self, learn_rate, loss, input, output, lrschedule):
        self.input = input #input vector 
        self.output = output #Target class
        self.layers = [] #assume 7 nodes in the next hidden layer
        
        self.learning_rate = learn_rate
        self.training_epochs = 2

        self.lrsched = lrschedule #=0 when using constant lr, else =1 with decay

        self.decay = self.learning_rate/self.training_epochs # for learning rate scheduler 

        match loss:
            # binary cross entropy
            case 1 : self.loss_function = bce
            # hinge loss
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
        l = layer.Layer(30, nodes_per_layer[0], activations[0])
        #assign the dataset features as input to the first layer
        l.input = self.input
        self.layers.append(l)
        print("adding input layer")

        #append hidden layers
        for i in range (len(nodes_per_layer)-1):
            self.layers.append(layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1]))
            print("added 1 hidden layer")

        #append output layer
        self.layers.append(layer.Layer(nodes_per_layer[-1], 0, 0 ))  #output layer has no output connections or activation function
        print("added 1 output layer")

    def propogate_forward(self):
        for currlayer, nextlayer in zip(self.layers[:-1], self.layers[1:]):
            #calculate weighted sum
            currlayer.wsum = currlayer.weights.dot(currlayer.input) + currlayer.bias
            #apply activation
            currlayer.output = currlayer.activation(currlayer.wsum)
            #set next layers input as the output frm this layer 
            nextlayer.input = currlayer.output

    def propogate_backward(self):
        # hold cache of gradient vector for weights and biases
        wgrad, bgrad = [], []

        #use one hot encoding with 2 output nodes
        if self.layers[-1].num_nodes != 1:
            true_Y = one_hot(self.output)
        else: #with only one node, no need to use one hot encoding
            true_Y = self.output

        #find the error between the predicted value of the network and the true value(labels)
        layer = self.layers[-2]
        error = layer.output - true_Y



        
        #no. of samples in the current train/test set
        m = self.output.size
        
        #back propogate all other layers
        for i in range(-2, -(len(self.layers)+1), -1):
            
            wgrad.append((1/ m) * np.dot(error, self.layers[i].input.T))
            bgrad.append((1/m) *np.sum(error))

            if i == -(len(self.layers)):
                break
            #find the error of the next layer
            error = np.dot((self.layers[i].weights).T,error) * self.layers[i-1].activation_prime(self.layers[i-1].wsum)
        
        #after back prop, update the weights and biases
        self.parameter_update(wgrad, bgrad)


    def parameter_update(self, wgrad, bgrad):
        for layer, wupdate, bupdate in zip(reversed(self.layers[:-1]), wgrad, bgrad):
            layer.weights = layer.weights - (self.learning_rate * wupdate)
            layer.bias = layer.bias - (self.learning_rate * bupdate)
    
    def lrschedule(self, epoch):
       self.learning_rate *= 1/(1 + self.decay * (epoch))

    def train_sgd(self):

        
        for j in range(self.training_epochs):

            if self.lrschedule:
                #set in the learning schedule 
                self.lrschedule(j)

            # update weights after each sample has been propogate forward and backward

            for i in range(200):
                self.propogate_forward()
                self.propogate_backward()

                """remove before submission?"""
                if i%10 == 0:
                    print("Iteration: ", i)
                    predictions = get_predictions(self.layers[-1].input)
                    print("Accuracy ", get_accuracy(predictions, self.output))
            predictions = get_predictions(self.layers[-1].input)
        print("Final Accuracy ", get_accuracy(predictions, self.output))
        

    def test(self, input, output):
        self.input = input
        self.output = output

        self.layers[0].input = self.input
        self.propogate_forward()

        predictions = get_predictions(self.layers[-1].input)
        return get_accuracy(predictions, self.output)

"""
    def train_mini_batch(x,y,bsize):
        # assert x.size == y.size
        mbatches = []
        data = np.arange(x.size)
        np.random.shuffle(data)
        for i in range(0, x.size, bsize):
            last = min(i + bsize, x.size)
            batch = data[i:last]
            mbatches.append((x[batch], y[batch]))
        return mbatches

    def gradientDescent(X, y, learning_rate=0.001, batch_size=32):
        theta = np.zeros((X.shape[1], 1))
        error_list = []
        max_iters = 3
        for itr in range(max_iters):
            mini_batches = create_mini_batches(X, y, batch_size)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                theta = theta - learning_rate * gradient(X_mini, y_mini, theta)
                error_list.append(cost(X_mini, y_mini, theta))
"""

def get_predictions(A2):
    print("HERE")
    print("A2", A2)
    if A2.shape[0] <= 1:
        temp = A2 > 0.5 # 0.5 is threshold value for output node 1
        return temp.astype(int)
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):


    return np.sum(predictions == Y) / Y.size

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y





# # adding epsilon to the predicted output to avoid log(0) error and for stability
# EPSILON = 1e-7  

# def bce(pred, y):
#     return -np.mean((y * np.log(pred + EPSILON)) + (1 - y ) * np.log(1 - pred + EPSILON))

"""                                     LOSS FUNCTIONS                            """
#BINARY CROSS ENTROPY
def bce(y_pred,y_true): #https://stackoverflow.com/questions/67615051/implementing-binary-cross-entropy-loss-gives-different-answer-than-tensorflows
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)  
    print("the error is",  -np.mean(term_0+term_1))
    return -np.mean(term_0+term_1)


