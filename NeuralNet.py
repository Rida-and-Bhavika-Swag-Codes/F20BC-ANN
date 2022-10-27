import layer
import numpy as np
class ANN:

    """
    Initialise network with hyperparameters
    """
    def __init__(self, input, output, learn_rate = 0.1 , epoch = 1, loss = 1, lrschedule = 0):
        self.input = input # input vector 
        self.output = output # target class
        self.layers = [] 
        
        self.learning_rate = learn_rate
        self.training_epochs = epoch

        self.lrsched = None
        self.decay = None

        match loss:
            case 1 : self.loss_function = bce
            case 2 : self.loss_function = hinge_loss
            case 3 : self.loss_function = square_loss
            case other: 
                print("no loss function selected")
                self.loss_function = None

        if lrschedule == 1:
            self.lrsched = lrschedule # =0 when using constant lr, else =1 with decay
            self.decay = self.learning_rate/self.training_epochs # for learning rate scheduler 

    def setLayers(self, activations, nodes_per_layer):

        # append input layer
        l = layer.Layer(30, nodes_per_layer[0], activations[0])
        # assign the dataset features as input to the first layer
        l.input = self.input
        self.layers.append(l)
        print("added input layer")
        layer.Layer.get_properties(l)

        # append hidden layers
        for i in range (len(nodes_per_layer)-1):
            l = layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1])
            self.layers.append(l)
            print("added 1 hidden layer")
            layer.Layer.get_properties(l)

        # append output layer
        l = layer.Layer(nodes_per_layer[-1], 0, 0)
        self.layers.append(l)  # output layer has no output connections or activation function
        print("added output layer")
        layer.Layer.get_properties(l)

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

        # use one hot encoding with 2 output nodes
        if self.layers[-1].num_nodes != 1:
            true_y = one_hot(self.output)
        else: # with only one node, no need to use one hot encoding
            true_y = self.output

        # find the error between the predicted value of the network and the true value(labels)
        layer = self.layers[-2]
        error = layer.output - true_y

        # no. of samples in the current train/test set
        m = self.output.size
        
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


    def parameter_update(self, wgrad, bgrad):
        # updating weights and bias per layer
        for layer, wupdate, bupdate in zip(reversed(self.layers[:-1]), wgrad, bgrad):
            layer.weights = layer.weights - (self.learning_rate * wupdate)
            layer.bias = layer.bias - (self.learning_rate * bupdate)
    
    def lrschedule(self, epoch):
        # learning rate to decrease with each epoch
        self.learning_rate *= 1/(1 + self.decay * (epoch))

    def train_sgd(self):
        loss = []
        for j in range(self.training_epochs):

            if self.lrsched:
                #set in the learning schedule 
                self.lrschedule(j)

            # update weights after each sample has been propogate forward and backward
            for i in range(200):
                self.propogate_forward()
                loss.append(self.loss_function(predictions, self.output))
                self.propogate_backward()

            predictions = get_predictions(self.layers[-1].input)
        return sum(loss)/len(loss), get_accuracy(predictions, self.output)
        

    def test(self, input, output):
        # setting X_test as input and y_test as output
        self.input = input
        self.output = output

        # updating input layer input
        self.layers[0].input = self.input
        self.propogate_forward()

        predictions = get_predictions(self.layers[-1].input)
        return get_accuracy(predictions, self.output)
    
    def get_properties(self):
        print("Number of layers:", len(self.layers))
        print("Learning rate:", self.learning_rate)
        print("Training epochs:", self.training_epochs)
        print("Loss function:", self.loss_function)
        print("Decay:", self.decay)



    def train_mini_batch(self, x, y, bsize):
        # assert x.size == y.size 
        mbatches = []
        data = np.arange(x.size)
        np.random.shuffle(data)
        for i in range(0, x.size, bsize):
            last = min(i + bsize, x.size)
            batch = data[i:last]
            mbatches.append((x[batch], y[batch]))
        return mbatches
        
    # def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #     if shuffle:
    #         indices = np.arange(inputs.shape[0])
    #         np.random.shuffle(indices)
    #     for start_idx in range(0, inputs.shape[0], batchsize):
    #         end_idx = min(start_idx + batchsize, inputs.shape[0])
    #         if shuffle:
    #             excerpt = indices[start_idx:end_idx]
    #         else:
    #             excerpt = slice(start_idx, end_idx)
    #         yield inputs[excerpt], targets[excerpt]
    """
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