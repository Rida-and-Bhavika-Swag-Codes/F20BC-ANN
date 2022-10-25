import layer
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
        match loss:
            case 1 : self.loss_function, self.loss_prime = bce, dbce
            case 2 : self.loss_function, self.loss_prime = mse, dmse
            case 3 : self.loss_function, self.loss_prime = mae, dmae
            case other: print("no loss function selected")

    def setLayers(self, activations, *nodes_per_layer):
        print("the activations are", activations)
        print("the variable argument is ", nodes_per_layer)
        print("number of hidden layers", len(nodes_per_layer)-1, "\n")
        print("input to the network is", self.input)
        print("the output is", self.output)

        #append input layer
        #l = layer.Layer(len(self.input), nodes_per_layer[0], activations[0])
        l = layer.Layer(30, 10, activations[0])
        self.layers.append(l)
        print("adding input layer")

        #append hidden layers
        for i in range (len(nodes_per_layer)-1):
            #self.layers.append(layer.Layer(nodes_per_layer[i], nodes_per_layer[i+1], activations[i+1]))
            l = layer.Layer(10, 2, activations[0])
            print("added 1 hidden layer")

        #append output layer
        #self.layers.append(layer.Layer(nodes_per_layer[-1], 0, 0 ))#output layer has no output connections or activation function
        self.layers.append(layer.Layer(2, 0, 0 ))
        print("added 1 output layer")

    def propogate_forward(self):
        # propogate through the first layer
        # first_layer = self.layers[0]
        # wsum = first_layer.weights.dot(self.input) + first_layer.bias
        # first_layer.output = first_layer.activation(2)(wsum)


        for currlayer, nextlayer in zip(self.layers[:-1], self.layers[1:]):
            print("weights", currlayer.weights)
            print("the activations of this layer", currlayer.input)
            #calculate weighted sum
            wsum = currlayer.weights.dot(currlayer.input) + currlayer.bias
            #apply activation
            currlayer.output = currlayer.activation(2)(wsum)
            #set next layers input as the output frm this layer
            nextlayer.input = currlayer.output
            


    def train(self):
        for i in range(200):
            self.propogate_forward()
            # self.propogate_backward()

            # if i%10 == 0:
            #     print("Iteration: ", i)
            #     predictions = get_predictions(self.layers[-1].input)
            #     print("Accuracy ", get_accuracy(predictions, self.output))


    

    
         