
class ANN:

    """initialise network with hyperparameters"""
    def __init__(self):
        self.layers = [] #add input layer by default?
        self.loss_function = None
        self.num_hidden_layers = None
        #self.learning_rate = None
        #self.training_epochs = None
        #self.GD_type = None
        #self.dropout_rate = None

    """train ANN"""
    def train(self):
        pass

    """test ANN"""
    def test(self):
        #cross validation, train-test split or both
        pass


#loss functions
"""mean squared error loss function"""
def mse():
    pass

def b_cross_entropy():
    pass

def hinge_loss():
    pass








