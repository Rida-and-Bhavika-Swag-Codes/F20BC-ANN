import dataset
import NeuralNet
from ipynb.fs.defs.crossvalidation import kfoldcv

def main():
    # loading the dataset
    data = dataset.load()

    print("Initializing Feedforward Neural Network..\n")

    # asking user for hyperparameters
    nodes = []
    activations = []
    #input_nodes = int(input("Enter number of nodes in input layer:"))
    activations.append(int(input("Select input layer's activation function" + 
        "\n1. Hyperbolic tangent (tanh𝑥)\n2. ReLU (Rectified Linear Unit)\n3. Logistic function (Sigmoid)\n")))

    nhiddenlayers = int(input("Enter number of hidden layers:"))
    for i in range(nhiddenlayers):
        nodes.append(int(input("Enter number of hidden neurons in hidden layer " + str((i + 1)) + ":")))
        activations.append(int(input("Select hidden layer's activation function" + 
        "\n1. Hyperbolic tangent (tanh𝑥)\n2. ReLU (Rectified Linear Unit)\n3. Logistic function (Sigmoid)\n")))

    nodes.append(int(input("Enter number of output nodes:")))

    lr = float(input("Enter learning rate:"))
    lrschedule = input("Control learning rate? (learning rate scheduler) [y/N]")
    if lrschedule.lower() == "y":
        lrschedule = 1
        print("Learning rate scheduler active")
    else:
        lrschedule = 0
        print("Learning rate will remain constant")
    
    epoch = int(input("Enter number of training epochs:"))
    loss = int(input("Select type of loss function\n1. Binary Cross Entropy\n2. Hinge Loss\n3. Square Loss\n"))
    #gd = int(input("Select Gradient Descent Algorithm\n1. Stochastic\n2. Batch\n3. Mini Batch\n"))

    cv_or_split = int(input("Do K - Fold Cross Validation (Enter 1) or Train - Test Split (Enter 2)"))

    if cv_or_split not in [1, 2]:
        print("incorrect/no option selected, defaulting to train/test split")

    if cv_or_split == 1:
        ann = NeuralNet.ANN(0, 0, lr, epoch, loss, lrschedule)
        kfoldcv(ann, data)

    else:
        print("\nTraining Neural Network..")
        # splitting
        X_train, y_train, X_test, y_test = dataset.train_test(data)
        ann = NeuralNet.ANN(X_train, y_train, lr, epoch, loss, lrschedule)
        ann.setLayers(activations, nodes)

        ann.get_properties()
        
        avgloss, accuracy, time = ann.train_sgd()
        print("\nAverage loss over", epoch, "epoch(s) is", avgloss)
        print("Training accuracy over", epoch, "epoch(s) is", round(accuracy * 100, 2), "%")
        print("Training time over", epoch, "epoch(s) is", time, "seconds")

        print("\nTesting Neural Network..")
        test = ann.test(X_test, y_test)
        print("Testing accuracy is", round(test * 100, 2), "%")

main()