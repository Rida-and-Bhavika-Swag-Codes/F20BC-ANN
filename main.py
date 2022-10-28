import dataset
import NeuralNet

def main():
    # loading the dataset
    data = dataset.load()

    print("Initializing Feedforward Neural Network..\n")

    # asking user for hyperparameters
    nodes = []
    activations = []
    #input_nodes = int(input("Enter number of nodes in input layer:"))

    nhiddenlayers = int(input("\nEnter number of hidden layers:"))
    for i in range(nhiddenlayers):
        nodes.append(int(input("Enter number of hidden neurons in hidden layer " + str((i + 1)) + ":")))
        activations.append(int(input("\nSelect hidden layer's activation function" + 
        "\n1. Hyperbolic tangent (tanh𝑥)\n2. ReLU (Rectified Linear Unit)\n3. Logistic function (Sigmoid)\n")))

    nodes.append(int(input("\nEnter number of output nodes (1 or 2):")))
    
    activations.append(int(input("\nSelect output layer's activation function" + 
        "\n1. Hyperbolic tangent (tanh𝑥)\n2. ReLU (Rectified Linear Unit)\n3. Logistic function (Sigmoid)\n")))


    lr = float(input("\nEnter learning rate:"))
    lrschedule = input("Control learning rate? (learning rate scheduler) [y/N]:")
    if lrschedule.lower() == "y":
        lrschedule = 1
        print("Learning rate scheduler active")
    else:
        lrschedule = 0
        print("Learning rate will remain constant")
    
    epoch = int(input("\nEnter number of training epochs:"))
    loss = int(input("\nSelect type of loss function\n1. Binary Cross Entropy\n2. Hinge Loss\n3. Square Loss\n"))

    typegd = int(input("\nSelect Gradient Descent Algorithm\n1. Stochastic\n2. Batch\n3. Mini Batch\n"))
    bsize = 1
    if typegd == 3:
        bsize = int(input("Enter the number of mini batches:"))

    cv_or_split = int(input("\nDo K - Fold Cross Validation (Enter 1) or Train - Test Split (Enter 2):"))

    if cv_or_split not in [1, 2]:
        print("incorrect/no option selected, defaulting to train/test split")

    if cv_or_split == 1:
        print("\nStarting K Fold Cross Validation")
        nfolds = int(input("Please enter the number of folds (1 - 10):"))
        dataset.kfoldcv(data, nfolds, lr, epoch, loss, lrschedule, typegd , bsize, activations, nodes)

    else:
        print("\nStarting Train Test Split")
        # splitting
        X_train, y_train, X_test, y_test = dataset.train_test(data)

        print("\nTraining Neural Network..")
        ann = NeuralNet.ANN(X_train, y_train, lr, epoch, loss, lrschedule, typegd , bsize)
        ann.setLayers(activations, nodes)

        ann.get_properties()
        
        avgloss, accuracy, time = ann.train()
        print("\nAverage loss over", epoch, "epoch(s) is", avgloss)
        print("Average training accuracy over", epoch, "epoch(s) is", round(accuracy * 100, 2), "%")
        print("Training time over", epoch, "epoch(s) is", time, "seconds")

        print("\nTesting Neural Network..")
        test = ann.test(X_test, y_test)
        print("Testing accuracy is", round(test * 100, 2), "%")

if __name__ == '__main__':
  main()