import numpy as np
import pandas as pd
import NeuralNet

def load():
    datasetNames = ['ID number', 'Diagnosis', 
                    'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
                    'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness', 'SE Compactness', 'SE Concavity', 'SE Concave Points', 'SE Symmetry', 'SE Fractal Dimension',
                    'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension']
    data = pd.read_csv('data/wdbc.data', names = datasetNames, sep = ',')
    data = data.replace({'M': 0, 'B':1})
    data = data.drop(columns=['ID number'])
    return data


# normalizing X
# reference: https://www.kaggle.com/code/joshbeau/tumor-diagnosis-neural-net-from-first-principals
def normalize(X):
    X_mean = np.mean(X, axis = 1, keepdims=True) # mean of each feature
    X_std = np.std(X, axis = 1, keepdims=True) # standard deviation of each feature
    return (X - X_mean)/(X_std) 

""" TRAIN TEST SPLIT """
def train_test(data):
    data = np.array(data)
    np.random.shuffle(data)
    
    # spliting to approx 80% train and 20% test
    train = data[115:data.shape[0]].T
    test = data[:114].T

    # normalizing X train
    X_train = normalize(train[1:])
    y_train = train[0].astype(int)

    # normalizing X test
    X_test = normalize(test[1:])
    y_test = test[0].astype(int)
    
    return X_train, y_train, X_test, y_test



""" K FOLD CROSS VALIDATION """

# creating folds (num of folds = nfolds)
def kfolds(data, nfolds):
    folds = []
    foldLength = int(data.shape[0]/nfolds)

    # shuffling data
    data = data.reindex(np.random.permutation(data.index))                                                          
    data = data.reset_index(drop = True)
    
    # creating folds
    idx = 0
    lidx = foldLength - 1
    for i in range(nfolds):
        folds.append(data.loc[idx : lidx])
        idx += foldLength
        lidx += foldLength
    return folds

# splitting each fold to training and testing 
# returning nested list
def kfoldsplit(folds):
    train_test = []

    for i in range(len(folds)):
        temp = []
        # folds_copy is training data
        folds_copy = folds.copy()
        # test is testing data
        test = folds_copy.pop(i)

        temp.append(pd.concat(folds_copy))
        temp.append(test)
        
        train_test.append(temp)
    # each element of train_test contains a list of train values
    # and a second list of test values
    return train_test


# running k fold cross validation
def kfoldcv(data, nfolds = 5, learn_rate = 0.1 , epoch = 200, loss = 1, lrschedule = 0, typegd = 3, bsize = 30, 
            activations = [3, 3, 3], nodes = [5, 5, 2]):

    folds = kfolds(data, nfolds)
    train_test = kfoldsplit(folds)

    accuracies = []
    fold = 1
    for i in train_test:
        print("\nCross Validating Fold", fold)
        X_test = np.asarray(i[-1].drop(i[0].columns[0], axis = 1)).T
        y_test = np.array(i[-1].iloc[:,0]).T

        X_test = normalize(X_test)

        X_train = np.array(i[0].drop(i[0].columns[0], axis = 1)).T
        y_train = np.array(i[0].iloc[:,0]).T
        
        X_train = normalize(X_train)


        ann = NeuralNet.ANN(X_train, y_train, learn_rate, epoch, loss, lrschedule, typegd , bsize)
        ann.setLayers(activations, nodes)
        ann.get_properties()

        print("\nTraining Neural Network..")
        avgloss, accuracy, time = ann.train()

        print("\nAverage loss over", ann.training_epochs, "epoch(s) in fold", fold, "is", avgloss)
        print("Average training accuracy over", ann.training_epochs, "epoch(s) in fold", fold, "is", round(accuracy * 100, 2), "%")
        print("Training time over", ann.training_epochs, "epoch(s) in fold", fold, "is", time, "seconds")

        print("\nTesting Neural Network..")
        accuracy = ann.test(X_test, y_test)
        accuracies.append(accuracy)
        print("Testing accuracy for fold", fold, "is", round(accuracy * 100, 2), "%")

        # for printing purposes
        fold += 1
    
    print("\nThe testing accuracies are:", ", ".join(str(i) for i in accuracies))
    print("Average accuracy is", round((sum(accuracies)/nfolds) * 100, 2), "%")
    return accuracies