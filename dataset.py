import numpy as np
import pandas as pd

def load():
    datasetNames = ['ID number', 'Diagnosis', 
                    'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
                    'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness', 'SE Compactness', 'SE Concavity', 'SE Concave Points', 'SE Symmetry', 'SE Fractal Dimension',
                    'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension']
    data = pd.read_csv('data/wdbc.data', names = datasetNames, sep = ',')
    data = data.replace({'M': 0, 'B':1})
    data = data.drop(columns=['ID number'])
    return data

def train_test(data):
    data = np.array(data)
    np.random.shuffle(data)
    
    # spliting to approx 80% train and 20% test
    train = data[115:data.shape[0]].T
    test = data[:114].T

    X_train = normalize(train[1:])
    y_train = train[0].astype(int)

    X_test = normalize(test[1:])
    y_test = test[0].astype(int)
    
    return X_train, y_train, X_test, y_test

# normalizing X
# reference: https://www.kaggle.com/code/joshbeau/tumor-diagnosis-neural-net-from-first-principals
def normalize(X):
    X_mean = np.mean(X, axis = 1, keepdims=True) # mean of each feature
    X_std = np.std(X, axis = 1, keepdims=True) # standard deviation of each feature
    return (X - X_mean)/(X_std) 