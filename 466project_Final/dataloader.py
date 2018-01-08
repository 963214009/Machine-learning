from __future__ import division  # floating point division
import math
import numpy as np
from sklearn.datasets import load_svmlight_file
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_susy():
    filename = 'datasets/Frogs_MFCCs.csv'
    dataset = pandas.read_csv(filename, delimiter=",")
    #print(dataset)
    le = preprocessing.LabelEncoder()
    a = dataset


    # n = len(dataset.columns.values)
    # print(dataset.values[:,n-2])


    for i in dataset.columns.values:
        if is_number(dataset[i][1]) == False:
            #dataset[i] = le.fit(dataset[i])
            dataset[i] = le.fit_transform(a[i])
    #print("####################")
    #print(dataset)
    #dataset = loadcsv(filename)

    #trainset, testset = splitdataset(dataset,trainsize, testsize)
    n = len(dataset.columns.values)
    #print(n)
    x = dataset.values[:, :n-4]
    #print(x)
    y = dataset.values[:,n-2]
    print(y)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y)
    return (Xtrain,ytrain),(Xtest,ytest)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass


    return False
def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset


def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    # Generate random indices without replacement, to make train and test sets disjoint
    np.random.seed(123)
    randindices = np.random.choice(dataset.shape[0], trainsize + testsize, replace=False)
    featureend = dataset.shape[1] - 1
    outputlocation = featureend
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0

    Xtrain = dataset[randindices[0:trainsize], featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize], outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize + testsize], featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize + testsize], outputlocation]

    if testdataset is not None:
        Xtest = dataset[:, featureoffset:featureend]
        ytest = dataset[:, outputlocation]

        # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:, ii]))
        if maxval > 0:
            Xtrain[:, ii] = np.divide(Xtrain[:, ii], maxval)
            Xtest[:, ii] = np.divide(Xtest[:, ii], maxval)

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0], 1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0], 1))))

    return ((Xtrain, ytrain), (Xtest, ytest))