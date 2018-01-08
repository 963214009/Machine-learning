from __future__ import division  # floating point division
from sklearn import preprocessing
import numpy as np
#import utilities as utils
import math
import random
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            print(self.params, parameters)
            self.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = self.threshold_probs(probs)
        return ytest

    def update_dictionary_items(self, dict1, dict2):
        """ Replace any common dictionary items in dict1 with the values in dict2
        There are more complicated and efficient ways to perform this task,
        but we will always have small dictionaries, so for our use case, this simple
        implementation is acceptable.
        """
        for k in dict1:
            if k in dict2:
                dict1[k]=dict2[k]

    def threshold_probs(self,probs):
        """ Converts probabilities to hard classification """
        classes = np.ones(len(probs), )
        classes[probs < 0.5] = 0
        return classes


#############################################################

class LinearSVC(Classifier):

	def __init__(self, parameters = {}):
		self.weight = svm.LinearSVC()
		self.params = {'regwgt':0.0}
		self.reset(parameters)

	def learn(self, Xtrain, ytrain):
		self.weight.fit(Xtrain, ytrain)

	def predict(self, Xtest):
		ytest = self.weight.predict(Xtest)

		return ytest

#############################################################
class Gaussian_NB(Classifier):

	def __init__(self, parameters = {}):
		self.weight = GaussianNB()
		self.params = {'regwgt':0.0}
		self.reset(parameters)

	def learn(self, Xtrain, ytrain):
		self.weight = self.weight.fit(Xtrain, ytrain)

	def predict(self, Xtest):
		ytest = self.weight.predict(Xtest)

		return ytest

#############################################################
class logit(Classifier):

	def __init__(self, parameters = {}):
		self.weight = linear_model.SGDClassifier(loss='log', penalty='l2', max_iter=100)
		self.params = {'regwgt':0.0}
		self.reset(parameters)

	def learn(self, Xtrain, ytrain):
		self.weight = self.weight.fit(Xtrain, ytrain)

	def predict(self, Xtest):
		ytest = self.weight.predict(Xtest)

		return ytest