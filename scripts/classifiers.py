#* classifiers.py
#*
#* ANLY 555 Fall 2022
#* Project Deliverable #3
#*
#* Due on Monday October 31 2022
#* Author: Elise Rust
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than 
#* the TAs, professor, textbook, and teammates.

#### Import packages
import csv
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import scipy.spatial
from scipy.stats import mode


class ClassifierAlgorithm:
    ''' 
    Base class to define the attributes and member methods for ClassifierAlgorithm
    
    This is an abstract class --> Subclasses include simplekNNClassifier & kdTreeKNNClassifier
    Subclasses define
    '''

    def __init__(self, labels="", predictors=None):
        ''' Initialize object of class ClassifierAlgorithm
        
        labels: STR, optional
            Key value for column containing the labels
        predictors: LIST, optional
            Key values of all columns to be used as predictors.
            If no value is given, then all variables will be used.
        '''

        self.labels = labels
        self.predictors=predictors
        print("Object of class ClassifierAlgorithm has been instantiated.")

    def train(self, trainingData):
        ''' Train ClassifierAlgorithm using training data'''
        print(f'{trainingData} has been trained for the classifier algorithm.')

    def test(self, testData, k):
        ''' Test ClassifierAlgorithm on test data'''
        print(f'The classifier algorithm has been tested with {testData}.')


class simpleKNNClassifier(ClassifierAlgorithm):
    ''' Subclass of base class ClassifierAlgorithm for a Simple KNN Classifier '''

    def __init__(self, labels=0, predictors=None):
        ''' Initialize object of class simpleKNNClassifier'''
        super().__init__()
        print("Object of subclass simpleKNNClassifier has been instantiated.")
    
    def to_tfidf(self, dataset, column):
        ''' Turn text into TFIDF Vectorizer format for KNN Classification'''
        data = []
        for element in dataset:
            data.append(element[column]) # Get all clean text into a list
        cv = CountVectorizer() # Initialize count vectorizer
        word_count_vector = cv.fit_transform(data).toarray() # Fit to data
        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
        tfidf_words = tfidf_transformer.fit_transform(word_count_vector).toarray()
        return tfidf_words

    def train(self, trainingData, trueLabels):
        ''' Train simpleKNNClassifier using training data
        
        Store the data and labels member attributes
        '''
        self.trainingData = trainingData
        self.trueLabels = trueLabels
    
    def test(self, testData, k):
        ''' Test simple KNNClassifier on test data
        
        Find the k closest training samples and return the mode of the k labels
        associated from the k closest training samples.

        testData: ARRAY of data to be used to calculate KNN
        k: INT 

        Returns: Predicted labels stored in a member attribute
        '''
        # For each element of data
        predicted_labels = [] # Initialize list of predicted labels
        for i in range(len(testData)):
            # Find k closest samples using Euclidean Distance
            distances = [] # Store distances of of all other points
            for j in range(len(self.trainingData)):
                d = np.linalg.norm(self.trainingData[j] - testData[i])
                label = self.trueLabels[j]
                distances.append([float(d), int(label)]) # Append individual distance
            
            distances = np.array(distances)
            dist_sort = np.sort(distances)[:k] # Sort and keep k 
            
            # Calculate mode of the k labels
            lab = mode(dist_sort) 
            lab = lab.mode[0][1]
            predicted_labels.append(lab) # Append to predicted labels list

        return predicted_labels


class Experiment:
    ''' Base class to define the attributes and member methods for class Experiment.

    Splits data into training and desting, implements k-fold cross validation and
    evaluates the accuracy of the Classifier Algorithm by outputing
    relevant accuracy metrics and evaluation plots.
    '''

    def __init__(self, data, labels, classifiers):
        ''' Initialize object of class Experiment
        
        Parameters:

        data: DataSet
            DataSet to run experiment on.
        labels: list
            True labels
        classifiers: list
            List of ClassifierAlgorithms to experiment with.
        '''
        self.data = data
        self.labels = labels
        self.classifiers = classifiers
        print("Object of class Experiment has been instantiated.")

    def runCrossVal(self, k):
        ''' Resampling tool to evaluate the ClassifierAlgorithm. Using K-fold cross-validation
        to compare the effectiveness of different classifiers.
        
        k : Number of groups to split data sample into
        '''
        # Split dataset into k folds
        folds = np.array_split(self.data, k)
        folds_labels = np.array_split(self.labels, k)

        # Initialize output matrix numSamples x numClassifiers
        output = np.zeros((len(folds), len(self.classifiers)), dtype=np.ndarray)

        # Iterate over folds:
        ## Use fold i as test set, and the other k-1 as training
        for i in range(k):
            test_data = folds[i]
            train_data = folds[:i]+folds[i+1:]

            # Loop through classifiers
            for j, classifier in enumerate(self.classifiers):
                # Train data
                classifier.train(self.data, self.labels)
                
                # Test all classifiers
                y_predict = np.array(classifier.test(self.data, 10)) 

                # Append to matrix 'output'
                output[i, j] = y_predict
        
        # Take average of all folds for each label
        output = np.mean(output, axis=0)

        return output


    def score(self, trueLabels, predictedLabels):
        ''' Returns the accuracy score of each classifier (# of correct predictions/total samples)'''
        score = np.mean(trueLabels != predictedLabels)
        return score

    def confusionMatrix(self, trueLabels, predictedLabels):
        ''' Generates a confusion matrix for a given classifier, illustrating how many predictions
        were correct by class.
        '''
        # Get the unique classes
        classes = np.unique(trueLabels)

        # Create an empty confusion matrix of the correct dimensions
        cMatrix = np.zeros((len(classes), len(classes)))

        # Fill confusion matrix
        ## Count number of instances of each class correctly and incorrectly predicted
        for i in range(len(classes)):
            for j in range(len(classes)):
                cMatrix[i,j] = np.sum((trueLabels==classes[i]) & (predictedLabels==classes[j]))

        return cMatrix
