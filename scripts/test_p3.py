#=====================================================================
# Testing script for Deliverable 3: Classifier Class and Experiment Class
#=====================================================================
import os
os.chdir('..')
print(os.getcwd())

# Import classes
from data_classes import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet)
from classifiers import (ClassifierAlgorithm, simpleKNNClassifier, Experiment)

# Approved libraries
import csv
import nltk
import numpy as np
import matplotlib.pyplot as plt

# 1) Load in dataset to use for classifier testing
# Test loading data
print("Testing the TextDataSet class first. Initialize object of class TextDataSet.")
print("Use 'data/text_data.csv' as the data here when prompted.")
text_data = TextDataSet("data/text_data.csv")
print("Print member attributes of class TextDataSet")
print("TextDataSet.filename:", text_data.filename)
#print("TextDataset content is: ", text_data.data)
# Test cleaning and explore methods
text_data.data = text_data.clean()
print("Calling the clean method: ", text_data.data)
print("\n\n")


# 2) Testing Classifier Class 
# Test initializing class
print("Testing the Classifier class first. Initialize object of class Classifier.")
classifier= ClassifierAlgorithm(labels="stars", predictors=["clean_text"])
print("Print member attributes of class Classifier")
print("Classifier.filename:", classifier.labels)
print("Classifier content is: ", classifier.predictors)
# Test the train() method
classifier.train(text_data.data)
# Test the test() method
classifier.test(text_data.data, 10)
print("\n\n")


# 3) Testing simpleKNNClassifier Class 
# Test initializing class
print("Testing the simpleKNNClassifier class first.")
knnclassifier= simpleKNNClassifier()
print("Print member attributes of class Classifier")
print("Classifier labels:", knnclassifier.labels)
print("Classifier predictors is: ", knnclassifier.predictors)
### TFIDF Vectorize the data
tfidf_text = knnclassifier.to_tfidf(text_data.data, "clean_text")
# Test the train() method
### Runtime on full data is ~2.5 hours so testing on a subset
subset = tfidf_text[:100, ]
print("Calling the train() method on training data: ")
trueLabels = []
for element in text_data.data:
    trueLabels.append(element["stars"])
knnclassifier.train(subset, trueLabels[:100])
print("The train dataset is: ", knnclassifier.trainingData)
print("The trueLabels are: ", knnclassifier.trueLabels)
# Test the test() method
print("Calling the test() method on testing data: ")
y_predict = knnclassifier.test(subset, 10)
print("The predicted labels for the first 100 reviews are: ", y_predict)
print("\n\n")

# 4) Testing Experiment Class
# Test initializing class
print("Testing the Experiment class. Initialize object of class Experiment.")
c_list = [knnclassifier]
exp = Experiment(subset, trueLabels[:100], c_list)
print("Print member attributes of class Experiment")
print("Experiment data: ", exp.data)
print("Experiment classifiers: ", exp.classifiers)
print("Experiment labels are: ", exp.labels)
# Run cross validation
output = exp.runCrossVal(10)
print("Running k fold cross validation: ", exp.runCrossVal(10))
# Output the accuracy score
print("The accuracy of this model is: ", exp.score(output[0], trueLabels[:100]))
# Print a confusion matrix
print("The confusion matrix for this experiment is: ", exp.confusionMatrix(trueLabels[:100], list(output[0])))
print("\n\n")