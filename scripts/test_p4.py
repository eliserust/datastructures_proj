#=====================================================================
# Testing script for Deliverable 4: Multiple Inheritance
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

# 1) Load in datasets to use for classifier testing
## Text Data
print("Testing the TextDataSet class first. Initialize object of class TextDataSet.")
print("Use 'data/text_data.csv' as the data here when prompted.")
text_data = TextDataSet("data/text_data.csv")
# Test cleaning and explore methods
text_data.data = text_data.clean()
print("Calling the clean method: ", text_data.data)
print("\n\n")
print("\n\n")


# 2) Testing simpleKNNClassifier Class --> WITH NEW METHOD predict_prob()
# Initializing class
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
# Train
knnclassifier.train(tfidf_text, trueLabels)
# knnclassifier.train(subset, trueLabels[:100])
print("The train dataset is: ", knnclassifier.trainingData)
print("The trueLabels are: ", knnclassifier.trueLabels)
# Test the test() method
print("Calling the test() method on testing data: ")
y_predict = knnclassifier.test(subset, 30)
print("The predicted labels for the first 100 reviews are: ", y_predict)
# Test the predict
prob_vector = knnclassifier.predict_probabilities(subset,30)
print("The predicted probabilities are: ", prob_vector)
print("\n\n")


# 4) Testing Experiment Class --> with ROC() method
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
## Create a ROC Curve
thresholds = np.array(list(np.random.uniform(low=0.0, high=1.0, size=(100,))))
roc_data = exp.ROC(thresholds, exp.labels, 10)
print("The ROC Curve for the data (One vs. Rest for Multiclass Labels): ")
exp.plot_ROC(roc_data)
