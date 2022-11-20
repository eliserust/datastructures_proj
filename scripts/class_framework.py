#* class_framework.py
#*
#* ANLY 555 Fall 2022
#* Project Deliverable #1
#*
#* Due on Sunday Sept. 25 2022
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
from cmath import isnan
import wordcloud
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import scipy.spatial
from scipy.stats import mode

class DataSet:
    ''' 
    Base class to define the attributes and member methods for DataSets
    
    This is an abstract class --> Subclasses include TimeSeriesDataSet, 
        TextDataSet, QuantDataSet, QualDataSet.
    Subclasses define type of DataSet, instantiate the DataSet class, and
        override attributes and methods for specific data types.
    We assume that data is loaded as a CSV because that is the most common 
        format. If necessary, other data types will be loaded.
    '''
    def __init__(self, filename):
        ''' 
        Initialize object of class Data using __readFromCSV and __load methods 
        Prompt users to enter the name of the file.
        '''
        filename = input('Enter the name of the file to read.')
        self.filename = filename # Member attribute filename
        self.data = self.readFromCSV(filename) # Load data via readFromCSV

    
    def readFromCSV(self, filename):
        ''' Load dataset from external source as object from class DataSet'''
        # Open file and read into a list of lists
        with open(filename) as file:
            content = file.readlines()
            
        # Convert list of lists to dictionary
        mydict = {}
        keys = content[0] # Store column names as keys
        keys = keys.split(',') # Split into list of keys
        for row in content[1:]:
            data = row.split(',') # Delimiter is ","
            for i in range(len(keys)):
                key = keys[i]
                if key not in mydict: # Check if key already in dictionary
                    mydict[key] = list() # Save values as lists
                mydict[key].append(data[i]) # Append data
        return mydict

    def load(self,filename):
        ''' Load dataset from external source as object from class DataSet
        Prompt users to enter the name of the file.
        '''
        # Open file and read into a list of lists
        with open(filename) as file: # Open file
            content = file.readlines() # Read each line of the data

        return content

    def clean(self):
        ''' Clean the dataset to appropriate standards'''
        print("Data of superclass DataSet will be cleaned specific to the datatype.")
        print("Please create an object of a subclass of DataSet to clean.")
        

    def explore(self):
        ''' Conduct basic exploratory analysis on dataset'''
        print("Data of superclass DataSet will be explored specific to the datatype.")
        print("Please create an object of a subclass of DataSet to explore.")



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
    
    def ROC(self, thresh, testlabels, partitions=10):
        ''' 
        Produce a ROC plot which contains a ROC curve for each algorithm.
        If there are more than two algorithms, overlay the ROC curves.
        If there are more than two classes, the ROC method will compute
            multiple (one versus rest) curves.

        prob_vector: Vector of probability thresholds for classifying data points
        predicted_labels: outcome of classification prediction
        partitions: Number of times to run comparison
        '''
    
        classes = list(np.unique(np.array(self.labels))) # Get unique classes
        classes = [float(i) for i in classes] # Make sure elements in list are floats

        # Compute ROC data for each class (One vs Rest)
        roc_data = [] # Master list for all classes

        for i in range(len(classes)):
            pos_class = classes[i] # Establish class i as the "positive" class
            neg_class = [] # Establish rest of the classes as "negative class"
            for c in classes:
                if c != i+1:
                    neg_class.append(c)
            neg_class = np.array(neg_class) # Datatype conversion

            #Loop through all the thresholds and calculate true positive and false positive rate for each
            data = []
            for i in range(partitions):
                threshold_vector = np.greater_equal(thresh, i/partitions).astype(int)

                # Calculate true positives and false positives
                y_binary = np.array(testlabels)
                y_binary[y_binary==pos_class] = 1.0
                #y_binary[y_binary!=1.] = 0.0

                true_pos = np.equal(threshold_vector, 1.0) & np.equal(1.0, y_binary)
                true_neg = np.isin(threshold_vector, neg_class) & np.isin(y_binary, neg_class)
                false_pos = np.equal(threshold_vector, pos_class) & np.isin(y_binary, neg_class)
                false_neg = np.isin(threshold_vector, neg_class) & np.equal(pos_class, y_binary)

                # Calculate true positive and false positive rates!
                tpr = true_pos.sum()/(true_pos.sum()+false_neg.sum())
                fpr = false_pos.sum()/(false_pos.sum()+true_neg.sum())
                data.append([fpr, tpr])

            roc_data.append(data)
        
        return roc_data


    def plot_ROC(self, roc_data):
        '''
        Plot a ROC Curve, using OvR data points computed in ROC() method.
        '''

        roc_data = np.nan_to_num(roc_data, 1.0)

        for i in range(len(roc_data)): # Loop through all classes
            data = roc_data[i]
            data.reshape(-1,2)
            plt.scatter(data[:,0], data[:,1], label = "Class"+str(i+1))
            plt.plot([0,1],[0,1]) # X=Y line
            plt.title('ROC Curve (One vs. Rest) From Scratch')
            plt.suptitle('The data classifies restaurant ratings & there are so many 5 star ratings that KNN predicts only 5.0. Hence, why there is no curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
        plt.show()


class QuantDataSet(DataSet):
    ''' Subclass of base class DataSet for Quantitative Data types '''

    def __init__(self, filename):
        ''' 
        Instantiate object of class QuantDataSet.
        Inherited from base class DataSet
        '''
        super().__init__(filename)
    
    def readFromCSV(self, filename):
        ''' 
        Read in QuantDataSet from existing .csv file.
        Override existing DataSet method
        '''
        return super().readFromCSV(filename)
    
    def load(self, filename):
        ''' Load dataset from external source as object from class QuantDataSet
        Override existing DataSet method'''
        return super().load(filename)
    
    def clean(self):
        ''' 
        Clean the dataset to appropriate standards
        Override existing DataSet method'''
        # Fill in missing values with the mean
        # Subset dictionary to remove string columns

        for key in self.data:
            values = self.data[key] # Save column i's data
            try:
                values = [int(x) for x in values] # Convert to ints
                avg_val = sum(values)/len(values) # Calculate mean of column
                for i in values:
                    if isnan(i): # If value is null
                     i = avg_val # Replace with avg.
            except ValueError: # Pass for non-int columns
                pass
        return self.data
    
    def explore(self):
        ''' Conduct basic exploratory analysis on dataset
        Override existing DataSet method
        '''
        
        averages = {} # Initialize avg dict
        totals = {} # Initialize totals dict
        for key in self.data:
            values = self.data[key] # Store column data
            try:
                values = [int(x) for x in values] # Convert to integers
                avg_val = sum(values)/len(values) # Calculate average
                total_val = sum(values) # Calculate total
                averages[key] = avg_val # Save to dict
                totals[key] = total_val # Save to dict
            except ValueError:
                pass
        

        # Plot #1: Plot average value of each column
        lists = sorted(averages.items()) # sorted by key, return a list of tuples  
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.ylabel('Average Value')
        plt.xlabel('Attribute')
        plt.show()

        # Plot #2: Plot total value of each column
        lists = sorted(averages.items()) # sorted by key, return a list of tuples  
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        plt.plot(x, y, 'ro')
        plt.ylabel('Total Value')
        plt.xlabel('Attribute')
        plt.show()


class QualDataSet(DataSet):
    ''' Subclass of base class DataSet for Qualitative Data types '''

    def __init__(self, filename):
        ''' 
        Instantiate object of class QualDataSet.
        Inherited from base class DataSet
        '''
        super().__init__(filename)
    
    def readFromCSV(self, filename):
        ''' 
        Read in QualDataSet from existing .csv file.
        Override existing DataSet method
        '''
        return super().readFromCSV(filename)
    
    def load(self, filename):
        ''' Load dataset from external source as object from class QualDataSet
        Override existing DataSet method'''
        return super().load(filename)
    
    def clean(self):
        ''' 
        Clean the dataset to appropriate standards
        Override existing DataSet method'''
        # Fill in missing values with the mode

        for key in self.data:
            values = self.data[key] # Store column data as value
            try:
                mode_val = max(set(values), key=values.count) # Calculate mode
                for i in values:
                    if i == '': # If value is empty
                        i = mode_val # Replace with mode
            except ValueError:
                pass
        return self.data

    
    def explore(self, key1, key2, key3):
        ''' Conduct basic exploratory analysis on dataset
        Override existing DataSet method
        '''

        fig, axs = plt.subplots(2)
        # Plot #1: Plot scatterplot of key1 vs. key2
        x = self.data[key1] # Store x
        y = self.data[key2] # Store y
        xlabel = x[0] # Save x label
        ylabel = y[0] # Save y label
        x.pop(0) # Remove label value
        y.pop(0) # Remove label value
        axs[0].scatter(x, y, alpha = 0.5) # Generate scatterplot
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)

        # Plot #2: Plot frequency historgram of key 3
        x = self.data[key3]
        xlabel = x[0] # Store x label
        x.pop(0) # remove label value
        axs[1].hist(x) # Generate histogram
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel("Frequency")


class TimeSeriesDataSet(DataSet):
    ''' Subclass of base class DataSet for data of type Time Series '''

    def __init__(self, filename):
        ''' 
        Instantiate object of class TimeSeriesDataSet.
        Inherited from base class DataSet
        '''
        super().__init__(filename)
        print("Object of subclass TimeSeriesDataSet has been instantiated")
    
    def readFromCSV(self, filename):
        ''' 
        Read in TimeSeriesDataSet from existing CSV.
        Override existing DataSet method
        '''
        return super().readFromCSV(filename)
    
    def load(self, filename):
        ''' Load dataset from external source as object from class TimeSeriesDataSet
        Override existing DataSet method'''
        return super().load(filename)
    
    def toTimeObject(self, key, format="%m/%d/%Y"):
        '''
        Take in column and converts to datetime object
        format: datetime format to convert date column to
        '''
        
        l = self.data[key] # Store date column
        # Convert to datetime object
        dates = [datetime.strptime(d, format).strftime(format) for d in l]
        return dates
    

    def clean(self, key, window):
        ''' 
        Clean the dataset to appropriate standards. Override existing DataSet method
        
        Parameters
        ----------
        key: str - Key for date column
        window : int - Window size

        Returns
        -------
        out: ndarray - array the same size as input containing median filtered result
        '''
        values = self.data[key] # Store column data
        values = np.array(values) # Convert to numpy array

        if window%2 == 0: # If window is even
            x = (window-1)//2
        else: # If window is odd
            x = (window-2)//2 
        before = values[:x].astype(float) # Store values up till x
        after = values[x:].astype(float) # Store values after x
        for i in range(len(values)):
            val = values[i].astype(float) # Take value i
            ar = np.append(val, before)
            ar = np.append(ar, after)
            values[i] = np.median(ar) # Replace value at i with median of val, val up to x, and val after x
        
        return values
        
    
    def explore(self, date_key, key2):
        ''' Conduct basic exploratory analysis on dataset
        Override existing DataSet method
        '''
        # Plot #1: Plot scatterplot of key1 vs. key2
        x = self.data[date_key]
        y = self.data[key2]
        fig, axs = plt.subplots(2)
        axs[0].scatter(x,y, alpha=0.5)
        axs[0].set_xlabel("Date")
        axs[0].set_ylabel(key2)



        # Plot #2: Density plot of key 3
        y = self.data[key2]
        y = np.array(y).astype(np.float)
        axs[1].boxplot(y, vert=True)
        axs[1].set_ylabel(key2)

        plt.show()



class TextDataSet(DataSet):
    ''' Subclass of base class DataSet for data of type Text '''

    def __init__(self, delim=',', new_line='/n', *args):
        ''' 
        Instantiate object of class TextDataSet.
        Inherited from base class DataSet

        Parameters:

        filename: str 
            Path to file
        delim: str
            How text gets split into chunks. Default is ",".
        new_line : string, optional  
            How text gets split at new lines. Default is "/n".
        '''
        filename = input('Enter the name of the file to read.')
        self.filename = filename # Member attribute filename
        self.delim = delim
        self.new_line = new_line
        self.data = self.readFromTXT(filename) # Load data via readFrom TXT
        
    
    def readFromTXT(self, filename):
        ''' 
        Read in TextDataSet from existing .txt file.
        Override existing DataSet method
        '''
        # Read in contents
        with open(filename) as f:
            reader = csv.DictReader(f) # Use DictReader from csv package

            data = [] # Initialize data list
            for row in reader:
                data.append(row) # Append each row of data dict to list
        
        return data
        
    
    def load(self, filename):
        ''' Load dataset from external source as object from class TextDataSet
        Override existing DataSet method'''
        return super().load(filename)
    
    
    def clean(self):
        ''' 
        Clean the dataset to appropriate standards
        Override existing DataSet method'''
        
        # Initialize Stemmer
        stemmer = PorterStemmer() # Initialize Stemmer
        stops = set(stopwords.words('english')) # Identify stopwords
        for element in self.data:
            s = element["text"] # Isolate text data
            toks = word_tokenize(s) # Tokenize string
            stemmed = [stemmer.stem(tok.lower()) for tok in toks] # Stem
            toks_nostop = [tok for tok in stemmed if tok not in stops] # Remove stopwords
            # Replace numbers with #
            for tok in toks_nostop:
                if type(tok) == int: # If word is a number
                    tok = '#' # Replace with hash #
            return " ".join(toks_nostop)


    
    def explore(self):
        ''' Conduct basic exploratory analysis on dataset
        Override existing DataSet method
        '''
        # Wordcloud for plot #1: https://www.geeksforgeeks.org/generating-word-cloud-python/
        words = ''
        for element in self.data:
            s = element["text"] # Get text data
            tokens = s.split() # Split into tokens
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower() # Lowercase all words
            words += " ".join(tokens)+" "
    
        # Initialize wordcloud object
        wordcloud = WordCloud(width = 800, height=800,
                                background_color='white',
                                min_font_size=10).generate(words)

        fig, axs = plt.subplots(2) # Initialize plt figure and axes
        axs[0].imshow(wordcloud) # Show wordcloud
        axs[0].axis("off")

        # Plot 2 - Frequency plot
        words = words.split(" ")
        freq = Counter(words).most_common(10) # Identify top 10 most common words
        x = []
        y = []
        for i in freq:
            x.append(i[0])
            y.append(i[1])
        axs[1].bar(x, y, color='pink')
        axs[1].set_xlabel("Words")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("10 Most common words")
        plt.show()


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
    
    def predict_probabilities(self, testData, k):
        '''
        Predict probability vector for each data point classification for ROC curve.

        Parameters:
        testData: ARRAY of data to be used to calculate KNN
        k: INT 

        Returns: Predicted labels
        '''
        classes = list(np.unique(np.array(self.trueLabels))) # Get unique classes
        classes = [float(i) for i in classes] # Make sure elements in list are floats
        
        # For each element of data
        predicted_prob = [] # Initialize list of predicted labels
        for i in range(len(testData)):
            # Find k closest samples using Euclidean Distance
            distances = [] # Store distances of of all other points
            for j in range(len(self.trainingData)):
                d = np.linalg.norm(self.trainingData[j] - testData[i])
                label = self.trueLabels[j]
                distances.append([float(d), int(label)]) # Append individual distance
            
            distances = np.array(distances)
            dist_sort = np.sort(distances)[:k] # Sort and keep k
            labels =  [i[1] for i in dist_sort]
        
            # Calculate percentage close labels that are each category
            pred = []
            for i in classes: # Get probability of all classes
                freq = (labels.count(i))/len(labels)
                pred.append(freq)

            predicted_prob.append(pred) # Append to predicted labels list

        return predicted_prob


class kdTreeKNNClassifier(ClassifierAlgorithm):
    ''' Subclass of base class ClassifierAlgorithm for a Simple KNN Classifier '''

    def __init__(self):
        ''' Initialize object of class kdTreeKNNClassifier'''
        super().__init__()
        print("Object of subclass kdTreeKNNClassifier has been instantiated.")
    
    def train(self, data):
        ''' Train kdTreeKNNClassifier using training data'''
        return super().train(data)
    
    def test(self, data):
        ''' Test simple kdTreeKNNClassifier on test data'''
        return super().test(data)

