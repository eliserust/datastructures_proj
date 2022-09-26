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


class DataSet:
    ''' 
    Base class to define the attributes and member methods for DataSets
    
    This is an abstract class --> Subclasses include TimeSeriesDataSet, 
        TextDataSet, QuantDataSet, QualDataSet.
    Subclasses define type of DataSet, instantiate the DataSet class, and
        override attributes and methods for specific data types.
    '''
    def __init__(self, filename):
        ''' Initialize object of class Data using __readFromCSV and __load methods '''
        self.filename = filename
        print("Object of superclass DataSet has been initialized")
    
    def __readFromCSV(self, filename):
        ''' Load dataset from external source as object from class DataSet'''
        print("Data of superclass DataSet has been read in from CSV")

    def __load(self,filename):
        ''' Load dataset from external source as object from class DataSet'''
        print("Data of superclass DataSet has been loaded.")

    def clean(self):
        ''' Clean the dataset to appropriate standards'''
        print("Data of superclass DataSet has been cleaned")

    def explore(self):
        ''' Conduct basic exploratory analysis on dataset'''
        print("Data of superclass DataSet has been explored.")


class ClassifierAlgorithm:
    ''' 
    Base class to define the attributes and member methods for ClassifierAlgorithm
    
    This is an abstract class --> Subclasses include simplekNNClassifier & kdTreeKNNClassifier
    Subclasses define
    '''

    def __init__(self):
        ''' Initialize object of class ClassifierAlgorithm'''
        print("Object of superclass ClassifierAlgorithm has been initialized.")

    def train(self, data):
        ''' Train ClassifierAlgorithm using training data'''
        print(f'{data} has been trained for the classifier algorithm.')

    def test(self, data):
        ''' Test ClassifierAlgorithm on test data'''
        print(f'The classifier algorithm has been tested with {data}.')


class Experiment:
    ''' Base class to define the attributes and member methods for class Experiment.

    Splits data into training and desting, implements k-fold cross validation and
    evaluates the accuracy of the Classifier Algorithm by outputing
    relevant accuracy metrics and evaluation plots.
    '''

    def __init__(self, data, classifiers):
        ''' Initialize object of class Experiment
        
        Parameters:

        data: DataSet
            DataSet to run experiment on.
        classifiers: list
            List of ClassifierAlgorithms to experiment with.
        '''
        self.data = data
        self.classifiers = classifiers
        print("Object of class Experiment has been instantiated.")

    def runCrossVal(self, k):
        ''' Resampling tool to evaluate the ClassifierAlgorithm. Using K-fold cross-validation
        to compare the effectiveness of different classifiers.
        
        k : Number of groups to split data sample into
        '''
        print("Cross Validation has been run")

    def score(self):
        ''' Returns the accuracy score of each classifier (# of correct predictions/total samples)'''
        print("The accuracy score has been generated.")

    def confusionMatrix(self):
        ''' Generates a confusion matrix for a given classifier, illustrating how many predictions
        were correct by class.
        '''
        print("The confusion matrix has been generated.")


class TimeSeriesDataSet(DataSet):
    ''' Subclass of base class DataSet for data of type Time Series '''

    def __init__(self, filename):
        ''' 
        Instantiate object of class TimeSeriesDataSet.
        Inherited from base class DataSet
        '''
        super().__init__(filename)
        print("Object of subclass TimeSeriesDataSet has been instantiated")
    
    def __readFromCSV(self, filename):
        ''' 
        Read in TimeSeriesDataSet from existing CSV.
        Override existing DataSet method
        '''
        return super().__readFromCSV(filename)
    
    def __load(self, filename):
        ''' Load dataset from external source as object from class TimeSeriesDataSet
        Override existing DataSet method'''
        return super().__load(filename)
    
    def clean(self):
        ''' 
        Clean the dataset to appropriate standards
        Override existing DataSet method'''
        print("Data of subclass TimeSeriesDataSet has been cleaned.")
    
    def explore(self):
        ''' Conduct basic exploratory analysis on dataset
        Override existing DataSet method
        '''
        print("Data of subclass TimeSeriesDataSet has been explored.")


class TextDataSet(DataSet):
    ''' Subclass of base class DataSet for data of type Text '''

    def __init__(self, delim, new_line, *args):
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
        super().__init__(*args)
        self.delim = delim
        self.new_line = new_line
        print("Object of subclass TextDataSet has been instantiated.")
        
    
    def __readFromTXT(self, filename):
        ''' 
        Read in TextDataSet from existing .txt file.
        Override existing DataSet method
        '''
        return super().__readFromCSV(filename)
    
    def __load(self, filename):
        ''' Load dataset from external source as object from class TextDataSet
        Override existing DataSet method'''
        return super().__load(filename)
    
    def clean(self):
        ''' 
        Clean the dataset to appropriate standards
        Override existing DataSet method'''
        print("Data of subclass TextDataSet has been cleaned.")
    
    def explore(self):
        ''' Conduct basic exploratory analysis on dataset
        Override existing DataSet method
        '''
        print("Data of subclass TextDataSet has been cleaned.")


class QuantDataSet(DataSet):
    ''' Subclass of base class DataSet for Quantitative Data types '''

    def __init__(self, filename):
        ''' 
        Instantiate object of class QuantDataSet.
        Inherited from base class DataSet
        '''
        super().__init__(filename)
        print("Object of subclass QuantDataSet has been instantiated")
    
    def __readFromCSV(self, filename):
        ''' 
        Read in QuantDataSet from existing .csv file.
        Override existing DataSet method
        '''
        return super().__readFromCSV(filename)
    
    def __load(self, filename):
        ''' Load dataset from external source as object from class QuantDataSet
        Override existing DataSet method'''
        return super().__load(filename)
    
    def clean(self):
        ''' 
        Clean the dataset to appropriate standards
        Override existing DataSet method'''
        print("Data of subclass QuantDataSet has been cleaned.")
    
    def explore(self):
        ''' Conduct basic exploratory analysis on dataset
        Override existing DataSet method
        '''
        print("Data of subclass QuantDataSet has been cleaned.")


class QualDataSet(DataSet):
    ''' Subclass of base class DataSet for Qualitative Data types '''

    def __init__(self, filename):
        ''' 
        Instantiate object of class QualDataSet.
        Inherited from base class DataSet
        '''
        super().__init__(filename)
        print("Object of subclass QualDataSet has been instantiated")
    
    def __readFromCSV(self, filename):
        ''' 
        Read in QualDataSet from existing .csv file.
        Override existing DataSet method
        '''
        return super().__readFromCSV(filename)
    
    def __load(self, filename):
        ''' Load dataset from external source as object from class QualDataSet
        Override existing DataSet method'''
        return super().__load(filename)
    
    def clean(self):
        ''' 
        Clean the dataset to appropriate standards
        Override existing DataSet method'''
        print("Data of subclass QualDataSet has been cleaned.")
    
    def explore(self):
        ''' Conduct basic exploratory analysis on dataset
        Override existing DataSet method
        '''
        print("Data of subclass QualDataSet has been cleaned.")


class simpleKNNClassifier(ClassifierAlgorithm):
    ''' Subclass of base class ClassifierAlgorithm for a Simple KNN Classifier '''

    def __init__(self):
        ''' Initialize object of class simpleKNNClassifier'''
        super().__init__()
        print("Object of subclass simpleKNNClassifier has been instantiated.")
    
    def train(self, data):
        ''' Train simpleKNNClassifier using training data'''
        return super().train(data)
    
    def test(self, data):
        ''' Test simple KNNClassifier on test data'''
        return super().test(data)


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

