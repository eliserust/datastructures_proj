#=====================================================================
# Testing script for Deliverable 1: Source Code Framework
#=====================================================================

# 1) Testing DataSet Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
# Import DataSet Class and subclasses
from class_framework import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet)

def DataSetTests():
    data = DataSet("quant_data.csv")
    print("Check ABC member attributes...")
    print("DataSet.filename:", data.filename)
    print("==============================================================")
    print("Check class member methods...\n")
    print("Instantiating the DataSet class again both the load()\
and the readFromCSV() methods run. Both are private member methods")
    data = DataSet("sample_file.ext")
    print("Now call DataSet.clean()...")
    data.clean()
    print("===========================================================")
    print("Now call DataSet.explore()...")
    data.explore()
    print("\n\n")

def QuantDataSetTests():
    data = QuantDataSet("quant_data.ext")
    print("Check inheritence ...")
    print("QuantDataSet.filename:",data.filename)
    print("===========================================================")
    print("Check member methods...\n")
    print("Check that clean and explore methods have been overriden...\n")
    print("QuantDataSet.clean():")
    data.clean()
    print("QuantDataSet.explore():")
    data.explore()
    print("\n\n")
    
def QualDataSetTests():
    data = QualDataSet("qual_data.ext")
    print("Check inheritence ...")
    print("QualDataSet.filename:",data.filename)
    print("===========================================================")
    print("Check QualDataSet member attributes...")
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("QuanlDataSet.clean():")
    data.clean()
    print("QuanlDataSet.explore():")
    data.explore()
    print("\n\n")
    
def TextDataSetTests():
    data = TextDataSet(",", "/n", "text_data.ext")
    print("Check inheritence ...")
    print("TextDataSet.filename:",data.filename)
    print("===========================================================")
    print("Check TextDataSet member attributes...")
    print("TextDataSet delimiter:", data.delim)
    print("TextDataSet new_line character:", data.new_line)
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("TextDataSet.clean():")
    data.clean()
    print("TextDataSet.explore():")
    data.explore()
    print("\n\n")
    
def TimeSeriesDataSetTests():
    data = TimeSeriesDataSet("time_series.ext")
    print("Check inheritence ...")
    print("TimeSeriesDataSet.filename:",data.filename)
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("TimeSeriesDataSet.clean():")
    data.clean()
    print("TimeSeriesDataSet.explore():")
    data.explore()
    print("\n\n")

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from class_framework import (ClassifierAlgorithm,
                                simpleKNNClassifier,kdTreeKNNClassifier)
                                        
def ClassifierAlgorithmTests():
    print("ClassifierAlgorithm Instantiation....")
    classifier = ClassifierAlgorithm()
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("ClassifierAlgorithm.train(data):")
    print(classifier.train(x))
    print("ClassifierAlgorithm.test(data):")
    print(classifier.test(x))
    print("===========================================================\n\n")

def simpleKNNClassifierTests():
    print("simpleKNNClassifier Instantiation....")
    classifier = simpleKNNClassifier()
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("simpleKNNClassifier.train(data):")
    print(classifier.train(x))
    print("simpleKNNClassifier.test(data):")
    print(classifier.test(x))
    print("===========================================================\n\n")

def kdTreeKNNClassifierTests():
    print("kdTreeKNNClassifier Instantiation....")
    classifier = kdTreeKNNClassifier()
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("kdTreeKNNClassifier.train(data):")
    print(classifier.train(x))
    print("kdTreeKNNClassifier.test(data):")
    print(classifier.test(x))
    print("===========================================================\n\n")
    

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from class_framework import Experiment

def ExperimentTests():
    print("Experiment class instantiation (Experiment(classifier,data))...")
    experiment = Experiment("data","classifier")
    print("==============================================================")
    print("Check member attributes...")
    print("Experiment.classifiers:",experiment.classifiers)
    print("Experiment.data:",experiment.data)
    print("==============================================================")
    print("Check class member methods...\n")
    print("Experiment.score():")
    experiment.score()
    print("Experiment.runCrossVal(numFolds):")
    experiment.runCrossVal("numFolds")
    print("==============================================================")
    print("Experiment.confusionMatrix())")
    experiment.confusionMatrix()
    

    
def main():
    DataSetTests()
    QuantDataSetTests()
    QualDataSetTests()
    TextDataSetTests()
    TimeSeriesDataSetTests()
    ClassifierAlgorithmTests()
    simpleKNNClassifierTests()
    kdTreeKNNClassifierTests()
    ExperimentTests()
    
if __name__=="__main__":
    main()
