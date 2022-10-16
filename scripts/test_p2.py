
#=====================================================================
# Testing script for Deliverable 2: Data Set Class
#=====================================================================
import os
os.chdir('..')
print(os.getcwd())

# Import DataSet Class and subclasses
from data_classes import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet)

# Approved libraries
import csv
import nltk
import numpy as np
import matplotlib.pyplot as plt

# 1) Testing DataSet Class 
# Test loading data
print("Testing the DataSet class first. Initialize object of class DataSet.")
print("Use 'data/quant_data.csv' as the data here when prompted.")
base_data = DataSet("data/quant_data.csv")
print("Print member attributes of class DataSet")
print("DataSet.filename:", base_data.filename)
print("Dataset content is: ", base_data.data)
# Test the readFromCSV method
content = base_data.readFromCSV("data/quant_data.csv")
print("The dictionary of data is: ", content)
# Test cleaning and exploring data
print("Calling the clean method: ", base_data.clean())
print("Calling the explore method: ", base_data.explore())
print("\n\n")


# 2) Testing QuantDataSet Class 
# Test loading data
print("Testing the QuantDataSet class first. Initialize object of class QuantDataSet.")
print("Use 'data/quant_data.csv' as the data here when prompted.")
quant_data = QuantDataSet("data/quant_data.csv")
print("Print member attributes of class QuantDataSet")
print("QuantDataSet.filename:", quant_data.filename)
print("QuantDataset content is: ", quant_data.data)
# Test the readFromCSV method
#content = quant_data.readFromCSV("data/quant_data.csv")
#print("The dictionary of data is: ", content)
# Test cleaning and exploring data
print("Calling the clean method: ", quant_data.clean())
print("Calling the explore method: ", quant_data.explore())
print("\n\n")

# 3) Testing QualDataSet Class 
# Test loading data
print("Testing the QualDataSet class first. Initialize object of class QualDataSet.")
print("Use 'data/qual_data.csv' as the data here when prompted.")
qual_data = QualDataSet("data/qual_data.csv")
print("Print member attributes of class QualDataSet")
print("QualDataSet.filename:", qual_data.filename)
print("QualDataset content is: ", qual_data.data)
# Test the readFromCSV method
#content = qual_data.readFromCSV("data/qual_data.csv")
#print("The dictionary of data is: ", content)
# Test cleaning and exploring data
print("Calling the clean method: ", qual_data.clean())
print("Calling the explore method: ", qual_data.explore("Q2", "Q3", "Q5"))
print("\n\n")

# 4) Testing TimeSeriesDataSet Class 
# Test loading data
print("Testing the TimeSeriesDataSet class first. Initialize object of class TimeSeriesDataSet.")
print("Use 'data/timeseries_data.csv' as the data here when prompted.")
time_data = TimeSeriesDataSet("data/timeseries_data.csv")
print("Print member attributes of class TimeSeriesDataSet")
print("TimeSeriesDataSet.filename:", time_data.filename)
print("TimeSeriesDataset content is: ", time_data.data)
# Test Time Conversion Method
print(time_data.toTimeObject("Date", format="%m/%d/%y"))
# Test cleaning and exploring data
print("Calling the clean method: ", time_data.clean("Daily Mean PM2.5 Concentration", 10))
print("Calling the explore method: ", time_data.explore("Date", "Daily Mean PM2.5 Concentration"))
print("\n\n")

# 5) Testing TextDataSet Class 
# Test loading data
print("Testing the TextDataSet class first. Initialize object of class TextDataSet.")
print("Use 'data/text_data.csv' as the data here when prompted.")
text_data = TextDataSet("data/text_data.csv")
print("Print member attributes of class TextDataSet")
print("TextDataSet.filename:", text_data.filename)
print("TextDataset content is: ", text_data.data)
# Test the readFromTXT method
content = text_data.readFromTXT("data/text_data.csv")
print("The dictionary of data is: ", content)
# Test cleaning and exploring data
print("Calling the clean method: ", text_data.clean())
print("Calling the explore method: ", text_data.explore())
print("\n\n")