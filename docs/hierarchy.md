``` mermaid
classDiagram
    DataSet <|-- TimeSeriesDataSet
    DataSet <|-- TextDataSet
    DataSet <|-- QuantDataSet
    DataSet <|-- QualDataSet
    
    class DataSet
    DataSet: +String filename
    DataSet: +__init__(self, filename)
    DataSet: +__readFromCSV(self, filename)
    DataSet: +__load__(self, filename)
    DataSet: +clean(self)
    DataSet: +explore(self)
    Address "1" <-- "0..1" DataSet:lives at

    class TextDataSet{
        +String delim
        +String new_line
        -__readFromCSV(self, filename)
        +__readFromTXT(self,filename)
    }

```