# Analog-Circuits-Fault-Diagnosis-Based-on-Machine-Learning
Dataset and code
data folder: save the txt file of the original S parameters and the data file saved to EXCEL
            (1) XX_x folder: Folder to store S parameters in .txt format. In the abbreviated term XX_x, XX represents the affected component and x represents the hard fault type (x=s represents a short circuit, x=o represents an open circuit).
            (2)DataExcel: Folder to store S parameters in .xlsx format.Each excel file contains 404 rows and 100 columns. Among them, 404 represents 101 sampling points, each sampling point has 4 S-parameters, and the S-parameters are arranged vertically. 
               The 100 columns represent 100 groups of samples for each hard fault.
txt2excel.py: The S parameters obtained by the test are saved in a txt file, and the txt file is saved as an EXCEL file in the above format through this file.
PCA.py: Read EXCEL file data, reduce the dimensionality of the data and display graphics
SVM.py: SVM multi-classifier, evaluate model performance by modifying model parameters and PCA dimensionality reduction data
DT.py: Decision tree multi-classifier, evaluate model performance by modifying model parameters and PCA dimensionality reduction data
