Data folder: save the txt file of the original S parameters and the data file saved to EXCEL.
  The folder naming format is XXY, XX represents the affected component and Y represents the hard fault type (Y=S represents a short circuit, Y=O represents an open circuit).
  DataExcel: Folder to store S parameters in .xlsx format.Each excel file contains 404 rows and 360 columns. Among them, 404 represents 101 sampling points, each sampling point has 4 S-parameters, and the S-parameters are arranged vertically.
  val:Validation set data. The save format is as described above.
txt2excel.py: Save S-parameter data saved in .txt format to excel
PCA.py: Reduce the dimensionality of the experimental data to 3 dimensions to verify the feasibility of the experiment.
SVM.py : Using SVM for multi-classification
RandomForest.py: Using Random Forest for multi-classification
10fold FCNet.py: A neural network with a residual module is used for classification, and a 10-fold cross-validation method is used to measure its accuracy.
