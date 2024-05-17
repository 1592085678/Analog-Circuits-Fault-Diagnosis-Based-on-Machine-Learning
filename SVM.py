from sklearn.svm import SVC
import pickle
import numpy as np
import pandas as pd
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

datalist=[]
pcadatalist=[]
svm_y=[]
def get_file_name_1(file_path):
    file_names = os.listdir(file_path)
    return file_names
file_path = 'C:/Users/15920/Desktop/jicheng/data/DataExcel'
file_names = get_file_name_1(file_path)
print(file_names)
for i in file_names:
    m_data = pd.read_excel(file_path+'/'+i,header=None)
    datalist.append(m_data)
alldata=np.concatenate((datalist[0],datalist[1]),axis=1)
for i in range(2,10):
    alldata=np.concatenate((alldata,datalist[i]),axis=1)
alldata=alldata.T
y=[]
for i in range(10):
    for j in range(400):
        y.append(i)
y=np.array(y)

X = alldata
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

for kernel in ['linear','poly','sigmoid','rbf']:
    svr = SVC(kernel = kernel,C=100)
    svr.fit(X_train,y_train)
    print(svr.score(X_test,y_test))
