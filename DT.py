from sklearn import tree
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

datalist=[]
pcadatalist=[]
svm_y=[]
def get_file_name_1(file_path):
    file_names = os.listdir(file_path)
    return file_names
file_path = 'C:/Users/15920/Desktop/jicheng/test/data_excel'
file_names = get_file_name_1(file_path)
print(file_names)
for i in file_names:
    m_data = pd.read_excel(file_path+'/'+i,header=None)
    datalist.append(m_data)
alldata=np.concatenate((datalist[0],datalist[1]),axis=1)
for i in range(2,10):
    alldata=np.concatenate((alldata,datalist[i]),axis=1)
alldata=alldata.T
for weishu in range(1,11):
    pca2 = PCA(n_components=weishu)
    pca2.fit(alldata)
    X = pca2.transform(alldata)
    y=[]
    for i in range(10):
        for j in range(40):
            y.append(i)
    y=np.array(y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random",
                                      max_depth=3, min_samples_leaf=10, min_samples_split=10 )
    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test) #返回预测的准确度
    print(score)