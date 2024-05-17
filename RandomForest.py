from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
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

X = alldata
y=[]
for i in range(10):
    for j in range(360):
        y.append(i)
y=np.array(y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
forest = RandomForestClassifier(n_estimators=256,max_depth=10)
forest = forest.fit(X_train, y_train)
score = forest.score(X_test, y_test) #返回预测的准确度
print(score)
with open('forest.pickle', 'wb') as f:
    pickle.dump(forest, f)  # 将训练好的模型clf存储在变量f中，且保存到本地