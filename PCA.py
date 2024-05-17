import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy
import warnings

warnings.filterwarnings("ignore")
#存储所有降维后的数据
datalist=[]
labellist=[]
def get_file_name_1(file_path):
    file_names = os.listdir(file_path)
    return file_names

file_path = 'C:/Users/15920/Desktop/jicheng/data/DataExcel'
file_names = get_file_name_1(file_path)
print(file_names)

for i in file_names:
    m_data = pd.read_excel(file_path+'/'+i,header=None)
    # m_data=np.array(m_data)
    datalist.append(m_data)

alldata=np.concatenate((datalist[0],datalist[1]),axis=1)
for i in range(2,10):
    alldata=np.concatenate((alldata,datalist[i]),axis=1)
alldata=alldata.T
plt.figure(figsize=(8, 6))
ax=plt.subplot(projection = '3d')
ax.set_title('3-D plot')
pca2 = PCA(n_components=3)
pca2.fit(alldata)
x_3d = pca2.transform(alldata)
print(np.shape(x_3d))
colors=['black','cyan','green','magenta','red','yellow','brown','aqua','skyblue','lime']
for i in range(0,4000,400):
    ax.scatter(x_3d[i:i+400, 0], x_3d[i:i+400, 1],x_3d[i:i+400, 2],color=colors[i//400])
ax.set_xlabel('1st Principal Component')
ax.set_ylabel('2nd Principal Component')
ax.set_zlabel('3nd Principal Component')
#plt.savefig('C:/Users/15920/Desktop/PCAresultT.svg',format='svg')
plt.show()




