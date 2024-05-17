import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
# 准备数据集
batch_size = 64
datalist=[]

def get_file_name_1(file_path):
    file_names = os.listdir(file_path)
    return file_names
file_path = 'C:/Users/15920/Desktop/jicheng/data/DataExcel'
file_names = get_file_name_1(file_path)
# print(file_names)
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


###############################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        if in_features != out_features:
            self.adjust_dim = nn.Linear(in_features, out_features)
        else:
            self.adjust_dim = None

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        if self.adjust_dim:
            residual = self.adjust_dim(residual)
        out += residual
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(404, 256)
        self.l2 = ResidualBlock(256, 256)
        self.l3 = ResidualBlock(256, 128)
        self.l4 = ResidualBlock(128, 128)
        self.l5 = ResidualBlock(128, 64)
        self.l6 = ResidualBlock(64, 64)
        self.l7 = ResidualBlock(64, 32)
        self.l8 = ResidualBlock(32, 32)
        self.l9 = ResidualBlock(32, 16)
        self.l10 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 404)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = F.relu(x)
        return x
#####################################################################

# Define the number of folds
num_folds = 10
# Initialize StratifiedKFold
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
# Define lists to store accuracies for each fold
accuracies = []

# Iterate over the folds
for fold, (train_indices, test_indices) in enumerate(kfold.split(X, y)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Split data into train and test sets for this fold
    X_train_fold, X_test_fold = X[train_indices], X[test_indices]
    y_train_fold, y_test_fold = y[train_indices], y[test_indices]

    # Convert data to PyTorch tensors
    X_train_fold_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
    X_test_fold_tensor = torch.tensor(X_test_fold, dtype=torch.float32)
    y_train_fold_tensor = torch.tensor(y_train_fold, dtype=torch.long)
    y_test_fold_tensor = torch.tensor(y_test_fold, dtype=torch.long)

    # Create TensorDataset
    train_dataset_fold = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
    test_dataset_fold = TensorDataset(X_test_fold_tensor, y_test_fold_tensor)

    # Create DataLoader
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
    test_loader_fold = DataLoader(test_dataset_fold, batch_size=batch_size)

    # Initialize the model
    net = Net()

    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Train the model
    for epoch in range(500):
        for inputs, labels in train_loader_fold:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set for this fold
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader_fold:
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        fold_accuracy = correct / total
        accuracies.append(fold_accuracy)
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy}")
        with open(f'test{fold + 1}.pickle', 'wb') as f:
            pickle.dump(net, f)  # 将训练好的模型clf存储在变量f中，且保存到本地

# Calculate and print average accuracy across all folds
avg_accuracy = np.mean(accuracies)
print(f"Average Accuracy: {avg_accuracy}")
