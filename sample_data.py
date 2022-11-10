import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# from torch.utils.data import
import numpy as np
import pandas as pd
from sklearn import preprocessing

from .config import *

df = pd.DataFrame(data=preprocessing.StandardScaler().fit_transform(np.random.randint(0, 10, size=(1000, 200))))

y = pd.Series(np.random.randint(0, 2, 500))
df = torch.tensor(df.values, dtype = float)
y = torch.tensor(y.values).reshape(500, 1)
df = df.reshape(500, 200, 2)

dataset = []

for i in range(len(df)):
    dataset.append([df[i], y[i]])
trainloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

df2 = pd.DataFrame(data=preprocessing.StandardScaler().fit_transform(np.random.randint(0, 10, size=(600, 200))))

y2 = pd.Series(np.random.randint(0, 2, 300))
df2 = torch.tensor(df2.values, dtype = float)
y2 = torch.tensor(y2.values).reshape(300, 1)
df2 = df2.reshape(300, 200, 2)
dataset2 = []
for i in range(len(df2)):
    dataset2.append([df2[i], y2[i]])
testloader = torch.utils.data.DataLoader(dataset2, shuffle=True, batch_size=batch_size)