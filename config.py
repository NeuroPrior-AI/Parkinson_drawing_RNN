import torch
import pandas as pd

num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr
device = torch.device("cuda")
input_size = 2 #number of features
hidden_size = 32 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
batch_size = 64
num_classes = 1 #number of output classes