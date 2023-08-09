import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import CustomLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### Load and process dataset ####
with open("../data/dataset_test.pkl", "rb") as f:
    data = pickle.load(f)

# Data normalization
data[:,:-3] = (1/255)*data[:,:-3]
data[:,-3] = (1/26)*data[:,-3]
data[:,-2] = (1/24)*data[:,-2]

# Split training and testing dataset
(n_samples, sample_dims) = data.shape
train_size = 3*n_samples//4
train_data = data[:train_size,:].copy()
test_data = data[train_size:,:].copy()


#### Defining parameters and model ####
num_epochs = 10

model = nn.Sequential(
    nn.Linear(721*2, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    nn.Tanh()
)
model.to(device)

loss_function = CustomLoss()
loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

inputs = torch.tensor(train_data[:,:-3], device=device, dtype=torch.float32)
targets = torch.tensor(train_data[:,-3:], device=device, dtype=torch.float32)

#### Training ####
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = loss_function(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model)
