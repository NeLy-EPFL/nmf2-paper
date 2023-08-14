import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from util import CustomLoss, CustomFeaturesDataset

import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = Path("../features_model/{date:%d-%m_%H-%M}".format(date=datetime.datetime.now()))
save_dir.mkdir(parents=True, exist_ok=True)

#### Load and process dataset ####
with open("../data/dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Split training and testing dataset
(n_samples, sample_dims) = data.shape
batch_size = 50

dataset = CustomFeaturesDataset(data, 3)
train, valid, test = random_split(dataset,[3/4, 1/8, 1/8])

# Save datasets
with open(save_dir / "train_dataset.pkl", "wb") as f:
    pickle.dump(train, f)
with open(save_dir / "valid_dataset.pkl", "wb") as f:
    pickle.dump(valid, f)
with open(save_dir / "test_dataset.pkl", "wb") as f:
    pickle.dump(test, f)

# Create dataloaders
trainloader = DataLoader(train, batch_size=batch_size)
validloader = DataLoader(valid, batch_size=batch_size)


#### Defining parameters and model ####
num_epochs = 1000

model = nn.Sequential(
    nn.Linear(721*2, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
    nn.Sigmoid()
)
model.to(device)

loss_function = CustomLoss()
loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

#### Training ####
train_loss = []
valid_loss = []

for epoch in trange(num_epochs):
    #Train set
    epoch_loss = 0
    for batch_in, batch_feats in trainloader:
        # Forward pass 
        outputs = model(batch_in)
        loss = loss_function(outputs, batch_feats)
        epoch_loss += loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss.append(epoch_loss)

    # Validation set
    epoch_loss = 0
    for batch_in, batch_feats in validloader:
        outputs = model(batch_in)
        loss = loss_function(outputs, batch_feats)
        epoch_loss += loss
    valid_loss.append(epoch_loss)

#### Saving and plotting results ####
torch.save(model, save_dir / f'model_{num_epochs}.pth')

with open(save_dir / "train_loss.pkl", "wb") as f:
    pickle.dump(train_loss, f)
with open(save_dir / "valid_loss.pkl", "wb") as f:
    pickle.dump(valid_loss, f)

plt.plot(range(num_epochs), train_loss)
plt.plot(range(num_epochs), valid_loss)
plt.show()
