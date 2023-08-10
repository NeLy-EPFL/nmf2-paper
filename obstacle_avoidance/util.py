import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):
        in_coord = inputs[:,:2]
        in_m = inputs[:,2]
        targ_coord = targets[:,:2]
        targ_m = targets[:,2]

        mloss = 10*F.mse_loss(torch.mul(targ_m,in_coord.T).T, targ_coord) 
        bloss = F.binary_cross_entropy(in_m, targ_m)
        # print(mloss)
        # print(bloss)

        return mloss+bloss
    

class CustomFeaturesDataset(Dataset):
    def __init__(self, data, n_features=3):
        self.observations = torch.tensor(data[:,:-n_features], device=device, dtype=torch.float32)
        self.features = torch.tensor(data[:,-n_features:], device=device, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return self.observations[idx,:], self.features[idx,:]
