import torch
import torch.nn as nn
import torch.nn.functional as F



class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):
        in_coord = inputs[:,:2]
        in_m = inputs[:,2]
        targ_coord = targets[:,:2]
        targ_m = targets[:,2]

        mloss = F.mse_loss(torch.mul(targ_m,in_coord.T).T, targ_coord) 
        bloss = F.binary_cross_entropy(in_m, targ_m)

        return mloss+bloss
