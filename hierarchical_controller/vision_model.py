import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pytorch_lightning as pl
import torchmetrics

from flygym.util.config import num_ommatidia_per_eye


class VisualFeaturePreprocessor(pl.LightningModule):
    def __init__(
        self,
        *,
        in_channels=1,
        conv_hidden_channels=4,
        conv_out_channels=2,
        linear_hidden_channels=16,
        classification_loss_coef=0.01,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.classification_loss_coef = classification_loss_coef
        
        # Define model layers
        self.conv1 = gnn.GCNConv(in_channels, conv_hidden_channels)
        self.conv2 = gnn.GCNConv(conv_hidden_channels, conv_out_channels)
        mlp_in_dim = 2 * conv_out_channels * num_ommatidia_per_eye
        self.linear1 = nn.Linear(mlp_in_dim, linear_hidden_channels)
        self.linear2 = nn.Linear(linear_hidden_channels, linear_hidden_channels)
        self.linear3 = nn.Linear(linear_hidden_channels, 3)
        
        # Define metrics
        self._f1 = torchmetrics.classification.BinaryF1Score()
        self._r2 = torchmetrics.R2Score()

    def forward(self, left_graph, right_graph, batch_size=1):
        conv_features_li = []
        for graph in [left_graph, right_graph]:
            x = self.conv1(graph.x.view(-1, 1), graph.edge_index)
            x = F.tanh(x)
            x = self.conv2(x, graph.edge_index)
            x = F.tanh(x)
            conv_features_li.append(x.view(batch_size, -1))
        conv_features = torch.concat(conv_features_li, axis=1)
        x = self.linear1(conv_features)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = self.linear3(x)
        pos_pred = x[:, :2]
        mask_pred = F.sigmoid(x[:, 2])
        return pos_pred, mask_pred
    
    def loss(self, pos_pred, mask_pred, pos_label, mask_label):
        detection_loss = F.binary_cross_entropy(mask_pred, mask_label.float())
        mask = (mask_label == 1) & (mask_pred > 0.5)
        pos_loss = F.mse_loss(pos_pred[mask, :], pos_label[mask, :])
        total_loss = pos_loss + self.classification_loss_coef * detection_loss
        return total_loss, pos_loss, detection_loss
    
    def get_metrics(self, pos_pred, mask_pred, pos_label, mask_label):
        mask_pred_bin = mask_pred > 0.5
        detection_f1 = self._f1(mask_pred_bin.int(), mask_label.int())
        mask = mask_label.bool() & mask_pred_bin
        if mask.sum() < 2:
            pos_r2 = torch.tensor(torch.nan)
        else:
            pos_r2 = self._r2(pos_pred[mask, :].flatten(), pos_label[mask, :].flatten())
        return pos_r2, detection_f1
    
    def training_step(self, batch, batch_idx):
        graphs_left = batch["graph_left"]  # this is a minibatch of graphs
        graphs_right = batch["graph_right"]
        pos_labels = batch["position"]
        mask_labels = batch["object_found"]
        batch_size = mask_labels.size(0)
        
        pos_pred, mask_pred = self.forward(graphs_left, graphs_right, batch_size)
        total_loss, pos_loss, detection = self.loss(
            pos_pred, mask_pred, pos_labels, mask_labels
        )
        
        self.log("train_total_loss", total_loss, batch_size=batch_size)
        self.log("train_pos_loss", pos_loss, batch_size=batch_size)
        self.log("train_detection_loss", detection, batch_size=batch_size)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        graphs_left = batch["graph_left"]  # this is a minibatch of graphs
        graphs_right = batch["graph_right"]
        pos_labels = batch["position"]
        mask_labels = batch["object_found"]
        batch_size = mask_labels.size(0)
        
        pos_pred, mask_pred = self.forward(graphs_left, graphs_right, batch_size)
        total_loss, pos_loss, detection = self.loss(
            pos_pred, mask_pred, pos_labels, mask_labels
        )
        r2, f1 = self.get_metrics(pos_pred, mask_pred, pos_labels, mask_labels)
        
        self.log("val_total_loss", total_loss, batch_size=batch_size)
        self.log("val_pos_loss", pos_loss, batch_size=batch_size)
        self.log("train_detection_loss", detection, batch_size=batch_size)
        self.log("val_pos_r2", r2, batch_size=batch_size)
        self.log("val_classification_f1", f1, batch_size=batch_size)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)