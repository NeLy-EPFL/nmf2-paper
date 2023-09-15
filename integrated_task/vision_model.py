import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pytorch_lightning as pl
import torchmetrics

from flygym.util.config import num_ommatidia_per_eye


def angular_loss(pred_angles, true_angles):
    pred_xy = torch.stack([torch.cos(pred_angles), torch.sin(pred_angles)], dim=1)
    tgt_xy = torch.stack([torch.cos(true_angles), torch.sin(true_angles)], dim=1)
    return 1 - torch.cosine_similarity(pred_xy, tgt_xy, dim=1)


def angular_r2(pred_angles, true_angles):
    true_x = torch.cos(true_angles)
    true_y = torch.sin(true_angles)
    pred_x = torch.cos(pred_angles)
    pred_y = torch.sin(pred_angles)
    x_mean = true_x.mean()
    y_mean = true_y.mean()
    tss = torch.sum((true_x - x_mean) ** 2 + (true_y - y_mean) ** 2)
    rss = torch.sum((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
    return (1 - rss / tss).item()


class VisualFeaturePreprocessor(pl.LightningModule):
    def __init__(
        self,
        *,
        in_channels=2,
        conv_hidden_channels=4,
        conv_out_channels=2,
        linear_hidden_channels=16,
        k_angle=0.5,
        k_dist=2,
        k_classification=0.1,
        k_azimuth=4,
        k_size=2,
        max_dist=10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.k_angle = k_angle
        self.k_dist = k_dist
        self.k_classification = k_classification
        self.k_azimuth = k_azimuth
        self.k_size = k_size
        self.max_dist = max_dist

        # Define model layers
        self.conv1 = gnn.GCNConv(in_channels, conv_hidden_channels)
        self.conv2 = gnn.GCNConv(conv_hidden_channels, conv_out_channels)
        mlp_in_dim = 2 * conv_out_channels * num_ommatidia_per_eye
        self.linear1 = nn.Linear(mlp_in_dim, linear_hidden_channels)
        self.linear2 = nn.Linear(linear_hidden_channels, linear_hidden_channels)
        self.linear3 = nn.Linear(linear_hidden_channels, 7)

        # Define metrics
        self._f1 = torchmetrics.classification.BinaryF1Score()
        self._r2 = torchmetrics.R2Score()

    def forward(self, left_graph, right_graph, batch_size=1):
        conv_features_li = []
        for graph in [left_graph, right_graph]:
            x = self.conv1(graph.x.view(-1, self.in_channels), graph.edge_index)
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
        angle_pred = x[:, 0]
        dist_pred = F.sigmoid(x[:, 1])
        mask_pred = F.sigmoid(x[:, 2])
        azimuth_pred = F.sigmoid(x[:, 3:5])
        rel_size_pred = F.sigmoid(x[:, 5:7])
        return {
            "angle": angle_pred,
            "dist": dist_pred,
            "mask": mask_pred,
            "azimuth": azimuth_pred,
            "rel_size": rel_size_pred,
        }

    def loss(
        self,
        angle_pred,
        angle_label,
        dist_pred,
        dist_label,
        mask_pred,
        mask_label,
        azimuth_pred,
        azimuth_label,
        rel_size_pred,
        rel_size_label,
    ):
        detection_loss = F.binary_cross_entropy(mask_pred, mask_label.float())
        mask = (mask_label == 1) & (mask_pred > 0.5)
        angle_loss = angular_loss(angle_pred[mask], angle_label[mask]).mean()
        dist_loss = F.mse_loss(dist_pred[mask], dist_label[mask])
        azimuth_loss = F.mse_loss(azimuth_pred[mask, :], azimuth_label[mask, :])
        size_loss = F.mse_loss(rel_size_pred[mask, :], rel_size_label[mask, :])
        total_loss = (
            self.k_angle * angle_loss
            + self.k_dist * dist_loss
            + self.k_classification * detection_loss
            + self.k_azimuth * azimuth_loss
            + self.k_size * size_loss
        )
        return {
            "total_loss": total_loss,
            "angle_loss": angle_loss,
            "dist_loss": dist_loss,
            "detection_loss": detection_loss,
            "azimuth_loss": azimuth_loss,
            "size_loss": size_loss,
        }

    def get_metrics(
        self,
        angle_pred,
        angle_label,
        dist_pred,
        dist_label,
        mask_pred,
        mask_label,
        azimuth_pred,
        azimuth_label,
        rel_size_pred,
        rel_size_label,
    ):
        mask_pred_bin = mask_pred > 0.5
        detection_f1 = self._f1(mask_pred_bin.int(), mask_label.int())
        mask = mask_label.bool() & mask_pred_bin
        if mask.sum() < 2:
            angle_r2 = torch.tensor(torch.nan)
            dist_r2 = torch.tensor(torch.nan)
            azimuth_r2 = torch.tensor(torch.nan)
            size_r2 = torch.tensor(torch.nan)
        else:
            angle_r2 = angular_r2(angle_pred[mask], angle_label[mask])
            dist_r2 = self._r2(dist_pred[mask], dist_label[mask])
            azimuth_r2 = self._r2(
                azimuth_pred[mask].flatten(), azimuth_label[mask].flatten()
            )
            size_r2 = self._r2(
                rel_size_pred[mask].flatten(), rel_size_label[mask].flatten()
            )
        return {
            "angle_r2": angle_r2,
            "dist_r2": dist_r2,
            "detection_f1": detection_f1,
            "azimuth_r2": azimuth_r2,
            "size_r2": size_r2,
        }

    def training_step(self, batch, batch_idx):
        graphs_left = batch["graph_left"]  # this is a minibatch of graphs
        graphs_right = batch["graph_right"]
        batch_size = batch["object_found"].size(0)
        pred = self.forward(graphs_left, graphs_right, batch_size)

        loss = self.loss(
            angle_pred=pred["angle"],
            angle_label=batch["angular_pos"][:, 0],
            dist_pred=pred["dist"],
            dist_label=batch["angular_pos"][:, 1],
            mask_pred=pred["mask"],
            mask_label=batch["object_found"],
            azimuth_pred=pred["azimuth"],
            azimuth_label=batch["azimuth"],
            rel_size_pred=pred["rel_size"],
            rel_size_label=batch["rel_size"],
        )

        self.log("train_total_loss", loss["total_loss"], batch_size=batch_size)
        self.log("train_angle_loss", loss["angle_loss"], batch_size=batch_size)
        self.log("train_distance_loss", loss["dist_loss"], batch_size=batch_size)
        self.log("train_detection_loss", loss["detection_loss"], batch_size=batch_size)
        self.log("train_azimuth_loss", loss["azimuth_loss"], batch_size=batch_size)
        self.log("train_size_loss", loss["size_loss"], batch_size=batch_size)

        return loss["total_loss"]

    def validation_step(self, batch, batch_idx):
        graphs_left = batch["graph_left"]  # this is a minibatch of graphs
        graphs_right = batch["graph_right"]
        batch_size = batch["object_found"].size(0)
        pred = self.forward(graphs_left, graphs_right, batch_size)

        loss = self.loss(
            angle_pred=pred["angle"],
            angle_label=batch["angular_pos"][:, 0],
            dist_pred=pred["dist"],
            dist_label=batch["angular_pos"][:, 1],
            mask_pred=pred["mask"],
            mask_label=batch["object_found"],
            azimuth_pred=pred["azimuth"],
            azimuth_label=batch["azimuth"],
            rel_size_pred=pred["rel_size"],
            rel_size_label=batch["rel_size"],
        )
        metrics = self.get_metrics(
            angle_pred=pred["angle"],
            angle_label=batch["angular_pos"][:, 0],
            dist_pred=pred["dist"],
            dist_label=batch["angular_pos"][:, 1],
            mask_pred=pred["mask"],
            mask_label=batch["object_found"],
            azimuth_pred=pred["azimuth"],
            azimuth_label=batch["azimuth"],
            rel_size_pred=pred["rel_size"],
            rel_size_label=batch["rel_size"],
        )

        self.log("val_total_loss", loss["total_loss"], batch_size=batch_size)
        self.log("val_angle_loss", loss["angle_loss"], batch_size=batch_size)
        self.log("val_distance_loss", loss["dist_loss"], batch_size=batch_size)
        self.log("val_detection_loss", loss["detection_loss"], batch_size=batch_size)
        self.log("val_azimuth_loss", loss["azimuth_loss"], batch_size=batch_size)
        self.log("val_size_loss", loss["size_loss"], batch_size=batch_size)
        self.log("val_angle_r2", metrics["angle_r2"], batch_size=batch_size)
        self.log("val_distance_r2", metrics["dist_r2"], batch_size=batch_size)
        self.log("val_detection_f1", metrics["detection_f1"], batch_size=batch_size)
        self.log("val_azimuth_r2", metrics["azimuth_r2"], batch_size=batch_size)
        self.log("val_size_r2", metrics["size_r2"], batch_size=batch_size)

        return loss["total_loss"]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
