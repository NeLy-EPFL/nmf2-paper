import numpy as np
import os


import scipy.ndimage as ndimage
import networkx as nx
from pathlib import Path

from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch_geometric as pyg
import torch
from torch.nn import ReLU, Module, Linear, Flatten
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.nn.pool import EdgePooling
import torch_geometric.nn as nn

import pkg_resources

from flygym.util.data import ommatidia_id_map_path, sample_visual_path
from flygym.util.vision import (
    raw_image_to_hex_pxls,
    hex_pxls_to_human_readable,
    num_pixels_per_ommatidia,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Functions for RL training and saving
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule (https://stable-baselines3.readthedocs.io/en/master/guide/examples.html).

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class SaveIntermediateModelsCallback(BaseCallback):
    """
    Callback for saving a model.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveIntermediateModelsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        return

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            save_path = os.path.join(self.log_dir, f"model{self.num_timesteps}_sec_")
            print(f"Saving intermediate model to {save_path}")
            self.model.save(save_path)
        return True


# Graph convolution feature extractor
class CustomModel(Module):
    def __init__(self, in_channels=2, hidden_channels=8, out_channels=1):
        super(CustomModel, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, ommatidia_intensity, graph):
        # print("before", ommatidia_intensity.shape)
        x = self.conv1(ommatidia_intensity, graph.edge_index.to(device))
        # print("after conv1", x.shape)
        x, new_graph = self.pool(x, graph)
        # print("after pool1", x.shape)
        x = x.relu()
        x = self.conv2(x, new_graph.edge_index.to(device))
        # print("after conv2", x.shape)
        x, _ = self.pool(x, new_graph)
        # print("after pool2", x.shape)

        return x
    
    def pool(self, input, graph, nodes_dimension=-2):
        # Keeping only nodes at even indices
        indices = torch.arange(0,input.shape[nodes_dimension],2, dtype=int).to(device)
        output = torch.index_select(input, nodes_dimension, indices)

        # Merging removed nodes into the even one before them
        graphx = pyg.utils.to_networkx(graph)
        odd_even = lambda u,v: (u//2 == v//2)
        graphx = nx.quotient_graph(graphx, odd_even, relabel=True)
        new_graph = pyg.utils.from_networkx(graphx).coalesce()

        return output, new_graph


class CustomGCN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 362, **kwargs):
        super().__init__(observation_space=observation_space, features_dim=features_dim)
        self.n_input_channels = observation_space.shape[-1]

        # Definition of the network layers
        self.gcn = CustomModel()

        # Computation of the graph edges
        self.pg_graph = self.init_ommatidia_graph()

    def forward(self, observations) -> torch.Tensor:
        observations = observations.to(device)
        features = torch.reshape(self.gcn(observations, self.pg_graph), (-1,self.features_dim))
        return features.to(device)
    
    def init_ommatidia_graph(self):
        ommatidia_id_map = np.load(ommatidia_id_map_path)
        dilation_kernel = np.ones((5, 5), dtype=bool)

        # Compute neighbor connections for each ommatidium (edges) and construct graph
        edges = set()
        node_pos = dict()
        for ommatidium_id in range(1, ommatidia_id_map.max() + 1):
            mask = ommatidia_id_map == ommatidium_id
            node_pos[ommatidium_id] = np.mean(np.argwhere(mask), axis=0)
            dilated_mask = ndimage.binary_dilation(mask, dilation_kernel)
            neighbor_ids = np.unique(ommatidia_id_map[dilated_mask])
            for neighbor_id in neighbor_ids:
                if neighbor_id not in [0, ommatidium_id]:
                    pair = sorted([ommatidium_id, neighbor_id])
                    edges.add(tuple(pair))
        
        graph = nx.from_edgelist(edges)
        return pyg.utils.from_networkx(graph)
