import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pytorch_lightning as pl
import torchmetrics
from typing import Tuple, Callable, Optional, List, Union
from dm_control.rl.control import PhysicsError

from flygym.envs.nmf_mujoco import MuJoCoParameters
from flygym.arena import BaseArena
from flygym.util.turning_controller import TurningController
from flygym.util.data import color_cycle_rgb
from flygym.util.config import num_ommatidia_per_eye


class ObstacleOdorArena(BaseArena):
    num_sensors = 4

    def __init__(
        self,
        terrain: BaseArena,
        obstacle_positions: np.ndarray = np.array([(7.5, 0), (12.5, 5), (17.5, -5)]),
        obstacle_colors: Union[np.ndarray, Tuple] = (0, 0, 0, 1),
        obstacle_radius: float = 1,
        obstacle_height: float = 4,
        odor_source: np.ndarray = np.array([[25, 0, 2]]),
        peak_intensity: np.ndarray = np.array([[1]]),
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: Optional[List[Tuple[float, float, float, float]]] = None,
        marker_size: float = 0.1,
        user_camera_settings: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float], float]
        ] = None,
    ):
        self.terrain_arena = terrain
        self.obstacle_positions = obstacle_positions
        self.root_element = terrain.root_element
        self.friction = terrain.friction
        self.obstacle_radius = obstacle_radius
        z_offset = terrain.get_spawn_position(np.zeros(3), np.zeros(3))[0][2]
        obstacle_colors = np.array(obstacle_colors)
        if obstacle_colors.shape == (4,):
            obstacle_colors = np.array(
                [obstacle_colors for _ in range(obstacle_positions.shape[0])]
            )
        else:
            assert obstacle_colors.shape == (obstacle_positions.shape[0], 4)

        self.odor_source = np.array(odor_source)
        self.peak_odor_intensity = np.array(peak_intensity)
        self.num_odor_sources = self.odor_source.shape[0]
        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )
        self.odor_dim = self.peak_odor_intensity.shape[1]
        self.diffuse_func = diffuse_func

        # Add markers at the odor sources
        if marker_colors is None:
            marker_colors = []
            num_odor_sources = self.odor_source.shape[0]
            for i in range(num_odor_sources):
                rgb = np.array(color_cycle_rgb[i % num_odor_sources]) / 255
                rgba = (*rgb, 1)
                marker_colors.append(rgba)
        for i, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors)):
            pos = list(pos)
            pos[2] += z_offset
            marker_body = self.root_element.worldbody.add(
                "body", name=f"odor_source_marker_{i}", pos=pos, mocap=True
            )
            marker_body.add(
                "geom", type="capsule", size=(marker_size, marker_size), rgba=rgba
            )

        # Reshape odor source and peak intensity arrays to simplify future claculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(_odor_source_repeated, self.odor_dim, axis=1)
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.num_sensors, axis=2
        )
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(
            _peak_intensity_repeated, self.num_sensors, axis=2
        )
        self._peak_intensity_repeated = _peak_intensity_repeated

        # Add obstacles
        self.obstacle_bodies = []
        obstacle_material = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        self.obstacle_z_pos = z_offset + obstacle_height / 2
        for i in range(obstacle_positions.shape[0]):
            obstacle_pos = [*obstacle_positions[i, :], self.obstacle_z_pos]
            obstacle_color = obstacle_colors[i]
            obstacle_body = self.root_element.worldbody.add(
                "body", name=f"obstacle_{i}", mocap=True, pos=obstacle_pos
            )
            self.obstacle_bodies.append(obstacle_body)
            obstacle_body.add(
                "geom",
                type="cylinder",
                size=(obstacle_radius, obstacle_height / 2),
                rgba=obstacle_color,
                material=obstacle_material,
            )

        # Add monitor cameras
        self.root_element.worldbody.add(
            "camera",
            name="side_cam",
            mode="fixed",
            pos=(12.5, -25, 10),
            euler=(np.deg2rad(75), 0, 0),
            fovy=50,
        )
        self.root_element.worldbody.add(
            "camera",
            name="back_cam",
            mode="fixed",
            pos=(-10, 0, 12),
            euler=(0, np.deg2rad(-60), -np.deg2rad(90)),
            fovy=50,
        )
        self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(10, 0, 40),
            euler=(0, 0, 0),
            fovy=50,
        )
        if user_camera_settings is not None:
            cam_pos, cam_euler, cam_fovy = user_camera_settings
            self.root_element.worldbody.add(
                "camera",
                name="user_cam",
                mode="fixed",
                pos=cam_pos,
                euler=cam_euler,
                fovy=cam_fovy,
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.terrain_arena.get_spawn_position(rel_pos, rel_angle)

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, w, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, w)
        scaling = self.diffuse_func(dist_euc)  # (n, k, w)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, w)
        return intensity.sum(axis=0)  # (k, w)


class NMFNavigation(TurningController):
    def __init__(
        self,
        arena,
        vision_model,
        ommatidia_graph,
        device=torch.device("cpu"),
        decision_dt=0.05,
        n_stabilisation_dur=0.3,
        distance_threshold=15,
        max_time=5,
        test_mode=False,
        debug_mode=False,
        **kwargs,
    ) -> None:
        self.debug_mode = debug_mode
        sim_params = MuJoCoParameters(
            render_playspeed=0.1,
            render_camera="birdeye_cam",
            enable_vision=True,
            render_raw_vision=test_mode,
            enable_olfaction=True,
            render_mode="saved" if test_mode else "headless",
            vision_refresh_rate=int(1 / decision_dt),
            enable_adhesion=True,
            actuator_kp=30,
            adhesion_gain=20,
        )
        super().__init__(
            sim_params=sim_params,
            arena=arena,
            stabilisation_dur=n_stabilisation_dur,
            detect_flip=True,
            **kwargs,
        )

        self.device = device
        self.ommatidia_graphs = [ommatidia_graph.clone(), ommatidia_graph.clone()]
        self.vision_model = vision_model.to(self.device)
        self.max_time = max_time
        self.arena = arena
        self.num_substeps = int(decision_dt / self.timestep)
        self.distance_threshold = distance_threshold

        # Override spaces
        # action space: 2D vector of amplitude and phase for oscillators on each side
        # observation space:
        #  - 2D vector of x-y position of object relative to the fly, norm. to [0, 1]
        #  - scalar probability that there is an object in view, [0, 1]
        #  - 2D vector of mean odor intensity on each side, norm. to [0, 1]
        #  - 2D vector of current oscillator amp. on each side, norm. to [0, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,))

        self._last_fly_tgt_dist = np.linalg.norm(
            np.zeros(2) - self.arena.odor_source[0, :2]
        )
        self._last_action = np.zeros((2,))

    def update_retina_graphs(self, intensities):
        intensities = torch.tensor(intensities, dtype=torch.float32).sum(axis=-1) / 255
        intensities = intensities.to(self.device)
        for i in range(2):
            self.ommatidia_graphs[i].x = intensities[i, :]

    def step(self, amplitude):
        try:
            obstacle_contact_counter = 0
            for i in range(self.num_substeps):
                raw_obs, _, raw_term, raw_trunc, info = super().step(amplitude)
                collision_forces = [
                    np.abs(self.physics.named.data.cfrc_ext[f"obstacle_{j}"]).sum()
                    for j in range(len(self.arena.obstacle_positions))
                ]
                if np.sum(collision_forces) > 1:
                    obstacle_contact_counter += 1
                super().render()
            collision = obstacle_contact_counter > 20
        except PhysicsError:
            print("Physics error, resetting environment")
            return np.zeros((7,), dtype="float32"), 0, False, True, {}

        # Check if visual inputs are rendered recently
        assert abs(self.curr_time - self._last_vision_update_time) < 0.5 * self.timestep

        # Parse observations
        self.update_retina_graphs(self.curr_visual_input)
        pos_pred, obj_prob = self.vision_model(*self.ommatidia_graphs)
        pos_pred = pos_pred.detach().numpy().squeeze() / self.distance_threshold
        pos_pred = np.clip(pos_pred, 0, 1)
        obj_prob = obj_prob.detach().numpy()
        odor_intensity = raw_obs["odor_intensity"][0, :].reshape(2, 2).mean(axis=0)
        odor_intensity /= self.arena.peak_odor_intensity[0, 0]
        last_action = self._last_action / 2 + 0.5
        obs = np.concatenate(
            [pos_pred, obj_prob, odor_intensity, last_action], dtype=np.float32
        )

        # Calculate reward
        # calculate distance reward
        fly_pos = super().get_observation()["fly"][0, :2]
        tgt_pos = self.arena.odor_source[0, :2]
        distance = np.linalg.norm(fly_pos - tgt_pos)
        distance_reward = self._last_fly_tgt_dist - distance
        self._last_fly_tgt_dist = distance

        # check if fly is too close to any obstacle
        has_collision = False
        for obst_pos in self.arena.obstacle_positions:
            if np.linalg.norm(fly_pos - obst_pos) < self.arena.obstacle_radius + 1:
                has_collision = True
                break

        # calculate final reward
        if distance < 2:
            reward = 30
            terminated = True
            info["state_desc"] = "success"
        elif collision:
            reward = -10
            terminated = True
            info["state_desc"] = "collision"
        elif info["flip"]:
            reward = -10
            terminated = True
            info["state_desc"] = "flipped"
        else:
            reward = distance_reward
            terminated = False
            info["state_desc"] = "seeking"

        # apply penalty for rapid turning
        action_diff = np.abs(amplitude - self._last_action).sum() / 20

        info["distance_reward"] = distance_reward
        info["has_collision"] = has_collision
        info["distance"] = distance
        truncated = (
            self.curr_time > self.max_time and not terminated
        )  # start a new episode

        if self.debug_mode:
            print(f"fly_pos: {fly_pos}, reward={reward}, state={info['state_desc']}")
            if terminated:
                print("terminated")
            if truncated:
                print("truncated")

        self._last_action = amplitude
        return obs, reward, terminated, truncated, info

    def reset(self, seed=0):
        super().reset()
        obs = np.array([0, 0, 0, 0, 0, 0.5, 0.5], dtype="float32")
        self._last_fly_tgt_dist = np.linalg.norm(
            np.zeros(2) - self.arena.odor_source[0, :2]
        )
        self._last_action = np.zeros((2,))
        if self.debug_mode:
            print("resetting environment")
        return obs, {"state_desc": "reset"}


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
