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
import flygym.util.config as config
import flygym.util.vision as vision


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
            rgb = np.array(color_cycle_rgb[1]) / 255
            marker_colors = [(*rgb, 1)] * self.num_odor_sources
            num_odor_sources = self.odor_source.shape[0]
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
            pos=(odor_source[0, 0] / 2, -25, 10),
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
        obj_threshold=50,
        decision_dt=0.05,
        n_stabilisation_dur=0.3,
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

        self.max_time = max_time
        self.arena = arena
        self.num_substeps = int(decision_dt / self.timestep)
        self.obj_threshold = obj_threshold

        # Override spaces
        # action space: 2D vector of amplitude and phase for oscillators on each side
        # observation space:
        #  - 2D vector of x-y position of object relative to the fly, norm. to [0, 1]
        #  - scalar probability that there is an object in view, [0, 1]
        #  - 2D vector of mean odor intensity on each side, norm. to [0, 1]
        #  - 2D vector of current oscillator amp. on each side, norm. to [0, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(9,))

        self._last_fly_tgt_dist = np.linalg.norm(
            np.zeros(2) - self.arena.odor_source[0, :2]
        )
        self._last_turning_signal = 0

        # Compute x-y position of each ommatidium
        self.coms = np.empty((config.num_ommatidia_per_eye, 2))
        for i in range(config.num_ommatidia_per_eye):
            mask = vision.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)

    def step(self, turning_signal):
        # turning_signal = turning_signal[0]
        amplitude = np.array([1, -1]) * turning_signal
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
            return np.zeros((9,), dtype="float32"), 0, False, True, {}

        # Check if visual inputs are rendered recently
        assert abs(self.curr_time - self._last_vision_update_time) < 0.25 * self.timestep

        # Parse observations
        visual_features = self._get_visual_features()
        odor_intensity = raw_obs["odor_intensity"][0, :].reshape(2, 2).mean(axis=0)
        odor_intensity /= self.arena.peak_odor_intensity[0, 0]
        last_action = self._last_turning_signal / 2 + 0.5
        obs = np.array(
            [*visual_features, *odor_intensity, last_action], dtype=np.float32
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

        # calculate tentative reward
        if distance < 2:
            reward = 30
            terminated = True
            info["state_desc"] = "success"
        elif collision:
            reward = -1
            terminated = True
            info["state_desc"] = "collision"
        elif info["flip"]:
            reward = -5
            terminated = True
            info["state_desc"] = "flipped"
        else:
            reward = distance_reward
            terminated = False
            info["state_desc"] = "seeking"

        # penalty for too close to the cylinder
        fly_obstacle_distance = np.linalg.norm(
            fly_pos - self.arena.obstacle_positions[0, :]
        )
        obstacle_penalty = 1 - (fly_obstacle_distance - 3) / 2
        obstacle_penalty = np.clip(obstacle_penalty, 0, 1)
        reward -= obstacle_penalty

        # apply penalty for rapid turning
        action_diff_penalty = (
            np.abs(turning_signal[0] - self._last_turning_signal) * 0.25
        )
        # print(turning_signal, self._last_turning_signal)
        reward -= action_diff_penalty

        info["distance_reward"] = distance_reward
        info["has_collision"] = has_collision
        info["distance"] = distance
        truncated = (
            self.curr_time > self.max_time and not terminated
        )  # start a new episode

        if self.debug_mode:
            print(
                f"fly_pos: {fly_pos}, final reward={reward}, state={info['state_desc']}"
            )
            print(
                f"  dist reward={distance_reward}, obstacle penalty={obstacle_penalty}, action diff={action_diff_penalty}"
            )
            if terminated:
                print("terminated")
            if truncated:
                print("truncated")

        self._last_turning_signal = turning_signal[0]
        return obs, reward, terminated, truncated, info

    def reset(self, seed=0):
        super().reset()
        obs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="float32")
        self._last_fly_tgt_dist = np.linalg.norm(
            np.zeros(2) - self.arena.odor_source[0, :2]
        )
        self._last_turning_signal = 0
        if self.debug_mode:
            print("resetting environment")
        return obs, {"state_desc": "reset"}

    def _get_visual_features(self):
        raw_obs = super().get_observation()
        # features = np.full((2, 3), np.nan)  # ({L, R}, {y_center, x_center, area})
        features = np.zeros((2, 3))
        for i, ommatidia_readings in enumerate(raw_obs["vision"]):
            is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold
            is_obj[
                np.arange(is_obj.size) % 2 == 1
            ] = False  # only use pale-type ommatidia
            is_obj_coords = self.coms[is_obj]
            if is_obj_coords.shape[0] > 0:
                features[i, :2] = is_obj_coords.mean(axis=0)
            features[i, 2] = is_obj_coords.shape[0]
        features[:, 0] /= config.raw_img_height_px  # normalize y_center
        features[:, 1] /= config.raw_img_width_px  # normalize x_center
        # features[:, :2] = features[:, :2] * 2 - 1  # center around 0
        features[:, 2] /= config.num_ommatidia_per_eye  # normalize area
        return features.flatten()
