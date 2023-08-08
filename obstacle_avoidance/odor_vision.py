import numpy as np
import gymnasium as gym
from typing import Tuple, Callable
from dm_control import mjcf
import os

import flygym.util.vision as vision
import flygym.util.config as config
from flygym.arena import BaseArena
from flygym.envs.nmf_mujoco import MuJoCoParameters

from cpg_controller import NMFCPG

class ObstacleOdorArena(BaseArena):
    """Terrain with an odor source and wall obstacles.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.

    Parameters
    ----------
    size : Tuple[float, float]
        The size of the terrain in (x, y) dimensions.
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    odor_source : np.ndarray
        The position of the odor source in (x, y, z) coordinates. The shape
        of the array is (n_sources, 3).
    peak_intensity : np.ndarray
        The peak intensity of the odor source. The shape of the array is
        (n_sources, n_dimensions). Note that the odor intensity can be
        multidimensional.
    diffuse_func : Callable
        The function that, given a distance from the odor source, returns
        the relative intensity of the odor. By default, this is a inverse
        square relationship.
    
    """
    def __init__(
        self,
        size: Tuple[float, float] = (50, 50),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        odor_source: np.ndarray = np.array([[25, 0, 0]]),
        peak_intensity: np.ndarray = np.array([[1]]),
        diffuse_func: Callable = lambda x: (x) ** -2,
        walls_dist: Tuple[float,float] = (12,5),
        walls_dims: Tuple[float,float,float] = (0.5,6,2),
    ):
        self.root_element = mjcf.RootElement()
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.45, 0.55, 0.65),
            rgb2=(0.4, 0.5, 0.6),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=ground_size,
            friction=friction,
        )

        # Add obstacles
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        walls_color = (0,0,0,1)
        self.root_element.worldbody.add(
            "geom",
            type="box",
            size=walls_dims,
            pos=(walls_dist[0], walls_dist[1], walls_dims[2]/2),
            rgba=walls_color,
            material=obstacle,
            friction=friction,
        )
        self.root_element.worldbody.add(
            "geom",
            type="box",
            size=walls_dims,
            pos=(2*walls_dist[0], -walls_dist[1], walls_dims[2]/2),
            rgba=walls_color,
            material=obstacle,
            friction=friction,
        )

        self.friction = friction
        self.odor_source = np.array(odor_source)
        self.peak_odor_intensity = np.array(peak_intensity)
        self.num_odor_sources = self.odor_source.shape[0]
        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )
        self.odor_dim = self.peak_odor_intensity.shape[1]
        self.diffuse_func = diffuse_func

        # Reshape odor source and peak intensity arrays to simplify future calculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(_odor_source_repeated, self.odor_dim, axis=1)
        _odor_source_repeated = np.repeat(_odor_source_repeated, 2, axis=2)
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(_peak_intensity_repeated, 2, axis=2)
        self._peak_intensity_repeated = _peak_intensity_repeated

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """
        Notes
        -----
        2: number of antennae
        3: spatial dimensionality
        k: data dimensionality
        n: number of odor sources

        Input - odor source position: [n, 3]
        Input - antennae position: [2, 3]
        Input - peak intensity: [n, k]
        Input - difusion function: f(dist)

        Reshape sources to S = [n, k*, 2*, 3] (* means repeated)
        Reshape antennae position to A = [n*, k*, 2, 3] (* means repeated)
        Subtract, getting an Delta = [n, k, 2, 3] array of rel difference
        Calculate Euclidean disctance: D = [n, k, 2]

        Apply pre-integrated difusion function: S = f(D) -> [n, k, 2]
        Reshape peak intensities to P = [n, k, 2*]
        Apply scaling: I = P * S -> [n, k, 2] element wise

        Output - Sum over the first axis: [k, 2]
        """
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, 2, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, 2)
        scaling = self.diffuse_func(dist_euc)  # (n, k, 2)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, 2)
        return intensity.sum(axis=0)  # (k, 2)


class NMFAvoidObstacle(NMFCPG):
    def __init__(
        self,
        decision_dt=0.05,
        n_stabilisation_steps: int = 5000,
        obj_threshold=50,
        max_time=2,
        **kwargs
    ) -> None:
        if "sim_params" in kwargs:
            sim_params = kwargs["sim_params"]
            del kwargs["sim_params"]
        else:
            sim_params = MuJoCoParameters()
        sim_params.enable_vision = True
        sim_params.vision_refresh_rate = int(1 / decision_dt)
        self.max_time = max_time

        super().__init__(
            sim_params=sim_params,
            n_oscillators=6,
            n_stabilisation_steps=n_stabilisation_steps,
            **kwargs
        )
        self.decision_dt = decision_dt
        self.obj_threshold = obj_threshold
        self.num_sub_steps = int(decision_dt / self.timestep)

        # Override spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,))


    def reset(self):
        raw_obs, info = super().reset()
        self.arena.reset(new_spawn_pos=True, new_move_mode=True)
        return raw_obs, info

    def step(self, amplitude):
        for i in range(self.num_sub_steps):
            raw_obs, _, raw_term, raw_trunc, raw_info = super().step(amplitude)
            super().render()

        print(raw_obs)
        obs = raw_obs['odor_intensity'][0]
        print(obs)
        reward = np.mean(obs)

        truncated = raw_trunc or self.curr_time >= self.max_time
        terminated = raw_term
        return obs, reward, terminated, truncated, raw_info

