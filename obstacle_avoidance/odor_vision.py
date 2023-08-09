import numpy as np
import gymnasium as gym
from typing import Tuple, Callable
from dm_control import mjcf
import os

import flygym.util.vision as vision
import flygym.util.config as config
from flygym.arena.mujoco_arena import OdorArena
from flygym.envs.nmf_mujoco import MuJoCoParameters

from cpg_controller import NMFCPG

class ObstacleOdorArena(OdorArena):
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
        walls_dist: Tuple[float,float] = (12,5),
        walls_dims: Tuple[float,float,float] = (0.5,6,2),
        **kwargs
    ):
        super().__init__(odor_source = np.array([[25, 0, 0]]), peak_intensity=np.array([[1000]]),**kwargs)
    
        # Add obstacles
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        walls_color = (0,0,0,1)

        self.walls_positions = [
            np.array([walls_dist[0], walls_dist[1], walls_dims[2]/2]), 
            np.array([2*walls_dist[0], -walls_dist[1], walls_dims[2]/2])
        ]

        for p in self.walls_positions:
            self.root_element.worldbody.add(
                "geom",
                type="box",
                size=walls_dims,
                pos=p,
                rgba=walls_color,
                material=obstacle,
                friction=self.friction,
            )

    
    def get_walls_distance(self, position: np.ndarray):
        norms = []
        distances = []
        for p in self.walls_positions:
            vec = np.array(p) - position
            norms.append(np.linalg.norm(vec))
            distances.append(vec[:2])

        distances = np.array(distances)
        idx = np.argmin(np.array(norms))
        return distances[idx]




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
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))


    def reset(self):
        raw_obs, info = super().reset()
        self.arena.reset(new_spawn_pos=True, new_move_mode=True)
        return raw_obs, info

    def step(self, amplitude):
        for i in range(self.num_sub_steps):
            raw_obs, _, raw_term, raw_trunc, raw_info = super().step(amplitude)
            super().render()

        obs = raw_obs['odor_intensity'][0]
        reward = np.mean(obs)

        truncated = raw_trunc or self.curr_time >= self.max_time
        terminated = raw_term
        return obs, reward, terminated, truncated, raw_info


class NMFObservation(NMFCPG):
    def __init__(
        self,
        decision_dt=0.0001,
        n_stabilisation_steps: int = 0,
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


    def reset(self):
        raw_obs, info = super().reset()
        self.arena.reset(new_spawn_pos=True, new_move_mode=True)
        return raw_obs, info

    def step(self, amplitude):
        for i in range(self.num_sub_steps):
            raw_obs, _, raw_term, raw_trunc, raw_info = super().step(amplitude)
            super().render()

        obs = self._process_obs(raw_obs)

        truncated = raw_trunc or self.curr_time >= self.max_time
        terminated = raw_term
        return obs, 0, terminated, truncated, raw_info
    
    def _process_obs(self, raw_obs):
        vision = raw_obs['vision'].max(axis=2)

        distance = self.arena.get_walls_distance(raw_obs['fly'][0,:])
        # Check that an obstacle is visible
        see_obs = np.count_nonzero(vision < self.obj_threshold) > 0
        
        return np.concatenate((vision.flatten(), see_obs*distance, [see_obs]))

