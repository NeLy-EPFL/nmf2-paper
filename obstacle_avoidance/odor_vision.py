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
        size: Tuple[float, float] = (100, 100),
        walls_dist: Tuple[float,float] = (12,5),
        walls_dims: Tuple[float,float,float] = (0.5,6,2),
        **kwargs
    ):
        super().__init__(odor_source = np.array([[25, 0, 0]]), peak_intensity=np.array([[1000]]), **kwargs)
    
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
        
        self.root_element.worldbody.add(
            "camera",
            name="arena_camera_1",
            mode="fixed",
            pos=(-10, 0, 20),
            euler=(0, -np.pi / 4, -np.pi / 2),
            fovy=60,
        )

        (thick, long, height) = walls_dims
        corners = np.array([
            [center+[i,j,0] for i in [-thick/2,thick/2] for j in [-long/2,long/2]] for center in self.walls_positions
        ])
        self.corner_positions = corners[:,:,:2]


    def _is_visible(self, obs_idx, pos, orientation):
        visible = False
        vec_fly = np.array([np.cos(orientation[0]+np.pi/2),np.sin(orientation[0]+np.pi/2)])

        for c in self.corner_positions[obs_idx]:
            dist = np.linalg.norm(c-pos)
            vec_corner = np.array((1/dist)*(c-pos))
            costheta = np.dot(vec_corner,vec_fly)
            if costheta>(-1/np.sqrt(2)):
                visible = True

        return visible

    def get_walls_distance(self, position: np.ndarray, orientation: np.ndarray):
        norms = []
        distances = []
        for idx, p in enumerate(self.walls_positions):
            if self._is_visible(idx,position[:2], orientation):
                # Fly direction vector
                (xf,yf) = (np.cos(orientation[0]+np.pi/2), np.sin(orientation[0]+np.pi/2))
                # Basis change matrix (to get distance in fly referential)
                trans_mat = np.array([[xf,yf],[-yf,xf]])
                vec = trans_mat@(np.array(p[:2]) - position[:2])

                norms.append(np.linalg.norm(vec))
                distances.append(vec)
        if distances:     
            distances = np.array(distances)
            idx = np.argmin(np.array(norms))
            return [*distances[idx], True]
        else:
            return [0, 0, False]      


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
        n_stabilisation_steps: int = 5000,
        obj_threshold=50,
        max_time=2,
        pos_range=[[0,30],[-12,12]],
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

        (self.xmin,self.xmax) = pos_range[0]
        (self.ymin,self.ymax) = pos_range[1]

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
        # Normalize vision and reduce to one channel
        vision = (1/255)*raw_obs['vision'].max(axis=2)

        features = self.arena.get_walls_distance(raw_obs['fly'][0,:], raw_obs['fly'][2,:])
        # if features[-1]:
        #     # Normalize distance to range of possible positions
        #     features[0] = (features[0]-self.xmin)/(self.xmax-self.xmin)
        #     features[1] = (features[1]-self.ymin)/(self.ymax-self.ymin)

        # Check that the visual input contains an obstacle, otherwise mask features
        # see_obs = np.count_nonzero(vision < self.obj_threshold) > 0
        # if see_obs and features[-1]==False:
        #     print("sees smth but shouldn't")

        return np.concatenate((vision.flatten(), features))

