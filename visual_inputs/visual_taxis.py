import numpy as np
import gymnasium as gym
from typing import Tuple
from dm_control import mjcf
from dm_control.rl.control import PhysicsError

import flygym.util.vision as vision
import flygym.util.config as config
from flygym.arena import BaseArena
from flygym.envs.nmf_mujoco import MuJoCoParameters

from cpg_controller import NMFCPG


class MovingObjArena(BaseArena):
    """Flat terrain with a hovering moving object.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.
    ball_pos : Tuple[float,float,float]
        The position of the floating object in the arena.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of the terrain in (x, y) dimensions.
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    obj_radius : float
        Radius of the spherical floating object in mm.
    obj_spawn_pos : Tuple[float,float,float]
        Initial position of the object, by default (0, 2, 1).
    move_mode : string
        Type of movement performed by the floating object.
        Can be "random" (default value), "straightHeading", "circling" or "s_shape".
    move_speed : float
        Speed of the moving object. Angular velocity if move_mode=="circling" or "s_shape".
    """

    def __init__(
        self,
        size: Tuple[float, float] = (200, 200),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        obj_radius: float = 1,
        obj_spawn_pos: Tuple[float, float, float] = (0, 2, 0),
        move_mode: str = "random",
        move_speed: float = 25,
    ):
        self.root_element = mjcf.RootElement()
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.4, 0.4, 0.4),
            rgb2=(0.5, 0.5, 0.5),
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
        self.root_element.worldbody.add("body", name="b_plane")
        # Add ball
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        self.root_element.worldbody.add(
            "body", name="ball_mocap", mocap=True, pos=obj_spawn_pos, gravcomp=1
        )
        self.object_body = self.root_element.find("body", "ball_mocap")
        self.object_body.add(
            "geom",
            name="ball",
            type="sphere",
            size=(obj_radius, obj_radius),
            rgba=(0.0, 0.0, 0.0, 1),
            material=obstacle,
        )
        self.friction = friction
        self.init_ball_pos = (obj_spawn_pos[0], obj_spawn_pos[1], obj_radius)
        self.ball_pos = self.init_ball_pos
        self.move_mode = move_mode
        self.move_speed = move_speed
        if move_mode == "straightHeading":
            self.direction = 0.5 * np.pi * (np.random.rand() - 0.5)
        elif move_mode == "circling":
            self.rotation_direction = np.random.choice([-1, 1])
            self.rotation_center = (
                np.random.randint(0, 4),
                self.rotation_direction * np.random.randint(6, 12),
            )  # (10*np.random.rand(),10*np.random.rand())
            self.radius = np.linalg.norm(
                np.array(self.ball_pos[0:2]) - np.array(self.rotation_center)
            )
            self.theta = np.arcsin(
                (self.ball_pos[1] - self.rotation_center[1]) / self.radius
            )
            self.move_speed = move_speed / self.radius
        elif move_mode == "s_shape":
            self.pos_func = lambda t: np.array(
                [
                    move_speed * t + obj_spawn_pos[0],
                    0.25 * move_speed * np.sin(t * 3) + obj_spawn_pos[1],
                    obj_radius,
                ]
            )
        elif move_mode != "random":
            raise NotImplementedError

        self.root_element.worldbody.add(
            "camera",
            name="birdseye_cam",
            mode="fixed",
            pos=(0, 0, 50),
            euler=(0, 0, 0),
            fovy=40,
        )

        self.curr_time = 0
        self._obj_pos_history_li = [[self.curr_time, *self.ball_pos]]

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def step(self, dt, physics):
        if self.move_mode == "random":
            x_disp = self.move_speed * (np.random.rand() - 0.45) * dt
            y_disp = self.move_speed * (np.random.rand() - 0.5) * dt
            self.ball_pos = self.ball_pos + np.array([x_disp, y_disp, 0])
        elif self.move_mode == "straightHeading":
            x_disp = self.move_speed * np.cos(self.direction) * dt
            y_disp = self.move_speed * np.sin(self.direction) * dt
            self.ball_pos = self.ball_pos + np.array([x_disp, y_disp, 0])
        elif self.move_mode == "circling":
            self.theta = self.theta + self.rotation_direction * self.move_speed * dt
            self.theta %= 2 * np.pi
            x = self.rotation_center[0] + self.radius * np.cos(self.theta)
            y = self.rotation_center[1] + self.radius * np.sin(self.theta)
            self.ball_pos = np.array([x, y, self.ball_pos[2]])
        elif self.move_mode == "s_shape":
            self.ball_pos = self.pos_func(self.curr_time)

        physics.bind(self.object_body).mocap_pos = self.ball_pos

        self.curr_time += dt
        self._obj_pos_history_li.append([self.curr_time, *self.ball_pos])
    
    def reset(self, physics):
        self.curr_time = 0
        self.ball_pos = self.init_ball_pos
        physics.bind(self.object_body).mocap_pos = self.ball_pos
        self._obj_pos_history_li = [[self.curr_time, *self.ball_pos]]
        

    @property
    def obj_pos_history(self):
        return np.array(self._obj_pos_history_li)

    # def reset(self, new_spawn_pos=False, new_move_mode=False, new_move_speed=False):
    #     """Reset the object position in the arena and update characteristics of its movement.

    #     Parameters
    #     ----------
    #     new_spawn_pos : bool or Tuple[float,float,float]
    #         If boolean, indicates whether a new initial position for the object is drawn randomly (True)
    #         or the previous initial position is used (False - default).
    #         If tuple, new position to be used as initial object position.
    #     new_move_mode : bool or string
    #         If boolean, indicates whether a new move_mode for the object is drawn randomly from the set of
    #         possible move_mode (True) or if the previous one is kept (False).
    #         If string, new move_mode to be used for the object.
    #     new_move_speed : bool or float
    #         If boolean, indicates whether move_speed of the object is updated to the default for the object's
    #         move_mode (True) or if the previous one is kept (False).
    #         If float, value of the new move_speed.
    #     """
    #     if isinstance(new_spawn_pos, bool):
    #         if new_spawn_pos == True:
    #             self.init_ball_pos = (
    #                 np.random.randint(6, 8),
    #                 np.random.randint(-8, 8),
    #                 self.ball_pos[2],
    #             )
    #     else:
    #         self.init_ball_pos = new_spawn_pos

    #     if isinstance(new_move_mode, bool):
    #         if new_move_mode == True:
    #             self.move_mode = np.random.choice(
    #                 ["straightHeading", "s_shape"]
    #             )  # , "random"])
    #     else:
    #         self.move_mode = new_move_mode

    #     self.ball_pos = self.init_ball_pos
    #     if self.move_mode == "straightHeading":
    #         # Draw new random direction
    #         self.direction = 0.5 * np.pi * (np.random.rand() - 0.5)

    #     elif self.move_mode == "circling":
    #         # Draw new rotation direction and center
    #         self.rotation_direction = np.random.choice([-1, 1])
    #         self.rotation_center = (
    #             np.random.randint(0, 4),
    #             self.rotation_direction * np.random.randint(6, 12),
    #         )
    #         self.radius = np.linalg.norm(
    #             np.array(self.ball_pos[0:2]) - np.array(self.rotation_center)
    #         )
    #         self.theta = np.arcsin(
    #             (self.ball_pos[1] - self.rotation_center[1]) / self.radius
    #         )

    #     elif self.move_mode == "s_shape":
    #         self.radius = 10
    #         self.rotation_center = (self.ball_pos[0] + self.radius, 0)
    #         self.rotation_direction = np.random.choice([-1, 1])
    #         self.theta = np.pi

    #     if isinstance(new_move_speed, bool):
    #         if new_move_speed == True:
    #             base_speed = 0.003
    #             if self.move_mode == "straightHeading":
    #                 self.move_speed = base_speed
    #             elif self.move_mode == "circling" or self.move_mode == "s_shape":
    #                 self.move_speed = base_speed / self.radius
    #     else:
    #         self.move_speed = new_move_speed

        self.curr_time = 0


class NMFVisualTaxis(NMFCPG):
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
        self.num_substeps = int(decision_dt / self.timestep)

        # Override spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,))

        # Compute x-y position of each ommatidium
        self.coms = np.empty((config.num_ommatidia_per_eye, 2))
        for i in range(config.num_ommatidia_per_eye):
            mask = vision.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)

        self._last_offset_from_ideal = self._calc_offset_from_ideal(
            np.zeros(2), self.arena.ball_pos[:2]
        )

    @staticmethod
    def _calc_offset_from_ideal(fly_pos, obj_pos):
        fly_obj_distance = np.linalg.norm(fly_pos - obj_pos)
        return np.abs(fly_obj_distance - 5)

    def step(self, amplitude):
        try:
            for i in range(self.num_substeps):
                raw_obs, _, raw_term, raw_trunc, info = super().step(amplitude)
                super().render()
        except PhysicsError:
            print("Physics error, resetting environment")
            return np.zeros((6,), dtype="float32"), 0, False, True, {}

        assert abs(self.curr_time - self._last_vision_update_time) < 0.5 * self.timestep
        obs = self._get_visual_features().astype("float32")

        # calculate reward
        fly_pos = super().get_observation()["fly"][0, :2]
        curr_offset_from_ideal = self._calc_offset_from_ideal(
            fly_pos, self.arena.ball_pos[:2]
        )
        fly_obj_distance = np.linalg.norm(fly_pos - self.arena.ball_pos[:2])
        unadjusted_reward = self._last_offset_from_ideal - curr_offset_from_ideal
        if curr_offset_from_ideal > 15:  # too far from object, fail
            reward = -15
            terminated = True
            info["state_desc"] = "too far from object"
        elif obs[2] + obs[5] < 0.005:  # lost object from both eyes, fail
            reward = -15
            terminated = True
            info["state_desc"] = "object lost visually"
        elif curr_offset_from_ideal < 1:  # this is perfect, reward regardless of change
            reward = 3
            terminated = False
            info["state_desc"] = "ideal range"
        elif fly_obj_distance < 3:  # collision/too close, fail
            reward = -5
            terminated = True
            info["state_desc"] = "collision"
        else:  # reward is improvement from last step
            reward = unadjusted_reward
            terminated = False
            info["state_desc"] = "seeking"
        info["unadjusted reward"] = unadjusted_reward
        info["offset_from_ideal"] = curr_offset_from_ideal
        truncated = self.curr_time > 1 and not terminated  # start a new episode
        self._last_offset_from_ideal = curr_offset_from_ideal

        return obs, reward, terminated, truncated, info

    def reset(self):
        super().reset()
        self.arena.reset(self.physics)
        obs = self._get_visual_features().astype("float32")
        return obs, {}

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
            else: # Deal with cases where the object is seen by one eye only
                self._see_obj -= 1
                if self._last_observation is not None:
                    features[i, :2] = self._last_observation[3*i:3*i+1]
                else:
                    features[i, :2] = 0
            features[i, 2] = is_obj_coords.shape[0]
        features[:, 0] /= config.raw_img_height_px  # normalize y_center
        features[:, 1] /= config.raw_img_width_px  # normalize x_center
        # features[:, :2] = features[:, :2] * 2 - 1  # center around 0
        features[:, 2] /= config.num_ommatidia_per_eye  # normalize area

        self._last_observation = features.flatten()
        return features.flatten()

    def _calc_delta_dist(self, fly_pos, obj_pos):
        dist_from_obj = np.linalg.norm(fly_pos - obj_pos)
        if self._last_offset_from_ideal is not None:
            delta_dist = self._last_offset_from_ideal - dist_from_obj
        else:
            delta_dist = 0
        self._last_offset_from_ideal = dist_from_obj
        return delta_dist

    def _compute_orientation_reward(self, fly_orient, fly_pos, obj_pos):
        terminated = False
        pitch_threshold = np.pi/2

        # Termination with penalty if the fly has tipped over
        if abs(fly_orient[2]) > pitch_threshold: #### which is pitch???
            reward = -200
            terminated = True
            return reward, terminated

        dist_from_obj = np.linalg.norm(fly_pos - obj_pos)
        vec_fly = np.array([np.cos(fly_orient[0]+np.pi/2),np.sin(fly_orient[0]+np.pi/2)])
        vec_obj = np.array((1/dist_from_obj)*(obj_pos[:2]-fly_pos[:2]))
        cosangle = np.dot(vec_obj, vec_fly)

        # Termination if object out of field of view
        if cosangle < (-1/np.sqrt(2)):
            terminated = True
            reward = -1
        elif self._last_cosangle is not None:
            if cosangle>self._last_cosangle:
                reward = abs(cosangle)
            else:
                reward = cosangle
        else:
            reward = 0
        self._last_cosangle = cosangle
        
        # elif self._last_cosangle is not None:
        #     reward = 10*(cosangle-self._last_cosangle)
        # else:
        #     reward = 0
        # self._last_cosangle = cosangle

        return reward, terminated
        

