import copy
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from dm_control.rl.control import PhysicsError
from flygym import Camera, Fly
from flygym.examples.turning_controller import HybridTurningNMF

from flygym.arena import BaseArena
from vision_model import VisualFeaturePreprocessor
from flygym.util import get_data_path
from flygym.examples.head_stabilization import HeadStabilizationInferenceWrapper

model_dir = get_data_path("flygym", "data") / "trained_models" / "head_stabilization"
head_stabilization_model = HeadStabilizationInferenceWrapper(
    model_path=model_dir / "all_dofs_model.ckpt",
    scaler_param_path=model_dir / "joint_angle_scaler_params.pkl",
)


def fit_line(pt0, pt1):
    rise = pt1[1] - pt0[1]
    run = pt1[0] - pt0[0]
    slope = rise / run
    intercept = pt0[1] - pt0[0] * slope
    return lambda x: slope * x + intercept


class NMFNavigation(gym.Env):
    def __init__(
        self,
        arena_factory: Callable[[], BaseArena],
        vision_model: VisualFeaturePreprocessor,
        ommatidia_graph,
        device="cpu",
        obj_threshold=50,
        decision_dt=0.05,
        n_stabilisation_dur=0.3,
        max_time=5,
        test_mode=False,
        debug_mode=False,
        spawn_x_range=(-2.5, 2.5),
        spawn_orient_range=(np.pi / 2 - np.deg2rad(10), np.pi / 2 + np.deg2rad(10)),
        descending_range=(0.2, 1),
        obs_margin_m=4,
        tgt_margin_epsilon=2,
        tgt_margin_q=3,
        fly_obs_dist_horizon=10,
        render_camera="birdeye_cam",
        render_playspeed=0.5,
        vision_refresh_rate=None,
        **kwargs,
    ) -> None:
        if vision_refresh_rate is None:
            vision_refresh_rate = int(1 / decision_dt)
        self.debug_mode = debug_mode

        self.vision_model = vision_model.to(device)
        self.ommatidia_graph_l = ommatidia_graph.to(device).clone()
        self.ommatidia_graph_r = ommatidia_graph.to(device).clone()
        self.device = device
        self.spawn_y_range = spawn_x_range
        self.spawn_orient_range = spawn_orient_range
        self.arena_factory = arena_factory
        self.n_stabilisation_dur = n_stabilisation_dur
        self.obs_margin_m = obs_margin_m
        self.tgt_margin_epsilon = tgt_margin_epsilon
        self.fly_obs_dist_horizon = fly_obs_dist_horizon
        self.tgt_margin_q = tgt_margin_q
        self.controller_kwargs = kwargs
        self.arena = self.arena_factory()
        self.test_mode = test_mode

        contact_sensor_placements = [
            f"{leg}{segment}"
            for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for segment in [
                "Tibia",
                "Tarsus1",
                "Tarsus2",
                "Tarsus3",
                "Tarsus4",
                "Tarsus5",
            ]
        ]

        self.contact_sensor_placements = contact_sensor_placements

        self.fly = Fly(
            name="0",
            enable_adhesion=True,
            draw_adhesion=True,
            enable_vision=True,
            actuator_kp=45,
            adhesion_force=40,
            render_raw_vision=test_mode,
            enable_olfaction=True,
            vision_refresh_rate=vision_refresh_rate,
            head_stabilization_model=head_stabilization_model,
            neck_kp=1000,
            detect_flip=True,
            contact_sensor_placements=list(contact_sensor_placements),
        )

        self.cam = Camera(
            fly=self.fly,
            camera_id=render_camera,
            fps=30,
            play_speed=render_playspeed,
        )

        self.controller = HybridTurningNMF(
            fly=self.fly,
            cameras=[self.cam],
            arena=self.arena,
            timestep=1e-4,
            **self.controller_kwargs,
        )

        self.controller.reset(seed=0)

        self.descending_range = descending_range
        self.vision_hist = []
        self.odor_hist = []
        self._x_pos_hist = []
        self._back_camera_x_offset = self.arena.back_cam.pos[0]

        self.max_time = max_time
        self.num_substeps = int(decision_dt / self.controller.timestep)
        self.obj_threshold = obj_threshold

        # Override spaces
        # action space: 2D vector of amplitude and phase for oscillators on each side
        # observation space:
        #  - 2D vector of x-y position of object relative to the fly, norm. to [0, 1]
        #  - scalar probability that there is an object in view, [0, 1]
        #  - 2D vector of mean odor intensity on each side, norm. to [0, 1]
        #  - 2D vector of current oscillator amp. on each side, norm. to [0, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,))

        raw_obs = self.controller.get_observation()
        fly_pos = raw_obs["fly"][0, :2]
        fly_heading = raw_obs["fly"][2, 0] - np.pi / 2
        obs_pos = self.arena.obstacle_positions[0]  # assuming there's only one here
        tgt_pos = self.arena.odor_source[0, :2]
        self._last_fly_tgt_dist = np.linalg.norm(fly_pos - tgt_pos)
        (
            self._last_score_obs_heading,
            self._last_score_tgt_heading,
        ) = self._calc_heading_score(
            fly_pos=fly_pos,
            obs_pos=obs_pos,
            tgt_pos=tgt_pos,
            fly_heading=fly_heading,
        )

        # Compute x-y position of each ommatidium
        retina = self.fly.retina
        self.coms = np.empty((retina.num_ommatidia_per_eye, 2))
        for i in range(retina.num_ommatidia_per_eye):
            mask = retina.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)

    def turn_bias_to_descending_signal(self, turn_bias):
        descending_span = self.descending_range[1] - self.descending_range[0]
        descending_signal = np.ones((2,)) * self.descending_range[1]
        if turn_bias < 0:
            descending_signal[0] -= np.abs(turn_bias) * descending_span
        else:
            descending_signal[1] -= np.abs(turn_bias) * descending_span
        return descending_signal

    def _calc_heading_score(
        self,
        fly_pos,
        obs_pos,
        tgt_pos,
        fly_heading,
    ):
        obs_dir = np.arctan2(obs_pos[1] - fly_pos[1], obs_pos[0] - fly_pos[0])
        tgt_dir = np.arctan2(tgt_pos[1] - fly_pos[1], tgt_pos[0] - fly_pos[0])
        obs_dir_rel = obs_dir - fly_heading
        tgt_dir_rel = tgt_dir - fly_heading
        fly_obs_dist = np.linalg.norm(fly_pos - obs_pos)
        fly_tgt_dist = np.linalg.norm(fly_pos - tgt_pos)
        obs_ang_radius = np.arctan2(self.arena.obstacle_radius, fly_obs_dist)
        tgt_ang_radius = np.arctan2(self.tgt_margin_epsilon, fly_tgt_dist)

        func_obs_heading = fit_line([0, 1], [self.obs_margin_m * obs_ang_radius, 0])
        score_obs_heading = func_obs_heading(np.abs(obs_dir_rel))
        score_obs_heading = np.clip(score_obs_heading, 0, 1)
        func_tgt_heading = fit_line(
            [tgt_ang_radius, 1], [self.tgt_margin_q * tgt_ang_radius, 0]
        )
        score_tgt_heading = func_tgt_heading(np.abs(tgt_dir_rel))
        score_tgt_heading = np.clip(score_tgt_heading, 0, 1)

        return score_obs_heading, score_tgt_heading

    def step(self, turn_bias):
        ## Step physics =====
        turning_signal = self.turn_bias_to_descending_signal(turn_bias)
        try:
            obstacle_contact_counter = 0
            for i in range(self.num_substeps):
                raw_obs, _, raw_term, raw_trunc, raw_info = self.controller.step(
                    turning_signal
                )
                collision_forces = [
                    np.abs(
                        self.controller.physics.named.data.cfrc_ext[f"obstacle_{j}"]
                    ).sum()
                    for j in range(len(self.arena.obstacle_positions))
                ]
                if np.sum(collision_forces) > 1:
                    obstacle_contact_counter += 1
                back_cam = self.controller.arena.back_cam
                # print(back_cam.pos)
                # back_cam.pos[0] = raw_obs["fly"][0, 0]
                self._x_pos_hist.append(raw_obs["fly"][0, 0])
                curr_cam_x_pos = back_cam.pos[0]
                if len(self._x_pos_hist) < 400:
                    smoothed_fly_pos = 0
                else:
                    smoothed_fly_pos = np.median(self._x_pos_hist[-800:])
                back_cam_x = (
                    max(curr_cam_x_pos, smoothed_fly_pos) + self._back_camera_x_offset
                )
                self.controller.physics.bind(back_cam).pos[0] = back_cam_x
                render_res = self.controller.render()[0]

                # if render_res is not None:
                #     import matplotlib.pyplot as plt
                #     plt.imshow(render_res)
                #     plt.show()
                #     assert False
                if render_res is not None:
                    self.odor_hist.append(raw_obs["odor_intensity"].copy())
                    self.vision_hist.append(raw_obs["vision"].copy())
        except PhysicsError:
            print("Physics error, resetting environment")
            return np.zeros((10,), dtype="float32"), 0, False, True, {}

        ## Verify state of physics simulation =====
        # check if visual inputs are rendered recently
        time_since_update = (
            self.controller.curr_time - self.controller.fly._last_vision_update_time
        )
        assert time_since_update >= 0
        # assert time_since_update < 0.25 * self.controller.timestep or np.isinf(
        #     self.fly._last_vision_update_time
        # )

        # check if the fly state
        has_collided = obstacle_contact_counter > 20
        has_flipped = raw_info["flip"]

        ## Fetch variables for reward and obs calculation =====
        fly_pos = raw_obs["fly"][0, :2]
        obs_pos = self.arena.obstacle_positions[0]  # assuming there's only one here
        tgt_pos = self.arena.odor_source[0, :2]
        fly_heading = raw_obs["fly"][2, 0] - np.pi / 2
        obs_dir = np.arctan2(obs_pos[1] - fly_pos[1], obs_pos[0] - fly_pos[0])
        tgt_dir = np.arctan2(tgt_pos[1] - fly_pos[1], tgt_pos[0] - fly_pos[0])
        obs_dir_rel = obs_dir - fly_heading
        tgt_dir_rel = tgt_dir - fly_heading
        fly_obs_dist = np.linalg.norm(fly_pos - obs_pos)
        fly_tgt_dist = np.linalg.norm(fly_pos - tgt_pos)
        obs_ang_radius = np.arctan2(self.arena.obstacle_radius, fly_obs_dist)
        tgt_ang_radius = np.arctan2(self.tgt_margin_epsilon, fly_tgt_dist)

        ## Calculate tentative costs
        func_obs_heading = fit_line([0, 1], [self.obs_margin_m * obs_ang_radius, 0])
        score_obs_heading = func_obs_heading(np.abs(obs_dir_rel))
        score_obs_heading = np.clip(score_obs_heading, 0, 1)
        func_tgt_heading = fit_line(
            [tgt_ang_radius, 1], [self.tgt_margin_q * tgt_ang_radius, 0]
        )
        score_tgt_heading = func_tgt_heading(np.abs(tgt_dir_rel))
        score_tgt_heading = np.clip(score_tgt_heading, 0, 1)
        score_obs_heading_2, score_tgt_heading_2 = self._calc_heading_score(
            fly_pos, obs_pos, tgt_pos, fly_heading
        )  # some refactorign needed
        assert score_obs_heading == score_obs_heading_2
        assert score_tgt_heading == score_tgt_heading_2

        ## Calculate reward and termination/truncation state =====
        k_dist = 1
        k_avoid = 7
        k_attract = 10 if fly_pos[0] > obs_pos[0] else 1
        r_success = 10
        r_fail = -5
        r_dist = k_dist * (self._last_fly_tgt_dist - fly_tgt_dist)
        p_avoid = k_avoid * (score_obs_heading - self._last_score_obs_heading)
        r_attract = k_attract * (score_tgt_heading - self._last_score_tgt_heading)

        # decide final reward and terminating states by case
        info = {}
        if fly_tgt_dist < self.tgt_margin_epsilon:
            reward = r_success
            terminated = True
            info["state_desc"] = "success"
        elif has_collided:
            reward = r_fail
            terminated = True
            info["state_desc"] = "collision"
        elif has_flipped:
            reward = r_fail
            terminated = True
            info["state_desc"] = "flipped"
        else:
            reward = r_dist + r_attract - p_avoid
            terminated = False
            info["state_desc"] = "seeking"

        # decide timeout condition
        if self.controller.curr_time > self.max_time and not terminated:
            truncated = True
            info["state_desc"] = "timeout"
        else:
            truncated = False

        # Make observation =====
        fly_obs_dist_norm = np.clip(fly_obs_dist / self.fly_obs_dist_horizon, 0, 1)
        obs_dir_rel_norm = np.clip((obs_dir_rel % (2 * np.pi)) / (2 * np.pi), 0, 1)
        turn_bias_norm = turn_bias[0] / 2 + 0.5
        visual_features = self._get_visual_features()
        odor_intensity = np.average(
            raw_obs["odor_intensity"][0, :].reshape(2, 2), axis=0, weights=[9, 1]
        )
        odor_intensity /= self.arena.peak_odor_intensity[0, 0]
        odor_intensity = np.clip(np.sqrt(odor_intensity), 0, 1)
        obs = np.array(
            [*visual_features, *odor_intensity, turn_bias_norm],
            dtype=np.float32,
        )

        ## Update state =====
        self._last_score_obs_heading = score_obs_heading
        self._last_score_tgt_heading = score_tgt_heading
        self._last_fly_tgt_dist = fly_tgt_dist

        ## Prepare debugging info =====
        info["fly_pos"] = fly_pos
        info["obs_pos"] = obs_pos
        info["tgt_pos"] = tgt_pos
        info["fly_heading"] = fly_heading
        info["obs_dir"] = obs_dir
        info["obs_dir_rel"] = obs_dir_rel
        info["tgt_dir"] = tgt_dir
        info["tgt_dir_rel"] = tgt_dir_rel
        info["fly_obs_dist"] = fly_obs_dist
        info["fly_tgt_dist"] = fly_tgt_dist
        info["score_obs_heading"] = score_obs_heading
        info["score_tgt_heading"] = score_tgt_heading
        info["r_dist"] = r_dist
        info["p_avoid"] = p_avoid
        info["r_attract"] = r_attract
        info["r_total"] = reward
        info["visual_features"] = [
            2 * np.deg2rad(270 / 2) * (visual_features[0] - 0.5),
            visual_features[1] * self.fly_obs_dist_horizon,
            visual_features[2],
            *visual_features[3:],
        ]
        info["odor_intensity"] = odor_intensity
        info["turn_bias"] = turn_bias
        info["terminated"] = terminated
        info["truncated"] = truncated
        if self.debug_mode:
            print("=======================")
            for k, v in info.items():
                print(f"  * {k}: {v}")

        return obs, reward, terminated, truncated, info

    def reset(self, seed=0, spawn_pos=None, spawn_orient=None):
        if self.spawn_y_range is not None and spawn_pos is None:
            spawn_pos = np.array([0, np.random.uniform(-5, 5), 0.2])
        if self.spawn_orient_range is not None and spawn_orient is None:
            spawn_yaw = np.random.uniform(
                self.spawn_orient_range[0], self.spawn_orient_range[1]
            )
            spawn_orient = np.array([0, 0, spawn_yaw])
        kwargs = copy.deepcopy(self.controller_kwargs)
        if spawn_pos is not None:
            kwargs["spawn_pos"] = spawn_pos
        if spawn_orient is not None:
            kwargs["spawn_orient"] = spawn_orient
        self.controller.close()
        self.arena = self.arena_factory()

        self.fly = Fly(
            name="0",
            enable_adhesion=True,
            draw_adhesion=True,
            enable_vision=True,
            actuator_kp=45,
            adhesion_force=40,
            render_raw_vision=self.test_mode,
            enable_olfaction=True,
            vision_refresh_rate=self.fly.vision_refresh_rate,
            head_stabilization_model=head_stabilization_model,
            neck_kp=1000,
            detect_flip=True,
            contact_sensor_placements=list(self.contact_sensor_placements),
            spawn_pos=spawn_pos,
            spawn_orientation=spawn_orient,
        )

        self.cam = Camera(
            fly=self.fly,
            camera_id=self.cam.camera_id,
            fps=30,
            play_speed=self.cam.play_speed,
        )

        self.controller = HybridTurningNMF(
            fly=self.fly,
            cameras=[self.cam],
            arena=self.arena,
            timestep=1e-4,
            **self.controller_kwargs,
        )

        self.controller.reset(seed=seed)

        self.odor_hist = []
        self.vision_hist = []
        self._x_pos_hist = []
        self.cam._frames = []

        obs = np.zeros((10,), dtype="float32")

        raw_obs = self.controller.get_observation()
        fly_pos = raw_obs["fly"][0, :2]
        fly_heading = raw_obs["fly"][2, 0] - np.pi / 2
        obs_pos = self.arena.obstacle_positions[0]  # assuming there's only one here
        tgt_pos = self.arena.odor_source[0, :2]
        self._last_fly_tgt_dist = np.linalg.norm(fly_pos - tgt_pos)
        (
            self._last_score_obs_heading,
            self._last_score_tgt_heading,
        ) = self._calc_heading_score(
            fly_pos=fly_pos,
            obs_pos=obs_pos,
            tgt_pos=tgt_pos,
            fly_heading=fly_heading,
        )
        if self.debug_mode:
            print("resetting environment")
        return obs, {"state_desc": "reset"}

    def _get_visual_features(self):
        intensities = self.controller.get_observation()["vision"]
        self.ommatidia_graph_l.x = torch.tensor(intensities[0, :, :]).to(self.device)
        self.ommatidia_graph_l.x = self.ommatidia_graph_l.x.float()
        self.ommatidia_graph_r.x = torch.tensor(intensities[1, :, :]).to(self.device)
        self.ommatidia_graph_r.x = self.ommatidia_graph_r.x.float()
        model_pred = self.vision_model(self.ommatidia_graph_l, self.ommatidia_graph_r)
        angle = model_pred["angle"].detach().cpu().numpy().squeeze()
        angle = 0.5 + np.clip(angle / np.deg2rad(270 / 2), -1, 1) / 2
        dist = model_pred["dist"].detach().cpu().numpy().squeeze()
        presence_logit = model_pred["mask"].detach().cpu().numpy().squeeze()
        azimuth = model_pred["azimuth"].detach().cpu().numpy().squeeze()
        rel_size = model_pred["rel_size"].detach().cpu().numpy().squeeze()

        # Manually calculated features
        if self.debug_mode:
            features = np.zeros((2, 3))
            for i, ommatidia_readings in enumerate(intensities):
                is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold
                is_obj[
                    np.arange(is_obj.size) % 2 == 1
                ] = False  # only use pale-type ommatidia
                is_obj_coords = self.coms[is_obj]
                if is_obj_coords.shape[0] > 0:
                    features[i, :2] = is_obj_coords.mean(axis=0)
                features[i, 2] = is_obj_coords.shape[0]

            retina = self.fly.retina
            features[:, 0] /= retina.nrows  # normalize y_center
            features[:, 1] /= retina.ncols  # normalize x_center
            # features[:, :2] = features[:, :2] * 2 - 1  # center around 0
            features[:, 2] /= retina.num_ommatidia_per_eye  # normalize area
            print(
                f"  ! Ly={azimuth[0]:.2f}({features[0, 1]:.2f})  "
                f"Ry={azimuth[1]:.2f}({features[1, 1]:.2f})"
            )
            print(
                f"  ! Ls={rel_size[0]:.2f}({features[0, 2]:.2f})  "
                f"Rs={rel_size[1]:.2f}({features[1, 2]:.2f})"
            )

        visual_features = np.array(
            [angle, dist, presence_logit, *azimuth, *rel_size], dtype=np.float32
        )
        return visual_features
