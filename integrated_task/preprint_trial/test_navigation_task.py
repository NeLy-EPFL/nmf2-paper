import pickle
import numpy as np
import stable_baselines3.common.env_checker as env_checker
import torch_geometric as pyg
from pathlib import Path
from tqdm import trange

from flygym.arena import MixedTerrain
from flygym.examples.obstacle_arena import ObstacleOdorArena
from rl_navigation import NMFNavigation
from vision_model import VisualFeaturePreprocessor


base_dir = Path("./")

## Load vision model =====
vision_model_path = base_dir / "../data/vision/visual_preprocessor.pt"
vision_model = VisualFeaturePreprocessor.load_from_checkpoint(vision_model_path).cpu()
ommatidia_graph_path = base_dir / "../data/vision/ommatidia_graph.pkl"
with open(ommatidia_graph_path, "rb") as f:
    ommatidia_graph_nx = pickle.load(f)
ommatidia_graph = pyg.utils.from_networkx(ommatidia_graph_nx).cpu()


def make_arena():
    terrain_arena = MixedTerrain(height_range=(0.3, 0.3), gap_width=0.2, ground_alpha=1)
    odor_arena = ObstacleOdorArena(
        terrain=terrain_arena,
        obstacle_positions=np.array([(7.5, 0)]),
        obstacle_radius=1,
        odor_source=np.array([[15, 0, 2]]),
        marker_size=0.5,
        obstacle_colors=(0, 0, 0, 1),
    )
    return odor_arena


sim = NMFNavigation(
    # spawn_pos=(7.5, 10, 0.2),
    # spawn_orient=(0, 0, np.pi / 2),
    vision_model=vision_model,
    ommatidia_graph=ommatidia_graph,
    arena_factory=make_arena,
    test_mode=True,
    debug_mode=True,
    decision_dt=0.1,
)

obs_hist = []
reward_hist = []
action_hist = []
obs, info = sim.reset(spawn_pos=(-0.2, -5, 0.2), spawn_orient=(0, 0, np.pi / 2))
for i in trange(20):
    action = np.array([0 if i < 10 else 0.6])
    action_hist.append(action)
    obs, reward, terminated, truncated, info = sim.step(action)
    obs_hist.append(obs)
    reward_hist.append(reward)
    if terminated:
        print("terminated")
        break

obs_hist = np.array(obs_hist)
reward_hist = np.array(reward_hist)
action_hist = np.array(action_hist)

sim.cam.save_video(base_dir / "test.mp4")

env_checker.check_env(sim)


# Run model
# sim = NMFNavigation(
#     arena_factory=make_arena,
#     vision_model=vision_model,
#     ommatidia_graph=ommatidia_graph,
#     test_mode=True,
#     debug_mode=True,
# )

# np.random.seed(0)
# sb3.common.utils.set_random_seed(0, using_cuda=True)
# start_from = base_dir / "logs/trial_c17_167200_steps.zip"
# train = False

# log_dir = "logs/"
# checkpoint_callback = callbacks.CheckpointCallback(
#     save_freq=100,
#     save_path=log_dir,
#     name_prefix="trial_8",
#     save_replay_buffer=True,
#     save_vecnormalize=True,
#     verbose=2,
# )
# my_logger = logger.configure(log_dir, ["tensorboard", "stdout", "csv"])
# model = sb3.SAC(
#     "MlpPolicy",
#     # env=sim,
#     env=sim,
#     policy_kwargs={"net_arch": [16, 16]},
#     verbose=2,
#     learning_rate=0.01,
# )
# if start_from is not None:
#     model = sb3.SAC.load(start_from)
# model.set_logger(my_logger)

# if train:
#     model.learn(total_timesteps=50_000, progress_bar=True, callback=checkpoint_callback)
#     model.save("models/trial_8")

# obs_hist = []
# reward_hist = []
# action_hist = []
# obs, info = sim.reset(spawn_pos=[-0.2, 0, 0.2], spawn_orient=(0, 0, np.pi / 2))
# for i in trange(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = sim.step(action)
#     if info["fly_tgt_dist"] < 3:
#         print("within 3mm")
#         break
#     action_hist.append(action)
#     obs_hist.append(obs)
#     reward_hist.append(reward)
#     if terminated:
#         print("terminated")
#         break

# obs_hist = np.array(obs_hist)
# reward_hist = np.array(reward_hist)
# action_hist = np.array(action_hist)

# sim.controller.save_video(base_dir / "outputs" / (Path(start_from).stem + ".mp4"))

# # fig, axs = plt.subplots(3, 2, figsize=(8, 9), tight_layout=True)
# # axs[0, 0].plot(obs_hist[:, 0])
# # axs[0, 0].plot(obs_hist[:, 1])
# # axs[0, 1].plot(obs_hist[:, 2])
# # axs[0, 1].plot(obs_hist[:, 3])
# # axs[1, 0].plot(obs_hist[:, 4])
# # axs[1, 0].plot(obs_hist[:, 5])
# # axs[1, 1].plot(action_hist)
# # axs[1, 1].set_ylim(-1, 1)
# # axs[2, 0].plot(reward_hist)
# # plt.savefig(base_dir / "outputs" / (Path(start_from).stem + ".png"))

# individual_frames_dir = Path("outputs/individual_frames")
# individual_frames_dir.mkdir(parents=True, exist_ok=True)

# snapshot_interval_frames = 30

# snapshots = np.array(
#     [sim.controller._frames[i] for i in range(0, len(sim.controller._frames), snapshot_interval_frames)]
# )
# background = np.median(snapshots, axis=0)

# for i in trange(0, snapshots.shape[0]):
#     img = snapshots[i, :, :, :]
#     is_background = np.isclose(img, background, atol=1).all(axis=2)
#     img_alpha = np.ones((img.shape[0], img.shape[1], 4)) * 255
#     img_alpha[:, :, :3] = img
#     img_alpha[is_background, 3] = 0
#     img_alpha = img_alpha.astype(np.uint8)
#     # break
#     imageio.imwrite(
#         individual_frames_dir / f"frame_{i}.png", img_alpha
#     )

# imageio.imwrite(individual_frames_dir / "background.png", background.astype(np.uint8))


# for trial in range(11):
#     start_y = np.arange(-10, 12, 2)[trial]
#     obs, info = sim.reset(spawn_pos=[-0.2, start_y, 0.2], spawn_orient=(0, 0, np.pi / 2))
#     for i in trange(100):
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = sim.step(action)
#         if info["fly_tgt_dist"] < 3:
#             print("within 3mm")
#             break
#         if terminated:
#             print("terminated")
#             break

#     obs_hist = np.array(obs_hist)
#     reward_hist = np.array(reward_hist)
#     action_hist = np.array(action_hist)

#     sim.controller.save_video(base_dir / "outputs" / (Path(start_from).stem + f"_trial{trial}.mp4"))
