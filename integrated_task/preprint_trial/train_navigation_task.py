import pickle
from pathlib import Path

import numpy as np
import stable_baselines3 as sb3
import stable_baselines3.common.callbacks as callbacks
import stable_baselines3.common.logger as logger
import torch_geometric as pyg
from flygym.arena import MixedTerrain
from flygym.examples.obstacle_arena import ObstacleOdorArena
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import trange

from rl_navigation import NMFNavigation
from vision_model import VisualFeaturePreprocessor


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


if __name__ == "__main__":
    ## Configs =====
    num_procs = 22
    base_dir = Path("/home/tlam/nmf2-paper/integrated_task/preprint_trial")

    ## Load vision model =====
    vision_model_path = base_dir / "../data/vision/visual_preprocessor.pt"
    vision_model = VisualFeaturePreprocessor.load_from_checkpoint(
        vision_model_path
    ).cpu()
    ommatidia_graph_path = base_dir / "../data/vision/ommatidia_graph.pkl"
    with open(ommatidia_graph_path, "rb") as f:
        ommatidia_graph_nx = pickle.load(f)
    ommatidia_graph = pyg.utils.from_networkx(ommatidia_graph_nx).cpu()

    ## Define MDP task =====
    def make_env():
        sim = NMFNavigation(
            arena_factory=make_arena,
            vision_model=vision_model,
            ommatidia_graph=ommatidia_graph,
            test_mode=False,
            debug_mode=False,
        )
        return sim

    ## Make vector environment =====
    print("Making vector env")
    vec_env = make_vec_env(make_env, n_envs=num_procs, vec_env_cls=SubprocVecEnv)
    print("Vector env created")

    ## Train model =====
    np.random.seed(0)
    sb3.common.utils.set_random_seed(0, using_cuda=True)

    start_from = None

    log_dir = str(base_dir / "logs/")
    checkpoint_callback = callbacks.CheckpointCallback(
        save_freq=100,
        save_path=log_dir,
        name_prefix=base_dir.name,
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=2,
    )
    my_logger = logger.configure(log_dir, ["tensorboard", "stdout", "csv"])
    if start_from is None:
        model = sb3.SAC(
            "MlpPolicy",
            # env=sim,
            env=vec_env,
            policy_kwargs={"net_arch": [32, 32]},
            verbose=2,
            learning_rate=0.01,
        )
    else:
        model = sb3.SAC.load(start_from)
        model.set_env(vec_env)
        print(model.verbose, model.learning_rate, model.policy_kwargs)
    model.set_logger(my_logger)

    print("Training start")
    model.learn(
        total_timesteps=500_000, progress_bar=True, callback=checkpoint_callback
    )
    model.save(str(base_dir / "data/rl/model"))
