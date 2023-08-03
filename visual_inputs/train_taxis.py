from stable_baselines3.common.logger import configure

from pathlib import Path

from flygym.envs.nmf_mujoco import MuJoCoParameters
from flygym.arena.mujoco_arena import FlatTerrain
from flygym.util.config import all_leg_dofs

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from visual_taxis import NMFVisualTaxis, MovingObjArena
from flygym.envs.nmf_mujoco import MuJoCoParameters

from util import linear_schedule, SaveIntermediateModelsCallback

arena = MovingObjArena(obj_spawn_pos=(5, 3, 0), move_mode="s_shape")
# sim_params = MuJoCoParameters(render_playspeed=0.2, render_camera="birdseye_cam")
sim_params = MuJoCoParameters(render_playspeed=0.2)
sim = NMFVisualTaxis(
    sim_params=sim_params,
    arena=arena,
    decision_dt=0.05,
    n_stabilisation_steps=5000,
    obj_threshold=50,
)

log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)

callback = SaveIntermediateModelsCallback(check_freq=20_000, log_dir=log_dir)

mynmf = Monitor(sim, filename=str(log_dir / f"train_log_MLP"))

new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])

nmf_model = PPO(MlpPolicy, mynmf, verbose=True, learning_rate=linear_schedule(0.003))

nmf_model.set_logger(new_logger)

print(nmf_model.policy)

nmf_model.learn(total_timesteps=400_000, progress_bar=True, callback=callback)
nmf_model.save(str(log_dir / f"saved_model_MLPlinearlr"))
mynmf.close()