import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import trange

from visual_taxis import NMFVisualTaxis, MovingObjArena
from flygym.envs.nmf_mujoco import MuJoCoParameters

from util import linear_schedule, SaveIntermediateModelsCallback
from tqdm import trange
from flygym.util.config import all_leg_dofs

from stable_baselines3 import PPO

decision_dt = 0.05

arena = MovingObjArena(obj_spawn_pos=(5, 3, 0), move_mode="s_shape", move_speed=15)
arena.reset(new_spawn_pos=True, new_move_mode=False, new_move_speed=25)
spawn = arena.init_ball_pos
sim_params = MuJoCoParameters(render_playspeed=0.2, render_camera="Animat/camera_top_zoomout", render_raw_vision=True)
sim = NMFVisualTaxis(
    sim_params=sim_params,
    arena=arena,
    decision_dt=decision_dt,
    n_stabilisation_steps=5000,
    obj_threshold=50,
)

dr = f'../logs_orient_'
model_name = "saved_model_MLPlinearlr_continue"
out_dir = Path(dr+f"/vis_{model_name}")
out_dir.mkdir(parents=True, exist_ok=True)


n_base = int(2/decision_dt)

# Load model from file
nmf_model = PPO.load(dr+"/"+model_name)

print(nmf_model.policy)

obs, _ = sim.reset()
obs_hist = []
visual_hist = []
action_history = []
rewards_hist = []

for i in trange(n_base):
    action, _ = nmf_model.predict(obs, deterministic=True)
    obs, rew, term, trunc, info = sim.step(np.array(action))
    sim.render()
    obs_hist.append(obs)
    visual_hist.append(sim.curr_raw_visual_input)
    action_history.append(action)
    rewards_hist.append(rew)

obs_hist = np.array(obs_hist)
sim.save_video("test.mp4")

print(action)
print(rewards_hist[-1])
plt.plot(np.arange(n_base), np.cumsum(rewards_hist))
plt.savefig(out_dir / f"{spawn}totrewards.png")
plt.show()
plt.plot(np.arange(n_base), rewards_hist)
plt.savefig(out_dir / f"{spawn}rewards.png")
plt.show()
plt.plot(np.arange(n_base), action_history)
plt.savefig(out_dir / f"{spawn}actions.png")
plt.show()
sim.save_video(out_dir / f'{spawn}resultcust.mp4')
sim.close()