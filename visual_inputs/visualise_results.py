import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import trange
import pickle

from visual_taxis import NMFVisualTaxis, MovingObjArena
from flygym.envs.nmf_mujoco import MuJoCoParameters

from util import linear_schedule, SaveIntermediateModelsCallback
from tqdm import trange
from flygym.util.config import all_leg_dofs

from stable_baselines3 import PPO

decision_dt = 0.05

arena = MovingObjArena(obj_spawn_pos=(5, 3, 0), move_mode="s_shape")
arena.reset(new_spawn_pos=True, new_move_mode=False)
spawn = arena.init_ball_pos
sim_params = MuJoCoParameters(render_playspeed=0.2, render_camera="Animat/camera_top_zoomout", render_raw_vision=True)
sim = NMFVisualTaxis(
    sim_params=sim_params,
    arena=arena,
    decision_dt=decision_dt,
    n_stabilisation_steps=5000,
    obj_threshold=50,
)

dr = f'../logs_'
model_name = "model15000"
out_dir = Path(dr+f"/eval_{model_name}")
out_dir.mkdir(parents=True, exist_ok=True)

n_base = int(2/decision_dt)

# Load model from file
nmf_model = PPO.load(dr+"/"+model_name)

print(nmf_model.policy)

distances = []
rew_straight = []
rew_s = []
folds = 20

for f in range(folds):
    print(f)
    obs, _ = sim.reset()
    spawn = sim.arena.init_ball_pos

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

    print(action)
    print(rewards_hist[-1])
    plt.plot(np.arange(n_base), np.cumsum(rewards_hist))
    plt.savefig(out_dir / f"{f}-{spawn}totrewards.png")
    plt.clf()
    plt.plot(np.arange(n_base), rewards_hist)
    plt.savefig(out_dir / f"{f}-{spawn}rewards.png")
    plt.clf()
    plt.plot(np.arange(n_base), action_history)
    plt.savefig(out_dir / f"{f}-{spawn}actions.png")
    plt.clf()
    sim.save_video(out_dir / f'{f}-{spawn}resultcust.mp4')

    # Save relevant metrics
    if sim.arena.move_mode == "straightHeading":
        rew_straight.append(rewards_hist)
    elif sim.arena.move_mode == "s_shape":
        rew_s.append(rewards_hist)

sim.close()

rewards = [np.array(rew_straight), np.array(rew_s)]
colors = [["tab:blue","tab:cyan"], ["orangered", "orange"], ["tab:green", "limegreen"]]
name = ["Straight", "S-shaped", "Both"]

# Saving results
with open(dr+"/rewards.pickle",'wb') as f:
    pickle.dump(rewards, f)
f.close()

print(rewards[0].shape, rewards[1].shape)

# Plotting mean and std of metrics across all runs
time = np.arange(rewards[0].shape[1])*0.05
for c in range(len(rewards)):
    col = colors[c]
    if len(rewards[c].shape)>1:
        mean_r = np.mean(rewards[c], axis=0)
        std_r = np.std(rewards[c], axis=0)
        plt.fill_between(time, mean_r-std_r, mean_r+std_r, color=col[1])
        plt.plot(time,mean_r, c=col[0])
        plt.title(name[c]+" trajectory")
        plt.xlabel("Time [s]")
        plt.ylabel("Instant reward")
        plt.legend()
        plt.savefig(out_dir / f"rewards_avg_std{c}")
        plt.clf()
        #plt.show()

for c in range(len(rewards)):
    col = colors[c]
    if len(rewards[c].shape)>1:
        cumul_rewards = np.cumsum(rewards[c], axis=1)
        mean_r = np.mean(cumul_rewards, axis=0)
        std_r = np.std(cumul_rewards, axis=0)
        plt.fill_between(time, mean_r-std_r, mean_r+std_r, color=col[1])
        plt.plot(time,mean_r, c=col[0], label=name[c])
plt.xlabel("Time [s]")
plt.ylabel("Cumulative reward")
plt.legend()
plt.savefig(out_dir / "cumulrewards_avg_std")
plt.clf()
#plt.show()


# For all runs combined
rewards_combined = [item for line in rewards for item in line]
rewards_combined = np.array(rewards_combined)
col = colors[2]

mean_r = np.mean(rewards_combined, axis=0)
std_r = np.std(rewards_combined, axis=0)
plt.fill_between(time, mean_r-std_r, mean_r+std_r, color=col[1])
plt.plot(time,mean_r, c=col[0])
plt.title(name[2]+" trajectories")
plt.xlabel("Time [s]")
plt.ylabel("Instant reward")
plt.legend()
plt.savefig(out_dir / f"rewards_avg_std")
plt.clf()
#plt.show()
