import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pickle
import os

from flygym.envs.nmf_mujoco import MuJoCoParameters
from flygym.arena.mujoco_arena import FlatTerrain
import flygym.util.vision as vision
from odor_vision import ObstacleOdorArena, NMFObservation


sim_params = MuJoCoParameters(render_playspeed=0.2, render_camera="Animat/camera_top_zoomout", render_raw_vision=True, enable_olfaction=True)

save_path = "../data"
if not os.path.exists(save_path):
   os.makedirs(save_path)

num_pos = 4000
steps = 2
pos_range = [[0,30],[-12,12]]

dataset = []

for f in trange(num_pos):
    arena = ObstacleOdorArena()
    spawn_pos = (np.random.randint(*pos_range[0]),np.random.randint(*pos_range[1]),0.5)
    spawn_orient = (0,0,1,np.random.random()*2*np.pi)
    sim = NMFObservation(
        sim_params=sim_params,
        arena=arena,
        obj_threshold=50,
        spawn_pos=spawn_pos,
        spawn_orient=spawn_orient,
        pos_range=pos_range
    )

    for i in range(steps):
        obs,_,_,_,_ = sim.step([0,0])
        dataset.append(obs)

    sim.close()

dataset = np.array(dataset)
with open(save_path+"/dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)
