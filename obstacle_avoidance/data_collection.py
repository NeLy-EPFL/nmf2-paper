import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from flygym.envs.nmf_mujoco import MuJoCoParameters
from flygym.arena.mujoco_arena import FlatTerrain
import flygym.util.vision as vision
from odor_vision import ObstacleOdorArena, NMFObservation

arena = ObstacleOdorArena()
sim_params = MuJoCoParameters(render_playspeed=0.2, render_camera="Animat/camera_top_zoomout", render_raw_vision=True, enable_olfaction=True)

sim = NMFObservation(
    sim_params=sim_params,
    arena=arena,
    obj_threshold=50,
)

obs,_,_,_,_ = sim.step([0,0])
print(obs.shape)

sim.close()