import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from tqdm import trange
from flygym.util.config import all_leg_dofs
from flygym.state import stretched_pose

from flygym.util.turning_controller import TurningController
from flygym.util.cpg_controller import plot_phase_amp_output_rules, sine_output

from dm_control.rl.control import PhysicsError

import multiprocessing

import pickle

#### CONSTANTS ####
TIMESTEP = 1e-4
RUN_TIME = 1.0
ACTUATOR_KP = 30
STABILISATION_DUR = 0.2

SLOPE_REVERSAL_TIME = 0.4

def initialize_nmf(gain):
    # Initialize the simulation
    sim_params = MuJoCoParameters(
        timestep=TIMESTEP,
        #render_mode="saved",
        render_mode="headless",
        render_camera="Animat/camera_right",
        render_playspeed=0.1,
        actuator_kp=ACTUATOR_KP,
        enable_adhesion=True,
        draw_adhesion=True,
        adhesion_gain=gain,
        align_camera_with_gravity =True
    )

    nmf = TurningController(
        sim_params=sim_params,
        init_pose=stretched_pose,
        actuated_joints=all_leg_dofs,
        spawn_pos = [0, 0, 0.2],
        stabilisation_dur = STABILISATION_DUR
    )
    return nmf

def run_slope_CPG(nmf, slope, seed, num_steps=int(RUN_TIME/TIMESTEP)):
    np.random.seed(seed)
    obs, _ = nmf.reset()
    _, _, obs_list = nmf.run_stabilisation()
    action = [1.0, 1.0]
    for _ in range(num_steps):
        try:
            obs, _, _, _, _ = nmf.step(action)
        except PhysicsError:
            break
        obs_list.append(obs)
        
        #_ = nmf.render()
        if np.isclose(nmf.curr_time, SLOPE_REVERSAL_TIME, TIMESTEP/2):
            #print("Reversing slope:", slope, nmf.sim_params.adhesion_gain)
            nmf.set_slope(slope, "y")
    return nmf, obs_list

def run_experiment(gain, slope, seed, output_path, filename_template, metadata):

    gain_folder = output_path / f"seed_{seed}/gain_{gain}"
    gain_folder.mkdir(parents=True, exist_ok=True)

    with open(output_path / filename_template.format(seed, gain, "metadata", ".pkl"), "wb") as f:
        pickle.dump(metadata, f)

    #video_path = output_path / filename_template.format(seed, gain, slope, ".mp4")
    pkl_path = output_path / filename_template.format(seed, gain, slope, ".pkl")
    if  pkl_path.exists(): #and video_path.exists():
        print("pkl already exists:", pkl_path)
        return
    
    print("Running experiment with gain:", gain, "slope:", slope, "seed:", seed)

    nmf = initialize_nmf(gain)
    nmf, obs_list = run_slope_CPG(nmf, slope, seed)

    #nmf.save_video(video_path)
    # save the data
    with open(pkl_path, "wb") as f:
        pickle.dump(obs_list, f)

    return obs_list


if __name__ == "__main__":
    #slopes_in_degrees = [0, 30, 60, 90, 120][::-1]
    slopes_in_degrees = np.arange(0, 121, 5)[::-1]
    #gains = [0.0, 10.0, 20.0]
    gains = np.arange(0.0, 61.0, 5.0)

    seeds = [0, 1, 2, 3, 4]

    output_path = Path("datapts_gainslope")
    filename_template = "seed_{}/gain_{}/slope_{}{}"

    # Change this loop to multiprocessing
    for seed in seeds:
        metadata = {"timestep":TIMESTEP,
                    "run_time":RUN_TIME,
                    "actuator_kp":ACTUATOR_KP,
                    "stabilisation_dur":STABILISATION_DUR,
                    "seed":seed,
                    "slope_reversal_time":SLOPE_REVERSAL_TIME}
        

        conditions = [(gain, slope, seed, output_path, filename_template, metadata) 
                      for gain in gains for slope in slopes_in_degrees]
        with multiprocessing.Pool(4) as pool:
            obs_lists = pool.starmap(run_experiment, conditions)

        pool.join()
        pool.close()

