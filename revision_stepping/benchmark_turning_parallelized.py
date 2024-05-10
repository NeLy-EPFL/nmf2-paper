import numpy as np
import matplotlib.pyplot as plt
import pickle
import flygym.mujoco
import flygym.mujoco.preprogrammed
from flygym.mujoco.examples.turning_controller import HybridTurningNMF
from flygym.mujoco.examples.common import PreprogrammedSteps
from flygym.common import get_data_path
import tqdm
from pathlib import Path
import cv2

from multiprocessing import Pool

###### CST #######

# define the turning behavior
run_time = 1.5
turning_duration = 0.5
straight_run_time = 0.5
post_turn_time = 0.2

# Defines the nmf parameters
adhesion = True
new_stiffness = 10.0
new_damping = 10.0
timestep = 1e-4
render_mode = "saved"
acutator_kp = 30.0

contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]
convergence_coef_val = 2.0

target_num_steps = int(run_time / timestep)

# define the step
use_old_data = True
old_step_path = get_data_path("flygym", "data") / "behavior/single_steps.pkl"

# define the visaulization parameters
arrow_scaling = 2
arrow_width = 0.1

# define the output folder
folder_name = "benchmark_turning_parallel"
folder_name += "_adhesion" if adhesion else ""
folder_name += f"_R={convergence_coef_val}"
folder_name += "_old" if use_old_data else ""
out_folder = Path(f"data/{folder_name}")
out_folder.mkdir(exist_ok=True, parents=True)

# Benchmark parameters
make_traj_plots = False
if make_traj_plots:
    out_traj_folder = out_folder / "traj_plots"
    out_traj_folder.mkdir(exist_ok=True, parents=True)
n_pts = 4
n_processes = 4

def initialize_nmf():

    if use_old_data:
        sim_params = flygym.mujoco.Parameters(
            timestep=1e-4, render_mode=render_mode, render_playspeed=0.1,
            draw_adhesion=adhesion, enable_adhesion=adhesion,
            actuator_gain=acutator_kp, tarsus_damping=0.05, tarsus_stiffness=2.2
            )
        preprogrammed_steps = PreprogrammedSteps(path=old_step_path, 
                                                 neutral_pos_indexes = np.zeros(6, dtype=int))
    else:
        sim_params = flygym.mujoco.Parameters(
            timestep=1e-4, render_mode=render_mode, render_playspeed=0.1,
            draw_adhesion=adhesion, enable_adhesion=adhesion, actuator_gain=acutator_kp
        )
        preprogrammed_steps = PreprogrammedSteps()
    
    nmf = HybridTurningNMF(
        xml = "mjcf_model" if use_old_data else "mjcf_ikpy_model",
        init_pose = "stretch",
        preprogrammed_steps=preprogrammed_steps,
        sim_params=sim_params,
        contact_sensor_placements=contact_sensor_placements,
        convergence_coefs=np.ones(6) * convergence_coef_val,
    )

    return nmf

# Experiment function
def run_experiment(k, turn_start_step, turn_end_step, turn_drive):
    print(f"Running experiment {k} out of {n_pts}")
    nmf = initialize_nmf()
    
    obs, info = nmf.reset(seed=42)
    
    turn_start_orientation = None
    turn_end_orientation = None
    RF_cpg_phase_turn_start = 0
    RF_leg_id = nmf.preprogrammed_steps.legs.index("RF")

    fly_pos = np.zeros((2, target_num_steps))
    fly_angs = np.zeros((3, target_num_steps))
    fly_orientations = np.zeros((2, target_num_steps))

    fly_pos[:, 0] = obs["fly"][0][:2].copy()
    fly_angs[:, 0] = obs["fly"][3].copy()
    fly_orientations[:, 0] = obs["fly_orientation"][:2].copy()

    rendered = None

    for i in range(1, target_num_steps):

        if i >= turn_start_step and i < turn_end_step:
            action = turn_drive
            if nmf.sim_params.render_mode and not rendered is None:
                cv2.circle(nmf._frames[-1], (60, 25), 20, (0, 0, 255), -1)
            
            if i == turn_start_step:
                turn_start_orientation = obs["fly_orientation"][:2].copy()
                RF_cpg_phase_turn_start = nmf.cpg_network.curr_phases[RF_leg_id] % (2*np.pi)
        
        else:
            action = np.array([1.0, 1.0])
            
            if i == turn_end_step:
                turn_end_orientation = obs["fly_orientation"][:2].copy()
        
        obs, reward, terminated, truncated, info = nmf.step(action)
        fly_pos[:, i] = obs["fly"][0][:2].copy()
        fly_orientations[:, i] = obs["fly_orientation"][:2].copy()
        fly_angs[:, i] = obs["fly"][2].copy()
        if nmf.sim_params.render_mode == "saved":
            rendered = nmf.render()
    
    if turn_end_orientation is None:
        turn_end_orientation = obs["fly_orientation"][:2].copy()
    
    turning_angle = (
        np.arctan2(turn_end_orientation[1], turn_end_orientation[0]) -
        np.arctan2(turn_start_orientation[1], turn_start_orientation[0])
                    )
    turning_angle_change = np.rad2deg(turning_angle)
    # save everything to a file
    with open(out_folder / f"turning_basic_{k}.pkl", "wb") as f:
        pickle.dump(
            {   "timestep": timestep,
                "run_time": run_time,
                "l_drive": turn_drive[0],
                "r_drive": turn_drive[1],
                "turn_start": turn_start_step,
                "turn_ends": turn_end_step,
                "cpg_phase_turn_start": RF_cpg_phase_turn_start,
                "turning_angle_change": turning_angle_change,
            },
            f,
        )
    with open(out_folder / f"turning_full_{k}.pkl", "wb") as f:
        pickle.dump(
            {
                "fly_pos": fly_pos.tolist(),
                "fly_angs": fly_angs.tolist(),
                "fly_orientation": fly_orientations.tolist(),
            },
            f
        )

    if nmf.sim_params.render_mode == "saved":
        nmf.save_video(out_folder / f"turning_{k}.mp4", stabilization_time=0)
    if make_traj_plots:
        save_traj_plots(k, fly_pos, turn_start_step, turn_end_step, turn_start_orientation, turn_end_orientation, turning_angle_change, turn_drive)


def save_traj_plots(k, fly_pos, turn_start_step, turn_end_step, turn_start_orientation, turn_end_orientation, turning_angle_change, turn_drive):
        fig = plt.figure()
        plt.plot(fly_pos[0, :], fly_pos[1, :], label="Trajectory")
        # add arrows for the base and end orientation
        plt.arrow(fly_pos[0, turn_start_step],
                    fly_pos[1, turn_start_step],
                    turn_start_orientation[0]*arrow_scaling, turn_start_orientation[1]*arrow_scaling,
                    color="green", width=arrow_width)
        plt.arrow(fly_pos[0, turn_end_step],
                    fly_pos[1, turn_end_step],
                    turn_end_orientation[0]*arrow_scaling, turn_end_orientation[1]*arrow_scaling,
                    color="red", width=arrow_width)
        plt.title(f"Turn {k} with drive {np.array2string(turn_drive, precision=3, floatmode='fixed')} and angle change {turning_angle_change:.3f}")
        plt.legend()
        plt.ylim([-16.0, 16.0])
        plt.xlim([-2.0, 30.0])
        plt.savefig(out_folder / f"traj_plots/trajectory_{k}.png", dpi=300)
        plt.close(fig)

def get_turning_params():
    np.random.seed(0)
    turn_start_times = np.random.uniform(straight_run_time, run_time - turning_duration - post_turn_time, size=n_pts)
    turn_end_times = turn_start_times + turning_duration
    turn_drives = np.random.uniform(-1, 1, size=(2, n_pts))
    turn_start_steps = np.floor(turn_start_times / timestep).astype(int)
    turn_end_steps = np.floor(turn_end_times /timestep).astype(int)
    return turn_start_steps, turn_end_steps, turn_drives


def main():
    if render_mode == "saved":
        assert n_pts < 20, "Too many simulations to run with render_mode='saved'"
    turn_start_steps, turn_end_steps, turn_drives = get_turning_params()


    arguments = [(k, turn_start_steps[k], turn_end_steps[k], turn_drives[:, k]) for k in range(n_pts)]
    with Pool(n_processes) as p:
        p.starmap(run_experiment, arguments)

    return None

if __name__ == "__main__":
    print("Running main")
    main()