import numpy as np
import matplotlib.pyplot as plt
import pickle
import flygym.mujoco
import flygym.mujoco.preprogrammed
from flygym.mujoco.examples.turning_controller import HybridTurningNMF
from flygym.mujoco.examples.common import PreprogrammedSteps
from tqdm import trange
from pathlib import Path
import cv2


###### CST

debug = False
ADHESION = True
use_old_data = True
plot_first = True

arrow_scaling = 2
arrow_width = 0.1

psi_base_phase = np.pi
n_pts = 200
run_time = 1.5
turning_duration = 0.5
straight_run_time = 0.5
post_turn_time = 0.2
psi_base_phase = np.pi
contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]
conv_coef_scale = 2.0


def main():
    single_step_path = Path("data/single_step_datasets/RF_0swing_RM_0stance_LH_1stance.pkl")
    out_folder = Path("data/benchmark_turning")
    folder_name = "benchmark_turning"
    folder_name += "_adhesion" if ADHESION else ""
    folder_name += f"_R={conv_coef_scale}"
    folder_name += "_old" if use_old_data else ""
    out_folder = Path(f"data/{folder_name}")
    out_folder.mkdir(exist_ok=True, parents=True)

    if use_old_data:
        sim_params = flygym.mujoco.Parameters(
        timestep=1e-4, render_mode="saved", render_playspeed=0.1, draw_adhesion=ADHESION, enable_adhesion=ADHESION, actuator_kp=50
    )
    else:
        sim_params = flygym.mujoco.Parameters(
            timestep=1e-4, render_mode="saved", render_playspeed=0.1, draw_adhesion=ADHESION, enable_adhesion=ADHESION, actuator_kp=50, tarsus_damping=10.0, tarsus_stiffness=10.0
        )
    if use_old_data:
        preprogrammed_steps = PreprogrammedSteps()
    else:
        preprogrammed_steps = PreprogrammedSteps(path=single_step_path, 
                                                 neutral_pos_indexes = np.ones(6)*np.pi)
    
    #preprogrammed_steps.neutral_pos = {
    #        leg: preprogrammed_steps._psi_funcs[leg](psi_base_phase)[:, np.newaxis] for leg in preprogrammed_steps.legs
    #    }
    
    np.random.seed(0)
    turn_start_times = np.random.uniform(straight_run_time, run_time - turning_duration - post_turn_time, size=n_pts)
    turn_end_times = turn_start_times + turning_duration
    turn_drives = np.random.uniform(-1, 1, size=(2, n_pts))
    with open(out_folder / "turning_drive.pkl", "wb") as f:
        pickle.dump(turn_drives, f)

    turn_start_steps = np.floor(turn_start_times / sim_params.timestep).astype(int)
    turn_end_steps = np.floor(turn_end_times / sim_params.timestep).astype(int)

    turning_angle_change = np.zeros(n_pts)

    if debug:
        assert n_pts < 20, "Too many points to debug"

    if debug or plot_first:
        sim_params.render_mode = "saved"
        sim_params.render_playspeed = 0.1
        sim_params.render_camera = "Animat/camera_top"

    xml_path = "mjcf_model" if use_old_data else "mjcf_ikpy_model"
    
    nmf = HybridTurningNMF(
        xml = xml_path,
        preprogrammed_steps=preprogrammed_steps,
        sim_params=sim_params,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(0, 0, 0.2),
        convergence_coefs=np.ones(6) * conv_coef_scale
    )
    target_num_steps = int(run_time / nmf.sim_params.timestep)

    if debug or plot_first:
        out_folder_debug = out_folder / "debug"
        out_folder_debug.mkdir(exist_ok=True, parents=True)
        fly_pos = np.zeros((2, target_num_steps))
        rendered = False


    for k in range(n_pts):
            
        print(f"############## Running turn {k} ##############")
        
        obs, info = nmf.reset(seed=42)

        turn_start_orientation = None
        turn_end_orientation = None
        print(turn_start_steps[k], turn_end_steps[k], target_num_steps)

        for i in trange(target_num_steps):
            if i >= turn_start_steps[k] and i < turn_end_steps[k]:
                action = turn_drives[:, k]
                if (debug or (plot_first and k <= 10)) and not rendered is None:
                    cv2.circle(nmf._frames[-1], (60, 25), 20, (0, 0, 255), -1)
                    
                if i == turn_start_steps[k]:
                    turn_start_orientation = obs["fly_orientation"][:2].copy()
            else:
                action = np.array([1.0, 1.0])
                if i == turn_end_steps[k]:
                    turn_end_orientation = obs["fly_orientation"][:2].copy()
            obs, reward, terminated, truncated, info = nmf.step(action)
            if debug or (plot_first and k <= 10):
                fly_pos[:, i] = obs["fly"][0][:2].copy()
                rendered = nmf.render()
        
        if turn_end_orientation is None:
            turn_end_orientation = obs["fly_orientation"][:2].copy()
        
        turning_angle = (
            np.arctan2(turn_end_orientation[1], turn_end_orientation[0]) -
            np.arctan2(turn_start_orientation[1], turn_start_orientation[0])
                        )
        turning_angle_change[k] = np.rad2deg(turning_angle)

        if debug or (plot_first and k <= 10):
            fig = plt.figure()
            plt.plot(fly_pos[0], fly_pos[1], label="Trajectory")
            # add arrows for the base and end orientation
            plt.arrow(fly_pos[0][turn_start_steps[k]],
                        fly_pos[1][turn_start_steps[k]],
                        turn_start_orientation[0]*arrow_scaling, turn_start_orientation[1]*arrow_scaling,
                        color="green", width=arrow_width)
            plt.arrow(fly_pos[0][turn_end_steps[k]],
                        fly_pos[1][turn_end_steps[k]],
                        turn_end_orientation[0]*arrow_scaling, turn_end_orientation[1]*arrow_scaling,
                        color="red", width=arrow_width)
            plt.title(f"Turn {k} with drive {np.array2string(turn_drives[:, k], precision=3, floatmode='fixed')} and angle change {turning_angle_change[k]:.3f}")
            plt.legend()
            plt.ylim([-16.0, 16.0])
            plt.xlim([-2.0, 30.0])
            plt.savefig(out_folder_debug / f"trajectory_{k}.png", dpi=300)
            plt.close(fig)
            nmf.save_video(out_folder_debug / f"trajectory_{k}.mp4", stabilization_time=0)

        if not nmf.sim_params.render_mode == "headless" and k >= 10:
                sim_params.render_mode = "headless"
                nmf = HybridTurningNMF(
                                        xml = xml_path,
                                        preprogrammed_steps=preprogrammed_steps,
                                        sim_params=sim_params,
                                        contact_sensor_placements=contact_sensor_placements,
                                        spawn_pos=(0, 0, 0.2),
                                        convergence_coefs=np.ones(6) * conv_coef_scale
                                    )
            
    with open(out_folder / "turning_angle_change.pkl", "wb") as f:
        pickle.dump(turning_angle_change, f)
    
    fig = plt.figure()
    plt.scatter(turn_drives[0], turn_drives[1], c=turning_angle_change,
                cmap="viridis")
    plt.colorbar()
    plt.xlabel("Left drive")
    plt.ylabel("Right drive")
    plt.title("Turning angle change")
    if debug:
        # add numbers to the points
        for i in range(n_pts):
            plt.text(turn_drives[0, i]+0.01, turn_drives[1, i], str(i))
    plt.ylim([-1.1, 1.1])
    plt.xlim([-1.1, 1.1])
    plt.axvline(0, color="black", linestyle="--")
    plt.axhline(0, color="black", linestyle="--")
    #set colorbar axis limits [-360 360]
    plt.clim(-360, 360)
    #colorbar ticks
    plt.colorbar(ticks=np.linspace(-360, 360, 9), label="Turning angle change (degrees)")
    # change color to have white for 0 degrees
    plt.set_cmap('seismic')
    plt.savefig(out_folder / "turning_angle_change.png", dpi=300)
    return None

if __name__ == "__main__":
    print("Running main")
    main()