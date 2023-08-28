import numpy as np
from pathlib import Path
import pickle
import pkg_resources

import argparse
import multiprocessing
import time

from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.util.config import all_leg_dofs
from flygym.state import stretched_pose

from flygym.util.cpg_controller import (
    advancement_transfer,
    phase_oscillator,
    initialize_solver,
    phase_biases_tripod_idealized,
)

import yaml

###### CONSTANTS ######
CONTROLLER_SEED = 42
STABILIZATION_DUR = 0.2
GRAVITY_SWITCHING_T = 0.4

LEGS = ["RF", "RM", "RH", "LF", "LM", "LH"] 
N_OSCILLATORS = len(LEGS)

COUPLING_STRENGTH = 10.0
AMP_RATES = 20.0
TARGET_AMPLITUDE = 1.0

RUN_TIME = 1.0

##### FUNCTIONS ######
def get_data_block(timestep, actuated_joints):
    data_path = Path(pkg_resources.resource_filename("flygym", "data"))
    with open(data_path / "behavior" / "single_steps.pkl", "rb") as f:
        data = pickle.load(f)
    # Interpolate 5x
    step_duration = len(data["joint_LFCoxa"])
    interp_step_duration = int(step_duration * data["meta"]["timestep"] / timestep)
    data_block = np.zeros((len(actuated_joints), interp_step_duration))
    measure_t = np.arange(step_duration) * data["meta"]["timestep"]
    interp_t = np.arange(interp_step_duration) * timestep
    for i, joint in enumerate(actuated_joints):
        data_block[i, :] = np.interp(interp_t, measure_t, data[joint])

    joint_ids = np.arange(len(actuated_joints)).astype(int)
    match_leg_to_joints = np.array(
        [i for joint in actuated_joints for i, leg in enumerate(LEGS) if leg in joint]
    )

    leg_swing_starts = {k:v/nmf.timestep for k,v in data["swing_stance_time"]["swing"].items()}
    leg_stance_starts = {k:v/nmf.timestep for k,v in data["swing_stance_time"]["stance"].items()}

    return data_block, match_leg_to_joints, joint_ids, leg_swing_starts, leg_stance_starts


####### CPG #########
def get_CPG_parameters(freq=12):

    frequencies = np.ones(N_OSCILLATORS) * freq

    # For now each oscillator have the same amplitude
    target_amplitudes = np.ones(N_OSCILLATORS) * TARGET_AMPLITUDE
    rates = np.ones(N_OSCILLATORS) * AMP_RATES

    phase_biases = phase_biases_tripod_idealized * 2 * np.pi
    coupling_weights = (np.abs(phase_biases) > 0).astype(float) * COUPLING_STRENGTH

    return frequencies, target_amplitudes, rates, phase_biases, coupling_weights


def run_CPG(nmf, data_block, match_leg_to_joints, joint_ids, slope, axis, base_path, leg_swing_starts, leg_stance_starts):

    print(f"Running CPG gravity {slope} {axis}")

    # Define save path
    save_path = (
        base_path
        / f"CPG_gravity_{slope}_{axis}.pkl"
    )
    if save_path.exists():
        print(f"CPG gravity {slope} {axis} already exists")
        return
    video_path = save_path.with_suffix(".mp4")

    nmf.reset()
    if axis == "x":
        nmf.sim_params.render_camera = "Animat/camera_front"
    elif axis == "y":
        nmf.sim_params.render_camera = "Animat/camera_left"

    n_stabilization_steps = int(STABILIZATION_DUR / nmf.timestep)
    gravity_switching_step = int(GRAVITY_SWITCHING_T / nmf.timestep)

    num_steps = int(RUN_TIME / nmf.timestep) + n_stabilization_steps
    interp_step_duration = data_block.shape[1]

    joints_to_leg = np.array([i for ts in nmf.last_tarsalseg_names for i, joint in enumerate(nmf.actuated_joints) if f"{ts[:2]}Coxa_roll" in joint])
    stance_starts_in_order = np.array([leg_stance_starts[ts[:2]] for ts in nmf.last_tarsalseg_names])
    swing_starts_in_order = np.array([leg_swing_starts[ts[:2]] for ts in nmf.last_tarsalseg_names])
    indices = np.zeros_like(nmf.actuated_joints, dtype=np.int64)


    # Get CPG parameters
    (
        frequencies,
        target_amplitudes,
        rates,
        phase_biases,
        coupling_weights,
    ) = get_CPG_parameters()

    # Initilize the simulation
    np.random.seed(CONTROLLER_SEED)
    start_ampl = np.ones(6) * 0.2
    solver = initialize_solver(
        phase_oscillator,
        "dopri5",
        nmf.curr_time,
        N_OSCILLATORS,
        frequencies,
        coupling_weights,
        phase_biases,
        start_ampl,
        rates,
        int_params={"atol": 1e-6, "rtol": 1e-6, "max_step": 100000},
    )

    joint_angles = np.zeros((num_steps, len(nmf.actuated_joints)))
    # Initalize storage
    obs_list = []

    for i in range(num_steps):
        res = solver.integrate(nmf.curr_time)
        phase = res[:N_OSCILLATORS]
        amp = res[N_OSCILLATORS : 2 * N_OSCILLATORS]

        if i == n_stabilization_steps:
            # Now set the amplitude to their real values
            solver.set_f_params(
                N_OSCILLATORS,
                frequencies,
                coupling_weights,
                phase_biases,
                target_amplitudes,
                rates,
            )
        if i == gravity_switching_step:
            nmf.set_slope(slope, axis)
        if i > n_stabilization_steps:
            indices = advancement_transfer(
                phase, interp_step_duration, match_leg_to_joints
            )
            # scale amplitude by interpolating between the resting values and i timestep value
            input_joint_angles = (
                data_block[joint_ids, 0]
                + (data_block[joint_ids, indices] - data_block[joint_ids, 0])
                * amp[match_leg_to_joints]
            )
        else:
            input_joint_angles = data_block[joint_ids, 0]

        joint_angles[i, :] = input_joint_angles
        #adhesion_signal = nmf.get_adhesion_vector()
        adhesion_signal = adhesion_signal = np.logical_or(indices[joints_to_leg] < swing_starts_in_order,
                                         indices[joints_to_leg] > stance_starts_in_order)
        action = {"joints": input_joint_angles, "adhesion": adhesion_signal}

        try:
            obs, _, _, _, _ = nmf.step(action)
            obs_list.append(obs)
            _ = nmf.render()
        except Exception as e:
            print(e)
            break
    if video_path:
        nmf.save_video(
            video_path, stabilization_time=STABILIZATION_DUR-0.05
        )

    # Save the data
    with open(save_path, "wb") as f:
        pickle.dump(obs_list, f)
    return


########### MAIN ############
if __name__ == "__main__":
    slopes_in_degrees = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90][::-1]
    """base_gravity = np.array([0, 0, -9810])
    base_gravity_norm = np.linalg.norm(base_gravity)
    # project the base gravity vector on the slope (compute corresponding x and z components)
    front_incline_gravity_vectors = [
        [-base_gravity_norm * np.cos(np.deg2rad(slope)), 0, -base_gravity_norm * np.sin(np.deg2rad(slope))]
        for slope in slopes_in_degrees
    ]
    side_incline_gravity_vectors = [
        [0, -base_gravity_norm * np.cos(np.deg2rad(slope)), -base_gravity_norm * np.sin(np.deg2rad(slope))]
        for slope in slopes_in_degrees
    ]"""

    # Initialize simulation but with flat terrain at the beginning to define the swing and stance starts
    sim_params = MuJoCoParameters(
        timestep=1e-4, render_mode="saved", render_playspeed=0.1, enable_adhesion=True, draw_adhesion=True,
        align_camera_with_gravity =True, draw_gravity=False,
    )
    nmf = NeuroMechFlyMuJoCo(
        sim_params=sim_params,
        init_pose=stretched_pose,
        actuated_joints=all_leg_dofs,
    )

    metadata = {"controller_seed": CONTROLLER_SEED, "run_time": RUN_TIME,
                "stabilization_dur": STABILIZATION_DUR,
                "gravity_switching_t": GRAVITY_SWITCHING_T,
                "coupling_strength": COUPLING_STRENGTH,
                "amp_rates": AMP_RATES,
                "target_amplitude": TARGET_AMPLITUDE,
                "legs": LEGS,
                "n_oscillators": N_OSCILLATORS,
                #"sim_params": nmf.sim_params,
                }

    # Load and process data block only once as this won't change
    data_block, match_leg_to_joints, joint_ids, leg_swing_starts, leg_stance_starts = get_data_block(
        nmf.timestep, nmf.actuated_joints
    )

    # Create folder to save data points
    base_path = Path(f"Data_points/slope_front")
    base_path.mkdir(parents=True, exist_ok=True)

    # save metadata
    metadata_path = base_path / "metadata.yml"
    if not metadata_path.exists():
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)

    start_exps = time.time()
    print("Starting front slope experiments")
    for slope in slopes_in_degrees:
        run_CPG(
            nmf,
            data_block,
            match_leg_to_joints,
            joint_ids,
            slope,
            "y",
            base_path,
            leg_swing_starts,
            leg_stance_starts,
        )

    print(f"{len(slopes_in_degrees)} experiments took {time.time()-start_exps:.2f} seconds")