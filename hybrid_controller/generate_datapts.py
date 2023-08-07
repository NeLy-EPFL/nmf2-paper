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

from flygym.util.CPG_helpers import (
    advancement_transfer,
    phase_oscillator,
    initialize_solver,
    phase_biases_tripod_idealized,
)
from flygym.util.Decentralized_helpers import (
    define_swing_stance_starts,
    update_stepping_advancement,
    compute_leg_scores,
    rule1_corresponding_legs,
    rule2_corresponding_legs,
    rule3_corresponding_legs,
    rule1_weight,
    rule2_weight,
    rule2_weight_contralateral,
    rule3_weight,
    rule3_weight_contralateral,
    percent_margin,
)

from flygym.arena.mujoco_arena import (
    FlatTerrain,
    GappedTerrain,
    BlocksTerrain,
    MixedTerrain,
)


########### CONSTANTS ############
ENVIRONEMENT_SEED = 0

N_STABILIZATION_STEPS = 2000
RUN_TIME = 1

LEGS = ["RF", "RM", "RH", "LF", "LM", "LH"]
N_OSCILLATORS = len(LEGS)

Z_SPAWN_POS = 0.5

# Need longer of period as coordination is a bit worse and legs are more dragged than stepped
ADHESION_OFF_DUR_DECENTRALIZED = 450

COUPLING_STRENGTH = 10.0
AMP_RATES = 20.0
TARGET_AMPLITUDE = 1.0


########### FUNCTIONS ############
####### Initialization #########
def get_arena(arena_type, seed=ENVIRONEMENT_SEED):
    if arena_type == "flat":
        return FlatTerrain()
    elif arena_type == "gapped":
        return GappedTerrain()
    elif arena_type == "blocks":
        return BlocksTerrain()
    elif arena_type == "mixed":
        return MixedTerrain(rand_seed=seed)  # seed for randomized block heights


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

    return data_block, match_leg_to_joints, joint_ids


####### CPG #########


def get_CPG_parameters(freq=7):
    # freq of 1/7 is 7 steps per second
    frequencies = np.ones(N_OSCILLATORS) * freq

    # For now each oscillator have the same amplitude
    target_amplitudes = np.ones(N_OSCILLATORS) * TARGET_AMPLITUDE
    rates = np.ones(N_OSCILLATORS) * AMP_RATES

    phase_biases = phase_biases_tripod_idealized * 2 * np.pi
    coupling_weights = (np.abs(phase_biases) > 0).astype(float) * COUPLING_STRENGTH

    return frequencies, target_amplitudes, rates, phase_biases, coupling_weights


def run_CPG(nmf, seed, data_block, match_leg_to_joints, joint_ids, video_path=None):
    nmf.reset()
    adhesion = nmf.adhesion

    num_steps = int(RUN_TIME / nmf.timestep) + N_STABILIZATION_STEPS
    interp_step_duration = data_block.shape[1]

    # Get CPG parameters
    (
        frequencies,
        target_amplitudes,
        rates,
        phase_biases,
        coupling_weights,
    ) = get_CPG_parameters()

    # Initilize the simulation
    np.random.seed(seed)
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
    adhesion_signal = np.zeros(6)
    # Initalize storage
    obs_list = []

    for i in range(num_steps):
        res = solver.integrate(nmf.curr_time)
        phase = res[:N_OSCILLATORS]
        amp = res[N_OSCILLATORS : 2 * N_OSCILLATORS]

        if i == N_STABILIZATION_STEPS:
            # Now set the amplitude to their real values
            solver.set_f_params(
                N_OSCILLATORS,
                frequencies,
                coupling_weights,
                phase_biases,
                target_amplitudes,
                rates,
            )
        if i > N_STABILIZATION_STEPS:
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
        if adhesion:
            adhesion_signal = nmf.get_adhesion_vector()
        else:
            adhesion_signal = np.zeros(6)

        action = {"joints": input_joint_angles, "adhesion": adhesion_signal}

        obs, _, _, _, _ = nmf.step(action)
        obs_list.append(obs)
        _ = nmf.render()

    if video_path:
        nmf.save_video(
            video_path, stabilization_time=N_STABILIZATION_STEPS * nmf.timestep - 0.1
        )

    return obs_list


####### Decentralized #########
def run_Decentralized(
    nmf, seed, data_block, leg_swings_starts, leg_stance_starts, video_path=None
):
    nmf.reset()
    adhesion = nmf.adhesion
    if adhesion:
        nmf.adhesion_OFF_dur = ADHESION_OFF_DUR_DECENTRALIZED

    # Get data block
    num_steps = int(RUN_TIME / nmf.timestep) + N_STABILIZATION_STEPS
    interp_step_duration = data_block.shape[1]

    np.random.seed(seed)

    leg_ids = np.arange(len(LEGS)).astype(int)
    leg_corresp_id = dict(zip(LEGS, leg_ids))
    n_joints = len(nmf.actuated_joints)
    joint_ids = np.arange(n_joints).astype(int)
    match_leg_to_joints = np.array(
        [
            i
            for joint in nmf.actuated_joints
            for i, leg in enumerate(LEGS)
            if leg in joint
        ]
    )

    # This serves to keep track of the advancement of each leg in the stepping sequence
    stepping_advancement = np.zeros(len(LEGS)).astype(int)

    leg_scores = np.zeros(len(LEGS))
    obs_list = []
    adhesion_signal = np.zeros(6)

    # Run the actual simulation
    for i in range(num_steps):
        # Decide in which leg to step
        initiating_leg = np.argmax(leg_scores)
        within_margin_legs = (
            leg_scores[initiating_leg] - leg_scores
            <= leg_scores[initiating_leg] * percent_margin
        )

        # If multiple legs are within the margin choose randomly among those legs
        if np.sum(within_margin_legs) > 1:
            initiating_leg = np.random.choice(np.where(within_margin_legs)[0])

        # If the maximal score is zero or less (except for the first step after stabilisation to initate the locomotion) or if the leg is already stepping
        if (
            leg_scores[initiating_leg] <= 0 and not i == N_STABILIZATION_STEPS + 1
        ) or stepping_advancement[initiating_leg] > 0:
            initiating_leg = None
        else:
            stepping_advancement[initiating_leg] += 1

        joint_pos = data_block[joint_ids, stepping_advancement[match_leg_to_joints]]

        if adhesion:
            adhesion_signal = nmf.get_adhesion_vector()
        else:
            adhesion_signal = np.zeros(6)

        action = {"joints": joint_pos, "adhesion": adhesion_signal}
        obs, _, _, _, _ = nmf.step(action)
        nmf.render()
        obs_list.append(obs)

        stepping_advancement = update_stepping_advancement(
            stepping_advancement, LEGS, interp_step_duration
        )

        rule1_contrib, rule2_contrib, rule3_contrib = compute_leg_scores(
            rule1_corresponding_legs,
            rule1_weight,
            rule2_corresponding_legs,
            rule2_weight,
            rule2_weight_contralateral,
            rule3_corresponding_legs,
            rule3_weight,
            rule3_weight_contralateral,
            stepping_advancement,
            leg_corresp_id,
            leg_stance_starts,
            interp_step_duration,
            LEGS,
        )

        leg_scores = rule1_contrib + rule2_contrib + rule3_contrib

    # Return observation list

    if video_path:
        nmf.save_video(
            video_path, stabilization_time=N_STABILIZATION_STEPS * nmf.timestep - 0.1
        )

    return obs_list


def run_experiment(
    seed,
    pos,
    data_block,
    nmf_params,
    arena_type,
    CPGpts_path,
    decentralizedpts_path,
    match_leg_to_joints,
    joint_ids,
    leg_swings_starts,
    leg_stance_starts,
):
    arena = get_arena(arena_type)
    nmf_params["spawn_pos"] = np.array([pos[0], pos[1], Z_SPAWN_POS])
    nmf = NeuroMechFlyMuJoCo(**nmf_params, arena=arena)
    # Generate CPG points
    CPG_path = (
        CPGpts_path / f"{arena_type}pts_seed{seed}_pos{pos[0]:.2f}_{pos[1]:.2f}.pkl"
    )
    if not CPG_path.is_file():
        CPG_obs_list = run_CPG(
            nmf,
            seed,
            data_block,
            match_leg_to_joints,
            joint_ids,
            video_path=CPG_path.with_suffix(".mp4"),
        )
        # Save as pkl
        with open(CPG_path, "wb") as f:
            pickle.dump(CPG_obs_list, f)

    # Generate Decentralized points
    decentralized_path = (
        decentralizedpts_path
        / f"{arena_type}pts_seed{seed}_pos{pos[0]:.2f}_{pos[1]:.2f}.pkl"
    )
    if not decentralized_path.is_file():
        decentralized_obs_list = run_Decentralized(
            nmf,
            seed,
            data_block,
            leg_swings_starts,
            leg_stance_starts,
            video_path=decentralized_path.with_suffix(".mp4"),
        )
        # Save as pkl
        with open(decentralized_path, "wb") as f:
            pickle.dump(decentralized_obs_list, f)


########### MAIN ############
def main(args):
    # Parse arguments for arena type and adhesion
    assert args.arena in [
        "flat",
        "gapped",
        "blocks",
        "mixed",
    ], "Arena type not recognized"

    arena_type = args.arena
    adhesion = args.adhesion

    np.random.seed(ENVIRONEMENT_SEED)

    # Generate random positions
    max_x = 5
    shift_x = 2.5
    max_y = 5
    shift_y = 2.5
    positions = np.random.rand(args.n_exp, 2)

    positions[:, 0] = positions[:, 0] * max_x + shift_x
    positions[:, 1] = positions[:, 1] * max_y + shift_y

    print()

    internal_seeds = [42, 33, 0, 100, 99, 56, 28, 7, 21, 13]
    assert args.n_exp <= len(internal_seeds), "Not enough internal seeds defined"
    internal_seeds = internal_seeds[: args.n_exp]

    # Initialize simulation but with flat terrain at the beginning to define the swing and stance starts
    sim_params = MuJoCoParameters(
        timestep=1e-4, render_mode="saved", render_playspeed=0.1
    )
    nmf = NeuroMechFlyMuJoCo(
        sim_params=sim_params,
        init_pose=stretched_pose,
        actuated_joints=all_leg_dofs,
        adhesion=adhesion,
    )

    # Load and process data block only once as this wont change
    data_block, match_leg_to_joints, joint_ids = get_data_block(
        nmf.timestep, nmf.actuated_joints
    )

    # Get stance and swing starts only once as this wont change
    leg_swing_starts, leg_stance_starts, _, _ = define_swing_stance_starts(
        nmf, data_block, use_adhesion=adhesion, n_steps_stabil=N_STABILIZATION_STEPS
    )

    # Create folder to save data points
    CPGpts_path = Path(f"Data_points/{arena_type}_CPGpts_adhesion{adhesion}")
    CPGpts_path.mkdir(parents=True, exist_ok=True)
    decentralizedpts_path = Path(
        f"Data_points/{arena_type}_Decentralizedpts_adhesion{adhesion}"
    )
    decentralizedpts_path.mkdir(parents=True, exist_ok=True)

    nmf_params = {
        "sim_params": sim_params,
        "init_pose": stretched_pose,
        "actuated_joints": all_leg_dofs,
        "adhesion": adhesion,
        "draw_adhesion": adhesion,
    }
    start_exps = time.time()
    print("Starting experiments")
    # Parallelize the experiment
    if args.parallel:
        task_configuration = [
            (
                seed,
                pos,
                data_block,
                nmf_params,
                arena_type,
                CPGpts_path,
                decentralizedpts_path,
                match_leg_to_joints,
                joint_ids,
                leg_swing_starts,
                leg_stance_starts,
            )
            for seed, pos in zip(internal_seeds, positions)
        ]
        with multiprocessing.Pool(4) as pool:
            pool.starmap(run_experiment, task_configuration)
        pool.join()
        pool.close()
    else:
        for pos, seed in zip(positions, internal_seeds):
            run_experiment(
                seed,
                pos,
                data_block,
                nmf_params,
                arena_type,
                CPGpts_path,
                decentralizedpts_path,
                match_leg_to_joints,
                joint_ids,
                leg_swing_starts,
                leg_stance_starts,
            )

    print(f"{args.n_exp} experiments took {time.time()-start_exps:.2f} seconds")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--arena", type=str, default="flat", help="Type of arena to use")
    args.add_argument("--adhesion", action="store_true", help="Use adhesion or not")
    args.add_argument(
        "--n_exp", type=int, default=10, help="Number of experiments to run"
    )
    args.add_argument(
        "--parallel", action="store_true", help="Run experiments in parallel"
    )
    args = args.parse_args()

    main(args)
