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

from flygym.util.hybrid_helpers import get_raise_leg

from flygym.arena.mujoco_arena import (
    FlatTerrain,
    GappedTerrain,
    BlocksTerrain,
    MixedTerrain,
)

import yaml

########### CONSTANTS ############
ENVIRONEMENT_SEED = 0

N_STABILIZATION_STEPS = 2000
RUN_TIME = 1.5

LEGS = ["RF", "RM", "RH", "LF", "LM", "LH"]
N_OSCILLATORS = len(LEGS)

Z_SPAWN_POS = 0.5

# Need longer of period as coordination is a bit worse and legs are more dragged than stepped
ADHESION_OFF_DUR_DECENTRALIZED = 450

COUPLING_STRENGTH = 10.0
AMP_RATES = 20.0
TARGET_AMPLITUDE = 1.0

ACTUATOR_KP = 30.0
ADHESION_GAIN = 40.0


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
    adhesion = nmf.sim_params.enable_adhesion

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


def run_hybrid(nmf, seed, data_block, match_leg_to_joints, joint_ids, raise_leg, video_path=None):
    nmf.reset()
    adhesion = nmf.sim_params.enable_adhesion
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

    adhesion_signal = np.zeros(6)
    # Initalize storage
    obs_list = []

    # Setup hybrid controller
    legs_in_hole = [False] * 6
    legs_in_hole_increment = np.zeros(6)

    floor_height = np.inf
    # Lowest point in the walking parts of the floor
    for i_g in range(nmf.physics.model.ngeom):
        geom = nmf.physics.model.geom(i_g)
        name = geom.name
        if "groundblock" in name:
            block_height = geom.pos[2] + geom.size[2]
            floor_height = min(floor_height, block_height)
    floor_height -= 0.05 # account for small penetrations of the floor

    # detect leg with "unatural" other than tarsus 4 or 5 contacts
    leg_tarsus1T_contactsensors = [
        [
            i
            for i, cs in enumerate(nmf.contact_sensor_placements)
            if tarsal_seg[:2] in cs and ("Tibia" in cs or "Tarsus1" in cs)
        ]
        for tarsal_seg in nmf.last_tarsalseg_names
    ]
    force_threshold = 5.0
    highest_proximal_contact_leg = [False] * 6
    legs_w_proximalcontact_increment = np.zeros(6)

    increase_rate = 0.1
    decrease_rate = 0.05

    last_tarsalseg_to_adh_id = [
        i
        for adh in nmf.adhesion_actuators
        for i, lts in enumerate(nmf.last_tarsalseg_names)
        if lts[:2] == adh.name[:2]
    ]

    for i in range(num_steps):
        if i > N_STABILIZATION_STEPS+500:
            # detect leg in gap show as blue tibia #only keep the deepest leg in the hole
            ee_z_pos = obs["end_effectors"][2::3]
            legs_in_hole = ee_z_pos < floor_height
            legs_in_hole = np.logical_and(legs_in_hole, ee_z_pos == np.min(ee_z_pos))
            for k, tarsal_seg in enumerate(nmf.last_tarsalseg_names):
                if legs_in_hole[k]:
                    legs_in_hole_increment[k] += increase_rate
                else:
                    if legs_in_hole_increment[k] > 0:
                        legs_in_hole_increment[k] -= decrease_rate

            # detect leg with "unatural" other than tarsus 4 or 5 contacts and show as red Femur (Only look at force x and z (stay on top of the blocks))
            tarsus1T_contact_force = np.mean(
                np.abs(obs["contact_forces"][::2, leg_tarsus1T_contactsensors]),
                axis=(0, -1),
            )
            # look for the highest force
            highest_proximal_contact_leg = np.logical_and(
                tarsus1T_contact_force > force_threshold,
                max(tarsus1T_contact_force) == tarsus1T_contact_force,
            )
            for k, tarsal_seg in enumerate(nmf.last_tarsalseg_names):
                if highest_proximal_contact_leg[k] and not legs_in_hole[k]:
                    legs_w_proximalcontact_increment[k] += increase_rate
                else:
                    if legs_w_proximalcontact_increment[k] > 0:
                        legs_w_proximalcontact_increment[k] -= decrease_rate

        joint_angle_increment = (raise_leg.T * legs_in_hole_increment).sum(axis=1) + (
            raise_leg.T * legs_w_proximalcontact_increment
        ).sum(axis=1)

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

        # Modify joint angles with hybrid input
        input_joint_angles = input_joint_angles + joint_angle_increment

        if adhesion:
            adhesion_signal = nmf.get_adhesion_vector()
            # if leg in an hole or contacting with the wrong part of the leg remove adhesion
            adhesion_signal[
                np.logical_or(legs_in_hole, highest_proximal_contact_leg)[
                    last_tarsalseg_to_adh_id
                ]
            ] = 0.0
        else:
            adhesion_signal = np.zeros(6)

        action = {"joints": input_joint_angles,
                  "adhesion": adhesion_signal}

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
    adhesion = nmf.sim_params.enable_adhesion
    if adhesion:
        nmf.adhesion_off_duration_steps = ADHESION_OFF_DUR_DECENTRALIZED

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
    hybridpts_path,
    match_leg_to_joints,
    joint_ids,
    leg_swings_starts,
    leg_stance_starts,
    raise_leg,
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

    # Generate hybrid points
    hybrid_path = (
        hybridpts_path
        / f"{arena_type}pts_seed{seed}_pos{pos[0]:.2f}_{pos[1]:.2f}.pkl"
    )
    if not hybrid_path.is_file():
        hybrid_obs_list = run_hybrid(
            nmf,
            seed,
            data_block,
            match_leg_to_joints,
            joint_ids,
            raise_leg,
            video_path=hybrid_path.with_suffix(".mp4"),
        )
        # Save as pkl
        with open(hybrid_path, "wb") as f:
            pickle.dump(hybrid_obs_list, f)

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
    max_x = 4.0
    shift_x = 2.0
    max_y = 4.0
    shift_y = 2.0
    positions = np.random.rand(args.n_exp, 2)

    positions[:, 0] = positions[:, 0] * max_x - shift_x
    positions[:, 1] = positions[:, 1] * max_y - shift_y


    internal_seeds = [42, 33, 0, 100, 99, 56, 28, 7, 21, 13]
    assert args.n_exp <= len(internal_seeds), "Not enough internal seeds defined"
    internal_seeds = internal_seeds[: args.n_exp]

    # Initialize simulation but with flat terrain at the beginning to define the swing and stance starts
    # Set high actuator kp to be able to overcome obstacles
    sim_params = MuJoCoParameters(
        timestep=1e-4,
        render_mode="saved",
        render_playspeed=0.1,
        enable_adhesion=adhesion,
        actuator_kp=ACTUATOR_KP,
        adhesion_gain=ADHESION_GAIN,
    )
    nmf = NeuroMechFlyMuJoCo(
        sim_params=sim_params,
        init_pose=stretched_pose,
        actuated_joints=all_leg_dofs,
    )

    #save metadata to yaml
    metadata = {"run_time": RUN_TIME,
                "n_stabilization_steps": N_STABILIZATION_STEPS,
                "coupling_strength": COUPLING_STRENGTH,
                "amp_rates": AMP_RATES,
                "target_amplitude": TARGET_AMPLITUDE,
                "legs": LEGS,
                "n_oscillators": N_OSCILLATORS,
                "adhesion_off_duration_decent": ADHESION_OFF_DUR_DECENTRALIZED,
                "adhesion_gain": ADHESION_GAIN,
                # "sim_params": nmf.sim_params,
                }
    metadata_path = Path(f"Data_points/{arena_type}_metadata.yaml")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)

    # Load and process data block only once as this wont change
    data_block, match_leg_to_joints, joint_ids = get_data_block(
        nmf.timestep, nmf.actuated_joints
    )

    # Get stance and swing starts only once as this wont change
    leg_swing_starts, leg_stance_starts, _, _ = define_swing_stance_starts(
        nmf, data_block, use_adhesion=adhesion, n_steps_stabil=N_STABILIZATION_STEPS
    )

    # Get the joint angles leading to a leg raise in each leg
    raise_leg = get_raise_leg(nmf)

    # Create folder to save data points
    CPGpts_path = Path(
        f"Data_points/{arena_type}_CPGpts_adhesion{adhesion}_kp{ACTUATOR_KP}"
    )
    CPGpts_path.mkdir(parents=True, exist_ok=True)
    decentralizedpts_path = Path(
        f"Data_points/{arena_type}_Decentralizedpts_adhesion{adhesion}_kp{ACTUATOR_KP}"
    )
    decentralizedpts_path.mkdir(parents=True, exist_ok=True)
    hybridpts_path = Path(
        f"Data_points/{arena_type}_hybridpts_adhesion{adhesion}_kp{ACTUATOR_KP}"
    )
    hybridpts_path.mkdir(parents=True, exist_ok=True)

    sim_params.draw_adhesion = adhesion
    nmf_params = {
        "sim_params": sim_params,
        "init_pose": stretched_pose,
        "actuated_joints": all_leg_dofs,
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
                hybridpts_path,
                match_leg_to_joints,
                joint_ids,
                leg_swing_starts,
                leg_stance_starts,
                raise_leg,
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
                hybridpts_path,
                match_leg_to_joints,
                joint_ids,
                leg_swing_starts,
                leg_stance_starts,
                raise_leg,
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
