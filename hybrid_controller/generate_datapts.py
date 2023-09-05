import numpy as np
from pathlib import Path
import pickle
import pkg_resources

import argparse
import multiprocessing
import time

import flygym.util.cpg_controller as cpg_controller
import flygym.util.decentralized_controller as decentralized_controller
import flygym.util.hybrid_controller as hybrid_controller
import flygym.arena.mujoco_arena as mujoco_arena
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.util.config import all_leg_dofs
from flygym.state import stretched_pose

from dm_control.rl.control import PhysicsError

import yaml

########### CONSTANTS ############
ENVIRONEMENT_SEED = 0

N_STABILIZATION_STEPS = 2000
RUN_TIME = 1

LEGS = ["RF", "RM", "RH", "LF", "LM", "LH"]
N_OSCILLATORS = len(LEGS)

Z_SPAWN_POS = 0.3

# Need longer of period as coordination is a bit worse and legs are more
# dragged than stepped
ADHESION_OFF_DUR = 0.03

COUPLING_STRENGTH = 10.0
AMP_RATES = 20.0
TARGET_AMPLITUDE = 1.0
START_AMPL = 0.0
FREQ = 12.0

ACTUATOR_KP = 30.0
ADHESION_GAIN = 40.0


########### FUNCTIONS ############
####### Initialization #########
def get_arena(arena_type, seed=ENVIRONEMENT_SEED):
    if arena_type == "flat":
        return mujoco_arena.FlatTerrain()
    elif arena_type == "gapped":
        return mujoco_arena.GappedTerrain(gap_width=0.5)
    elif arena_type == "blocks":
        return mujoco_arena.BlocksTerrain(
            rand_seed=seed,
        )
    elif arena_type == "mixed":
        return mujoco_arena.MixedTerrain(
            gap_width=0.5, rand_seed=seed
        )  # seed for randomized block heights


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

    leg_swing_starts = {
        k: round(v / timestep) for k, v in data["swing_stance_time"]["swing"].items()
    }
    leg_stance_starts = {
        k: round(v / timestep) for k, v in data["swing_stance_time"]["stance"].items()
    }

    return (
        data_block,
        match_leg_to_joints,
        joint_ids,
        leg_swing_starts,
        leg_stance_starts,
    )


####### CPG #########
def get_CPG_parameters(freq=12):
    frequencies = np.ones(N_OSCILLATORS) * freq

    # For now each oscillator have the same amplitude
    target_amplitudes = np.ones(N_OSCILLATORS) * TARGET_AMPLITUDE
    rates = np.ones(N_OSCILLATORS) * AMP_RATES

    phase_biases = cpg_controller.phase_biases_tripod_idealized * 2 * np.pi
    coupling_weights = (np.abs(phase_biases) > 0).astype(float) * COUPLING_STRENGTH

    return frequencies, target_amplitudes, rates, phase_biases, coupling_weights


def run_cpg(
    nmf,
    seed,
    data_block,
    match_leg_to_joints,
    joint_ids,
    leg_swing_starts,
    leg_stance_starts,
    video_path=None,
):
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
    start_ampl = np.ones(6) * START_AMPL
    solver = cpg_controller.initialize_solver(
        cpg_controller.phase_oscillator,
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

    joints_to_leg = np.array(
        [
            i
            for ts in nmf.last_tarsalseg_names
            for i, joint in enumerate(nmf.actuated_joints)
            if f"{ts[:2]}Coxa_roll" in joint
        ]
    )
    stance_starts_in_order = np.array(
        [leg_stance_starts[ts[:2]] for ts in nmf.last_tarsalseg_names]
    )
    swing_starts_in_order = np.array(
        [leg_swing_starts[ts[:2]] for ts in nmf.last_tarsalseg_names]
    )
    indices = np.zeros_like(nmf.actuated_joints, dtype=np.int64)

    adhesion_signal = np.zeros(6)
    # Initalize storage
    obs_list = []

    try:
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
                indices = cpg_controller.advancement_transfer(
                    phase, interp_step_duration, match_leg_to_joints
                )
                # scale amplitude by interpolating between the resting values and i
                # timestep value
                input_joint_angles = (
                    data_block[joint_ids, 0]
                    + (data_block[joint_ids, indices] - data_block[joint_ids, 0])
                    * amp[match_leg_to_joints]
                )
            else:
                input_joint_angles = data_block[joint_ids, 0]

            if adhesion:
                adhesion_signal = np.logical_or(
                    indices[joints_to_leg] < swing_starts_in_order,
                    indices[joints_to_leg] > stance_starts_in_order,
                )
            else:
                adhesion_signal = np.zeros(6)

            action = {"joints": input_joint_angles, "adhesion": adhesion_signal}

            obs, _, _, _, _ = nmf.step(action)
            obs_list.append(obs)
            _ = nmf.render()

    except PhysicsError:
        print("Simulation diverged, returning")

    if video_path:
        nmf.save_video(video_path, stabilization_time=0.2)

    return obs_list


def run_hybrid(
    nmf,
    seed,
    data_block,
    match_leg_to_joints,
    joint_ids,
    raise_leg,
    leg_swing_starts,
    leg_stance_starts,
    video_path=None,
):
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
    start_ampl = np.ones(6) * START_AMPL
    solver = cpg_controller.initialize_solver(
        cpg_controller.phase_oscillator,
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
    joints_to_leg = np.array(
        [
            i
            for ts in nmf.last_tarsalseg_names
            for i, joint in enumerate(nmf.actuated_joints)
            if f"{ts[:2]}Coxa_roll" in joint
        ]
    )
    stance_starts_in_order = np.array(
        [leg_stance_starts[ts[:2]] for ts in nmf.last_tarsalseg_names]
    )
    swing_starts_in_order = np.array(
        [leg_swing_starts[ts[:2]] for ts in nmf.last_tarsalseg_names]
    )
    indices = np.zeros_like(nmf.actuated_joints, dtype=np.int64)
    leg_in_stance = np.logical_or(
        indices[joints_to_leg] < swing_starts_in_order,
        indices[joints_to_leg] > stance_starts_in_order,
    )

    # Initalize storage
    obs_list = []

    # Setup hybrid controller
    legs_in_hole = [False] * 6
    legs_in_hole_increment = np.zeros(6)

    # detect leg with "unatural" other than tarsus 4 or 5 contacts
    leg_tarsus12T_contactsensors = [
        [
            i
            for i, cs in enumerate(nmf.contact_sensor_placements)
            if tarsal_seg[:2] in cs
            and ("Tibia" in cs or "Tarsus1" in cs or "Tarsus2" in cs)
        ]
        for tarsal_seg in nmf.last_tarsalseg_names
    ]
    force_threshold = -1.0
    proximal_contact_leg = [False] * 6
    legs_w_proximalcontact_increment = np.zeros(6)

    # change it to seconds so it is timestep independant
    increase_rate_stumble = 1 / 5e-4 * nmf.timestep  # 1 step every 500µs
    decrease_rate_stumble = 1 / 2e-3 * nmf.timestep

    increase_rate_hole = 1 / 2e-3 * nmf.timestep
    decrease_rate_hole = 1 / 3e-3 * nmf.timestep

    last_tarsalseg_to_adh_id = [
        i
        for adh in nmf.adhesion_actuators
        for i, lts in enumerate(nmf.last_tarsalseg_names)
        if lts[:2] == adh.name[:2]
    ]
    try:
        for i in range(num_steps):
            if i > N_STABILIZATION_STEPS:
                # detect leg in gap show as blue tibia #only keep the deepest leg in the hole
                # detect leg in gap show as blue tibia #only keep the deepest leg in the hole
                ee_z_pos = obs["end_effectors"][2::3]
                leg_to_thorax_zdistance = obs["fly"][0][2] - ee_z_pos
                # get the third furthest leg from the thorax (as tripod should be on the floor)
                third_furthest_leg = np.sort(leg_to_thorax_zdistance)[3]
                # print(np.sort(leg_to_thorax_zdistance))
                legs_in_hole = np.logical_and(
                    leg_to_thorax_zdistance > third_furthest_leg + 0.05,
                    leg_to_thorax_zdistance == np.max(leg_to_thorax_zdistance),
                )
                for k, tarsal_seg in enumerate(nmf.last_tarsalseg_names):
                    if legs_in_hole[k]:
                        # nmf.physics.named.model.geom_rgba[
                        #     "Animat/" + tarsal_seg[:2] + "Tibia_visual"
                        # ] = [0.0, 0.0, 1.0, 1.0]
                        legs_in_hole_increment[k] += increase_rate_hole
                    else:
                        # nmf.physics.named.model.geom_rgba[
                        #     "Animat/" + tarsal_seg[:2] + "Tibia_visual"
                        # ] = nmf.base_rgba
                        if legs_in_hole_increment[k] > 0:
                            legs_in_hole_increment[k] -= decrease_rate_hole

                # detect leg with "unatural" other than tarsus 2, 3, 4 or 5 contacts and show as red Femur (Only look at force along negative x)
                fly_orient = obs["fly_orient"]
                fly_orient[2] = 0.0
                fly_orient_norm = np.linalg.norm(fly_orient)
                fly_orient_unit = fly_orient / fly_orient_norm
                # Look for forces opposing the fly orientation (and thus progression)
                for k, contact_sensors in enumerate(leg_tarsus12T_contactsensors):
                    contact_forces = obs["contact_forces"][:, contact_sensors].T
                    min_scalar_proj_contact_force = np.inf
                    for contact_force in contact_forces:
                        scalar_proj_force = np.dot(
                            contact_force, fly_orient_unit
                        )  # scalar_value
                        min_scalar_proj_contact_force = min(
                            min_scalar_proj_contact_force, scalar_proj_force
                        )
                    proximal_contact_leg[k] = np.logical_and(
                        min_scalar_proj_contact_force < force_threshold,
                        np.logical_not(leg_in_stance[k]),
                    )

                for k, tarsal_seg in enumerate(nmf.last_tarsalseg_names):
                    if proximal_contact_leg[k] and not legs_in_hole[k]:
                        # nmf.physics.named.model.geom_rgba[
                        #     "Animat/" + tarsal_seg[:2] + "Femur_visual"
                        # ] = [1.0, 0.0, 0.0, 1.0]
                        legs_w_proximalcontact_increment[k] += increase_rate_stumble
                    else:
                        # nmf.physics.named.model.geom_rgba[
                        #     "Animat/" + tarsal_seg[:2] + "Femur_visual"
                        # ] = nmf.base_rgba
                        if legs_w_proximalcontact_increment[k] > 0:
                            legs_w_proximalcontact_increment[k] -= decrease_rate_stumble

            # Calculate joint angle increment
            incr_legs_in_hole = (raise_leg.T * legs_in_hole_increment).sum(axis=1)
            incr_prox_contact = (raise_leg.T * legs_w_proximalcontact_increment).sum(
                axis=1
            )
            joint_angle_increment = incr_legs_in_hole + incr_prox_contact

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
                indices = cpg_controller.advancement_transfer(
                    phase, interp_step_duration, match_leg_to_joints
                )

                # scale amplitude by interpolating between the resting values and i
                # timestep value
                input_joint_angles = (
                    data_block[joint_ids, 0]
                    + (data_block[joint_ids, indices] - data_block[joint_ids, 0])
                    * amp[match_leg_to_joints]
                )
            else:
                input_joint_angles = data_block[joint_ids, 0]

            # Modify joint angles with hybrid input
            input_joint_angles = input_joint_angles + joint_angle_increment

            leg_in_stance = np.logical_or(
                indices[joints_to_leg] < swing_starts_in_order,
                indices[joints_to_leg] > stance_starts_in_order,
            )

            if adhesion:
                adhesion_signal = leg_in_stance
                # if leg in an hole or contacting with the wrong part of the leg
                # remove adhesion
                adhesion_signal[
                    np.logical_or(legs_in_hole, proximal_contact_leg)[
                        last_tarsalseg_to_adh_id
                    ]
                ] = 0.0
            else:
                adhesion_signal = np.zeros(6)

            action = {"joints": input_joint_angles, "adhesion": adhesion_signal}

            obs, _, _, _, _ = nmf.step(action)
            obs_list.append(obs)
            _ = nmf.render()
    except PhysicsError:
        print("Simulation diverged, resetting")

    if video_path:
        nmf.save_video(video_path, stabilization_time=0.2)

    return obs_list


####### Decentralized #########
def run_decentralized(
    nmf, seed, data_block, leg_swing_starts, leg_stance_starts, video_path=None
):
    nmf.reset()
    adhesion = nmf.sim_params.enable_adhesion

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

    # This serves to keep track of the advancement of each leg in the stepping
    # sequence
    stepping_advancement = np.zeros(len(LEGS)).astype(int)
    increment = 1.0

    swing_start = min(list(leg_swing_starts.values()))
    stance_start = max(list(leg_stance_starts.values()))

    leg_swing_starts_decentralized = {}
    leg_stance_starts_decentralized = {}

    for l in leg_stance_starts.keys():
        leg_swing_starts_decentralized[l] = swing_start
        leg_stance_starts_decentralized[l] = stance_start

    leg_scores = np.zeros(len(LEGS))
    obs_list = []

    adhesion_signal = np.zeros(6)
    legs_to_adhesion = np.array(
        [leg_corresp_id[ts[:2]] for ts in nmf.last_tarsalseg_names]
    )
    stance_starts_in_order = np.array(
        [leg_stance_starts[ts[:2]] for ts in nmf.last_tarsalseg_names]
    )
    swing_starts_in_order = np.array(
        [leg_swing_starts[ts[:2]] for ts in nmf.last_tarsalseg_names]
    )

    try:
        # Run the actual simulation
        for i in range(num_steps):
            # Decide in which leg to step
            initiating_leg = np.argmax(leg_scores)
            within_margin_legs = (
                leg_scores[initiating_leg] - leg_scores
                <= leg_scores[initiating_leg] * decentralized_controller.percent_margin
            )
            if i == N_STABILIZATION_STEPS + 1:
                # Will not start with the hindlegs => remove them from the within margin legs
                for l in LEGS:
                    if "H" in l:
                        within_margin_legs[leg_corresp_id[l]] = False

            # If multiple legs are within the margin choose randomly among those legs
            if np.sum(within_margin_legs) > 1:
                initiating_leg = np.random.choice(np.where(within_margin_legs)[0])

            # If the maximal score is zero or less (except for the first step after
            # stabilisation to initate the locomotion) or if the leg is already stepping
            if (
                leg_scores[initiating_leg] <= 0 and not i == N_STABILIZATION_STEPS + 1
            ) or stepping_advancement[initiating_leg] > 0:
                initiating_leg = None
            else:
                stepping_advancement[initiating_leg] += increment

            rounded_stepping_advancement = np.round(stepping_advancement).astype(int)
            joint_pos = data_block[
                joint_ids, rounded_stepping_advancement[match_leg_to_joints]
            ]

            if adhesion:
                # adhesion_signal = nmf.get_adhesion_vector()
                adhesion_signal = np.logical_or(
                    stepping_advancement[legs_to_adhesion] < swing_starts_in_order,
                    stepping_advancement[legs_to_adhesion] > stance_starts_in_order,
                )
            else:
                adhesion_signal = np.zeros(6)

            action = {"joints": joint_pos, "adhesion": adhesion_signal}
            obs, _, _, _, _ = nmf.step(action)
            nmf.render()
            obs_list.append(obs)

            stepping_advancement = decentralized_controller.update_stepping_advancement(
                stepping_advancement, LEGS, interp_step_duration, increment
            )

            (
                rule1_contrib,
                rule2_contrib,
                rule3_contrib,
            ) = decentralized_controller.compute_leg_scores(
                decentralized_controller.rule1_corresponding_legs,
                decentralized_controller.rule1_weight,
                decentralized_controller.rule2_corresponding_legs,
                decentralized_controller.rule2_weight,
                decentralized_controller.rule2_weight_contralateral,
                decentralized_controller.rule3_corresponding_legs,
                decentralized_controller.rule3_weight,
                decentralized_controller.rule3_weight_contralateral,
                stepping_advancement,
                leg_corresp_id,
                leg_stance_starts_decentralized,
                interp_step_duration,
                LEGS,
            )

            leg_scores = rule1_contrib + rule2_contrib + rule3_contrib
    except PhysicsError:
        print("Simulation diverged, resetting")

    # Return observation list
    if video_path:
        nmf.save_video(video_path, stabilization_time=0.2)

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
    import random

    exp_id = random.randint(0, 100)
    arena = get_arena(arena_type)
    nmf_params["spawn_pos"] = np.array([pos[0], pos[1], Z_SPAWN_POS])
    nmf = NeuroMechFlyMuJoCo(**nmf_params, arena=arena)
    # Generate CPG points
    CPG_path = (
        CPGpts_path / f"{arena_type}pts_seed{seed}_pos{pos[0]:.2f}_{pos[1]:.2f}.pkl"
    )
    if not CPG_path.is_file():
        CPG_obs_list = run_cpg(
            nmf,
            seed,
            data_block,
            match_leg_to_joints,
            joint_ids,
            leg_swings_starts,
            leg_stance_starts,
            video_path=CPG_path.with_suffix(".mp4"),
        )
        # Save as pkl
        with open(CPG_path, "wb") as f:
            pickle.dump(CPG_obs_list, f)

    arena = get_arena(arena_type)
    nmf = NeuroMechFlyMuJoCo(**nmf_params, arena=arena)
    # Generate hybrid points
    hybrid_path = (
        hybridpts_path / f"{arena_type}pts_seed{seed}_pos{pos[0]:.2f}_{pos[1]:.2f}.pkl"
    )
    if not hybrid_path.is_file():
        hybrid_obs_list = run_hybrid(
            nmf,
            seed,
            data_block,
            match_leg_to_joints,
            joint_ids,
            raise_leg,
            leg_swings_starts,
            leg_stance_starts,
            video_path=hybrid_path.with_suffix(".mp4"),
        )
        # Save as pkl
        with open(hybrid_path, "wb") as f:
            pickle.dump(hybrid_obs_list, f)

    arena = get_arena(arena_type)
    nmf = NeuroMechFlyMuJoCo(**nmf_params, arena=arena)
    # Generate Decentralized points
    decentralized_path = (
        decentralizedpts_path
        / f"{arena_type}pts_seed{seed}_pos{pos[0]:.2f}_{pos[1]:.2f}.pkl"
    )
    if not decentralized_path.is_file():
        decentralized_obs_list = run_decentralized(
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
        hybridpts_path / f"{arena_type}pts_seed{seed}_pos{pos[0]:.2f}_{pos[1]:.2f}.pkl"
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

    internal_seeds = list(range(20))
    assert args.n_exp <= len(internal_seeds), "Not enough internal seeds defined"
    internal_seeds = internal_seeds[: args.n_exp]

    # Initialize simulation but with flat terrain at the beginning to define
    # the swing and stance starts. Set high actuator kp to be able to overcome
    # obstacles
    timestep = 1e-4
    sim_params = MuJoCoParameters(
        timestep=timestep,
        render_mode="saved",
        render_playspeed=0.1,
        render_camera="Animat/camera_left_top_zoomout",
        enable_adhesion=adhesion,
        draw_adhesion=True,
        actuator_kp=ACTUATOR_KP,
        adhesion_gain=ADHESION_GAIN,
        adhesion_off_duration=ADHESION_OFF_DUR,
    )
    nmf = NeuroMechFlyMuJoCo(
        sim_params=sim_params,
        init_pose=stretched_pose,
        actuated_joints=all_leg_dofs,
    )

    # save metadata to yaml
    metadata = {
        "run_time": RUN_TIME,
        "n_stabilization_steps": N_STABILIZATION_STEPS,
        "coupling_strength": COUPLING_STRENGTH,
        "amp_rates": AMP_RATES,
        "target_amplitude": TARGET_AMPLITUDE,
        "legs": LEGS,
        "n_oscillators": N_OSCILLATORS,
        "adhesion_off_duration_decent": ADHESION_OFF_DUR,
        "adhesion_gain": ADHESION_GAIN,
        "start_ampl": START_AMPL,
        # "sim_params": nmf.sim_params,
    }
    metadata_path = Path(f"data/{arena_type}_metadata.yaml")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)

    # Load and process data block only once as this wont change
    (
        data_block,
        match_leg_to_joints,
        joint_ids,
        leg_swing_starts,
        leg_stance_starts,
    ) = get_data_block(nmf.timestep, nmf.actuated_joints)

    # Get the joint angles leading to a leg raise in each leg
    raise_leg = hybrid_controller.get_raise_leg(nmf)

    # Create folder to save data points
    CPGpts_path = Path(f"data/{arena_type}_CPGpts_adhesion{adhesion}_kp{ACTUATOR_KP}")
    CPGpts_path.mkdir(parents=True, exist_ok=True)
    decentralizedpts_path = Path(
        f"data/{arena_type}_Decentralizedpts_adhesion{adhesion}_kp{ACTUATOR_KP}"
    )
    decentralizedpts_path.mkdir(parents=True, exist_ok=True)
    hybridpts_path = Path(
        f"data/{arena_type}_hybridpts_adhesion{adhesion}_kp{ACTUATOR_KP}"
    )
    hybridpts_path.mkdir(parents=True, exist_ok=True)

    cs_placements = [
        f"{side}{pos}Tarsus{i}" for side in "LR" for pos in "FMH" for i in range(1, 6)
    ]
    cs_placements += [f"{side}{pos}Tibia" for side in "LR" for pos in "FMH"]

    sim_params.draw_adhesion = adhesion
    nmf_params = {
        "sim_params": sim_params,
        "init_pose": stretched_pose,
        "actuated_joints": all_leg_dofs,
        "contact_sensor_placements": np.array(cs_placements),
    }
    start_exps = time.time()
    print("Starting experiments")
    # Parallelize the experiment
    if args.n_procs > 1:
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
    args.add_argument("--n_procs", type=int, default=1, help="Number of processes")
    args = args.parse_args()

    main(args)
