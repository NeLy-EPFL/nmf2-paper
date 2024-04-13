from itertools import chain, product
from pathlib import Path
from typing import Dict, List

import numpy as np
from dm_control.rl.control import PhysicsError
from flygym import Camera, Fly, SingleFlySimulation
from flygym.arena import BlocksTerrain, FlatTerrain, GappedTerrain, MixedTerrain
from flygym.examples import PreprogrammedSteps
from flygym.examples.cpg_controller import CPGNetwork
from flygym.examples.rule_based_controller import (
    RuleBasedSteppingCoordinator,
    construct_rules_graph,
)
from flygym.preprogrammed import get_cpg_biases
from joblib import Parallel, delayed
from tqdm import tqdm

import sys

sys.path.append("../control_signal")
from colored_fly import ColoredFly as Fly

########### SCRIPT PARAMS ############
arenas = ["flat", "gapped", "blocks", "mixed"]
controllers = ["cpg", "rule_based", "hybrid"]
save_dir = Path("outputs/obs")
video_dir = Path("outputs/videos")
metadata_path = Path("outputs/metadata.npz")

########### SIM PARAMS ############
ENVIRONEMENT_SEED = 0

n_exp = 20
max_x = 4.0
shift_x = 2.0
max_y = 4.0
shift_y = 2.0
Z_SPAWN_POS = 0.5

timestep = 1e-4
run_time = 1.5

########### CPG PARAMS ############
intrinsic_freqs = np.ones(6) * 12
intrinsic_amps = np.ones(6) * 1
phase_biases = get_cpg_biases("tripod")
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 20

########### RULE BASED PARAMS ############
rule_based_step_dur = 1 / np.mean(intrinsic_freqs)
weights = {
    "rule1": -10,
    "rule2_ipsi": 2.5,
    "rule2_contra": 1,
    "rule3_ipsi": 3.0,
    "rule3_contra": 2.0,
}
rules_graph = construct_rules_graph()

########### HYBRID PARAMS ############
correction_vectors = {
    # "leg pos": (Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1)
    # unit: radian
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}
right_leg_inversion = [1, -1, -1, 1, -1, 1, 1]
stumbling_force_threshold = -1
correction_rates = {"retraction": (800, 700), "stumbling": (2200, 2100)}
max_increment = 80
retraction_persistance = 20
persistance_init_thr = 20


########### FUNCTIONS ############
def save_obs_list(save_path, obs_list: List[Dict]):
    array_dict = {}
    for k in obs_list[0]:
        array_dict[k] = np.array([i[k] for i in obs_list])
    np.savez_compressed(save_path, **array_dict)


def get_arena(arena: str):
    if arena == "flat":
        return FlatTerrain()
    elif arena == "gapped":
        return GappedTerrain()
    elif arena == "blocks":
        # seed for randomized block heights
        return BlocksTerrain(rand_seed=ENVIRONEMENT_SEED)
    elif arena == "mixed":
        return MixedTerrain(rand_seed=ENVIRONEMENT_SEED)


def run_hybrid(
    sim: SingleFlySimulation,
    cpg_network: CPGNetwork,
    preprogrammed_steps: PreprogrammedSteps,
    run_time: float,
):
    retraction_correction = np.zeros(6)
    stumbling_correction = np.zeros(6)

    detected_segments = ["Tibia", "Tarsus1", "Tarsus2"]
    stumbling_sensors = {leg: [] for leg in preprogrammed_steps.legs}
    for i, sensor_name in enumerate(sim.fly.contact_sensor_placements):
        leg = sensor_name.split("/")[1][:2]  # sensor_name: eg. "Animat/LFTarsus1"
        segment = sensor_name.split("/")[1][2:]
        if segment in detected_segments:
            stumbling_sensors[leg].append(i)
    stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}

    obs, info = sim.reset()

    target_num_steps = int(run_time / sim.timestep)
    obs_list = []

    retraction_perisitance_counter = np.zeros(6)
    retraction_persistance_counter_hist = np.zeros((6, target_num_steps))

    phys_error = False

    for k in range(target_num_steps):
        # retraction rule: does a leg need to be retracted from a hole?
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.06:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
            if retraction_correction[leg_to_correct_retraction] > persistance_init_thr:
                retraction_perisitance_counter[leg_to_correct_retraction] = 1
        else:
            leg_to_correct_retraction = None

        # update persistance counter
        retraction_perisitance_counter[retraction_perisitance_counter > 0] += 1
        retraction_perisitance_counter[
            retraction_perisitance_counter > retraction_persistance
        ] = 0
        retraction_persistance_counter_hist[:, k] = retraction_perisitance_counter

        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []

        for i, leg in enumerate(preprogrammed_steps.legs):
            # update amount of retraction correction
            if (
                i == leg_to_correct_retraction or retraction_perisitance_counter[i] > 0
            ):  # lift leg
                increment = correction_rates["retraction"][0] * sim.timestep
                retraction_correction[i] += increment
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (0, 1, 1))
            else:  # condition no longer met, lower leg
                decrement = correction_rates["retraction"][1] * sim.timestep
                retraction_correction[i] = max(0, retraction_correction[i] - decrement)
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", None)

            # update amount of stumbling correction
            contact_forces = obs["contact_forces"][stumbling_sensors[leg], :]
            fly_orientation = obs["fly_orientation"]
            # force projection should be negative if against fly orientation
            force_proj = np.dot(contact_forces, fly_orientation)
            if (force_proj < stumbling_force_threshold).any():
                increment = correction_rates["stumbling"][0] * sim.timestep
                stumbling_correction[i] += increment
                sim.fly.change_segment_color(sim.physics, f"{leg}Femur", (1, 0, 1))
            else:
                decrement = correction_rates["stumbling"][1] * sim.timestep
                stumbling_correction[i] = max(0, stumbling_correction[i] - decrement)
                sim.fly.change_segment_color(sim.physics, f"{leg}Femur", None)

            # retraction correction is prioritized
            if retraction_correction[i] > 0:
                net_correction = retraction_correction[i]
                stumbling_correction[i] = 0
            else:
                net_correction = stumbling_correction[i]

            # get target angles from CPGs and apply correction
            my_joints_angles = preprogrammed_steps.get_joint_angles(
                leg, cpg_network.curr_phases[i], cpg_network.curr_magnitudes[i]
            )
            net_correction = np.clip(net_correction, 0, max_increment)
            if leg[0] == "R":
                net_correction *= right_leg_inversion[i]

            my_joints_angles += net_correction * correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
                leg, cpg_network.curr_phases[i]
            )
            # No adhesion in stumbling or retracted
            my_adhesion_onoff *= np.logical_not(
                (force_proj < stumbling_force_threshold).any()
                or i == leg_to_correct_retraction
            )
            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            obs_list.append(obs)
            sim.render()
        except PhysicsError:
            phys_error = True
            break

    return obs_list, phys_error


def run_rule_based(
    sim: SingleFlySimulation,
    controller: RuleBasedSteppingCoordinator,
    run_time: float,
):
    obs, info = sim.reset()
    obs_list = []
    physic_error = False
    for _ in range(int(run_time / sim.timestep)):
        controller.step()
        joint_angles = []
        adhesion_onoff = []
        for leg, phase in zip(controller.legs, controller.leg_phases):
            joint_angles_arr = controller.preprogrammed_steps.get_joint_angles(
                leg, phase
            )
            joint_angles.append(joint_angles_arr.flatten())
            adhesion_onoff.append(
                controller.preprogrammed_steps.get_adhesion_onoff(leg, phase)
            )
        action = {
            "joints": np.concatenate(joint_angles),
            "adhesion": np.array(adhesion_onoff),
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            obs_list.append(obs)
            sim.render()
        except PhysicsError:
            physic_error = True
            break

    return obs_list, physic_error


def run_cpg(
    sim: SingleFlySimulation,
    cpg_network: CPGNetwork,
    preprogrammed_steps: PreprogrammedSteps,
    run_time: float,
):
    obs, info = sim.reset()
    obs_list = []
    phys_error = False
    for _ in range(int(run_time / sim.timestep)):
        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(preprogrammed_steps.legs):
            my_joints_angles = preprogrammed_steps.get_joint_angles(
                leg, cpg_network.curr_phases[i], cpg_network.curr_magnitudes[i]
            )
            joints_angles.append(my_joints_angles)
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
                leg, cpg_network.curr_phases[i]
            )
            adhesion_onoff.append(my_adhesion_onoff)
        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            sim.render()
            obs_list.append(obs)
        except PhysicsError:
            phys_error = True
            break
    return obs_list, phys_error


def run(arena: str, seed: int, pos: np.ndarray, verbose: bool = False):
    save_paths = {c: save_dir / f"{c}_{arena}_{seed}.npz" for c in controllers}
    video_paths = {c: video_dir / f"{c}_{arena}_{seed}.mp4" for c in controllers}

    if all(p.exists() for p in chain(save_paths.values(), video_paths.values())):
        return
    else:
        pass

    preprogrammed_steps = PreprogrammedSteps()
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in preprogrammed_steps.legs
        for segment in ["Tibia"] + [f"Tarsus{i}" for i in range(1, 6)]
    ]

    # Initialize the simulation
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose="stretch",
        control="position",
        spawn_pos=pos,
        contact_sensor_placements=contact_sensor_placements,
    )
    terrain = get_arena(arena)
    cam = Camera(fly=fly, play_speed=0.1)
    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=terrain,
    )

    # run cpg simulation
    sim.reset()
    cpg_network = CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
        seed=seed,
    )
    cpg_network.reset()
    obs_list, phys_error = run_cpg(sim, cpg_network, preprogrammed_steps, run_time)
    displacements = obs_list[-1]["fly"][0] - obs_list[0]["fly"][0]

    if verbose:
        print(f"CPG experiment {seed}: {displacements}", end="")
        print(" ended with physics error" if phys_error else "")

    cam.save_video(video_paths["cpg"], 0)
    save_obs_list(save_paths["cpg"], obs_list)

    # run rule based simulation
    preprogrammed_steps.duration = rule_based_step_dur
    controller = RuleBasedSteppingCoordinator(
        timestep=timestep,
        rules_graph=rules_graph,
        weights=weights,
        preprogrammed_steps=preprogrammed_steps,
        seed=seed,
    )
    sim.reset()

    obs_list, phys_error = run_rule_based(sim, controller, run_time)
    displacements = obs_list[-1]["fly"][0] - obs_list[0]["fly"][0]

    if verbose:
        print(f"Rule based experiment {seed}: {displacements}", end="")
        print(" ended with physics error" if phys_error else "")

    cam.save_video(video_paths["rule_based"], 0)
    save_obs_list(save_paths["rule_based"], obs_list)

    # run hybrid simulation
    np.random.seed(seed)
    sim.reset()
    cpg_network.random_state = np.random.RandomState(seed)
    cpg_network.reset()
    obs_list, phys_error = run_hybrid(sim, cpg_network, preprogrammed_steps, run_time)
    displacements = obs_list[-1]["fly"][0] - obs_list[0]["fly"][0]

    if verbose:
        print(f"Hybrid experiment {seed}: {displacements}", end="")
        print(" ended with physics error" if phys_error else "")

    cam.save_video(video_paths["hybrid"], 0)
    save_obs_list(save_paths["hybrid"], obs_list)


########### MAIN ############
if __name__ == "__main__":
    # Create directories
    for d in [save_dir, video_dir, metadata_path.parent]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate random positions
    rng = np.random.RandomState(ENVIRONEMENT_SEED)
    positions = rng.rand(n_exp, 2) * (max_x, max_y) - (shift_x, shift_y)
    positions = np.column_stack((positions, np.full(n_exp, Z_SPAWN_POS)))

    # Save metadata to yaml
    np.savez_compressed(metadata_path, run_time=run_time, positions=positions)

    # Run experiments
    it = [(a, s, p) for a, (s, p) in product(arenas, enumerate(positions))]
    Parallel(n_jobs=-1)(delayed(run)(*i, True) for i in tqdm(it))
