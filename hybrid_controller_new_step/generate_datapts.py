import numpy as np
from pathlib import Path
import pickle

import argparse
import multiprocessing
import time


from flygym.examples.cpg_controller import CPGNetwork
from flygym.examples.rule_based_controller import (
    RuleBasedSteppingCoordinator,
    construct_rules_graph,
)
from flygym import Fly, Camera, SingleFlySimulation
from flygym.examples import PreprogrammedSteps
from flygym.arena import FlatTerrain, GappedTerrain, BlocksTerrain, MixedTerrain

from dm_control.rl.control import PhysicsError
from scipy.interpolate import CubicSpline

import yaml

########### SCRIPT PARAMS ############

out_folder = Path("simulation_results")
out_folder.mkdir(parents=True, exist_ok=True)
video_base_path = out_folder / "videos"
ENVIRONEMENT_SEED = 0

########### SIM PARAMS ############
Z_SPAWN_POS = 0.5

timestep = 1e-4
run_time = 1.5

########### CPG PARAMS ############
intrinsic_freqs = np.ones(6) * 12
intrinsic_amps = np.ones(6) * 1
phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 20

########### RULE BASED PARAMS ############
rule_based_step_dur = 1/np.mean(intrinsic_freqs)
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

correction_rates = {"retraction": (800, 700), "stumbling": (2200, 1800)}
max_increment = 80
retraction_persistance = 20
persistance_init_thr = 20

########### FUNCTIONS ############
def get_arena(arena_type):
    if arena_type == "flat":
        return FlatTerrain()
    elif arena_type == "gapped":
        return GappedTerrain()
    elif arena_type == "blocks":
        return BlocksTerrain(
            rand_seed=ENVIRONEMENT_SEED,
        )
    elif arena_type == "mixed":
        return MixedTerrain(rand_seed=ENVIRONEMENT_SEED)  # seed for randomized block heights

def run_hybrid_simulation(sim, cpg_network, preprogrammed_steps, run_time):

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

    adhesion_on_counter = np.zeros(6)
    
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
        retraction_perisitance_counter[retraction_perisitance_counter > retraction_persistance] = 0
        retraction_persistance_counter_hist[:, k] = retraction_perisitance_counter

        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []

        for i, leg in enumerate(preprogrammed_steps.legs):

            # update amount of retraction correction
            if i == leg_to_correct_retraction or retraction_perisitance_counter[i] > 0:  # lift leg
                increment = correction_rates["retraction"][0] * sim.timestep
                retraction_correction[i] += increment
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (1, 0, 0, 1))
            else:  # condition no longer met, lower leg
                decrement = correction_rates["retraction"][1] * sim.timestep
                retraction_correction[i] = max(0, retraction_correction[i] - decrement)
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (0.5, 0.5, 0.5, 1))

            # update amount of stumbling correction
            contact_forces = obs["contact_forces"][stumbling_sensors[leg], :]
            fly_orientation = obs["fly_orientation"]
            # force projection should be negative if against fly orientation
            force_proj = np.dot(contact_forces, fly_orientation)
            if (force_proj < stumbling_force_threshold).any():
                increment = correction_rates["stumbling"][0] * sim.timestep
                stumbling_correction[i] += increment
                sim.fly.change_segment_color(sim.physics, f"{leg}Femur", (1, 0, 0, 1))
            else:
                decrement = correction_rates["stumbling"][1] * sim.timestep
                stumbling_correction[i] = max(0, stumbling_correction[i] - decrement)
                sim.fly.change_segment_color(sim.physics, f"{leg}Femur", (0.5, 0.5, 0.5, 1))

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
            stumbling_rule_leg = (force_proj < stumbling_force_threshold).any() and not(
                leg_to_correct_retraction == i)
            
            retraction_rule_leg = i == leg_to_correct_retraction or retraction_perisitance_counter[i] > 0

            # No adhesion in stumbling or retracted
            rule_active = np.logical_not(stumbling_rule_leg or retraction_rule_leg)
            my_adhesion_onoff *= rule_active
            
            # increment 
            adhesion_on_counter[i] += my_adhesion_onoff
            # reset if adhesion is off
            adhesion_on_counter[i] *= my_adhesion_onoff
            
            my_adhesion_onoff = min(1, float(adhesion_on_counter[i])/50)

            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff),
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            obs_list.append(obs)
            sim.render()
        except PhysicsError:
            return obs_list, True          
            break 
        
    return obs_list, False

def run_rule_based_simulation(sim, controller, run_time):
    obs, info = sim.reset()
    obs_list = []
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
            return obs_list, True
        
    return obs_list, False

def run_cpg_simulation(nmf, cpg_network, preprogrammed_steps, run_time):
    obs, info = nmf.reset()
    obs_list = []
    for _ in range(int(run_time / nmf.timestep)):
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
            obs, reward, terminated, truncated, info = nmf.step(action)
            nmf.render()
            obs_list.append(obs)
        except PhysicsError:
            return obs_list, True
        
    return obs_list, False

    
def run_experiment(seed, pos, arena_type, out_path):

    pos_str = "_".join([f"{p:.2f}" for p in pos])
    preprogrammed_steps = PreprogrammedSteps()

    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in preprogrammed_steps.legs
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
        ]
    
    # Initialize the simulation
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose="stretch",
        control="position",
        spawn_pos=pos,
        contact_sensor_placements=contact_sensor_placements,
        actuator_forcerange = (-65.0, 65.0),

    )
    terrain = get_arena(arena_type)
    cam = Camera(fly=fly, play_speed=0.1)
    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=terrain,
    )       

    # run cpg simulation
    np.random.seed(seed)
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
    cpg_network.random_state = np.random.RandomState(seed)
    cpg_network.reset()
    cpg_obs_list, cpg_phys_error = run_cpg_simulation(sim, cpg_network, preprogrammed_steps, run_time)
    print(f"CPG experiment {seed}: {cpg_obs_list[-1]['fly'][0] - cpg_obs_list[0]['fly'][0]}", end="")
    if cpg_phys_error:
         print(" ended with physics error")
         if len(cpg_obs_list) > 0:
            cam.save_video(video_base_path / arena_type  / "cpg" / f"exp_{seed}_{pos_str}.mp4", 0)
    else:
        print("")
        cam.save_video(video_base_path / arena_type  / "cpg" / f"exp_{seed}_{pos_str}.mp4", 0)

    with open(out_path / "cpg" / f"exp_{seed}_{pos_str}.pkl", "wb") as f:
        pickle.dump(cpg_obs_list, f)

    # run hybrid simulation
    np.random.seed(seed)
    sim.reset()
    cpg_network.random_state = np.random.RandomState(seed)
    cpg_network.reset()
    hybrid_obs_list, hybrid_physics_error = run_hybrid_simulation(sim, cpg_network, preprogrammed_steps, run_time)
    print(f"Hybrid experiment {seed}: {hybrid_obs_list[-1]['fly'][0] - hybrid_obs_list[0]['fly'][0]}", end="")
    if hybrid_physics_error:
         print(" ended with physics error")
         if len(hybrid_obs_list) > 0:
            cam.save_video(video_base_path / arena_type  / "hybrid" / f"exp_{seed}_{pos_str}.mp4", 0)
    else:
        print("")
        cam.save_video(video_base_path / arena_type  / "hybrid" / f"exp_{seed}_{pos_str}.mp4", 0) 
    # Save the data
    with open(out_path / "hybrid" / f"exp_{seed}_{pos_str}.pkl", "wb") as f:
        pickle.dump(hybrid_obs_list, f)

    # run rule based simulation    
    preprogrammed_steps.duration = rule_based_step_dur
    controller = RuleBasedSteppingCoordinator(
                    timestep=timestep,
                    rules_graph=rules_graph,
                    weights=weights,
                    preprogrammed_steps=preprogrammed_steps,
                )
    np.random.seed(seed)
    sim.reset()
    
    rule_based_obs_list, ruled_based_physics_error = run_rule_based_simulation(sim, controller, run_time)
    print(f"Rule based experiment {seed}: {rule_based_obs_list[-1]['fly'][0] - rule_based_obs_list[0]['fly'][0]}", end="")
    if ruled_based_physics_error:
         print(" ended with physics error")
         if len(rule_based_obs_list) > 0:
            cam.save_video(video_base_path / arena_type  / "rule_based" / f"exp_{seed}_{pos_str}.mp4", 0)
    else:
        print("")
        cam.save_video(video_base_path / arena_type  / "rule_based" / f"exp_{seed}_{pos_str}.mp4", 0) 
    # Save the data
    with open(out_path / "rule_based" / f"exp_{seed}_{pos_str}.pkl", "wb") as f:
        pickle.dump(rule_based_obs_list, f)


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
    n_procs = args.n_procs
    out_path = args.out_folder
    for cont in ["cpg", "rule_based", "hybrid"]:
        (out_path / cont).mkdir(parents=True, exist_ok=True)
        (video_base_path / arena_type / cont).mkdir(parents=True, exist_ok=True)

    np.random.seed(ENVIRONEMENT_SEED)

    # Generate random positions
    max_x = 4.0
    shift_x = 2.0
    max_y = 4.0
    shift_y = 2.0
    positions = np.zeros((args.n_exp, 3))
    positions[:, :2] = np.random.rand(args.n_exp, 2)

    positions[:, 0] = positions[:, 0] * max_x - shift_x
    positions[:, 1] = positions[:, 1] * max_y - shift_y
    positions[:, 2] = Z_SPAWN_POS

    internal_seeds = np.arange(args.n_exp, dtype=int)
    assert args.n_exp <= len(internal_seeds), "Not enough internal seeds defined"
    internal_seeds = internal_seeds[: args.n_exp]
    
    # save metadata to yaml
    metadata = {
        "run_time": run_time,
    }
    metadata_path = Path(f"data/{arena_type}_metadata.yaml")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)

    start_exps = time.time()
    print("Starting experiments")
    # Parallelize the experiment
    if args.n_procs > 1:
        task_configuration = [
            (
                seed,
                pos,
                arena_type,
                out_path

            )
            for seed, pos in zip(internal_seeds, positions)
        ]
        with multiprocessing.Pool(n_procs) as pool:
            pool.starmap(run_experiment, task_configuration)
        pool.join()
        pool.close()
    else:
        for pos, seed in zip(positions, internal_seeds):
            run_experiment(
                seed,
                pos,  
                arena_type,
                out_path
            )

    print(f"{args.n_exp} experiments took {time.time()-start_exps:.2f} seconds")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--arena", type=str, default="flat", help="Type of arena to use")
    args.add_argument(
        "--n_exp", type=int, default=10, help="Number of experiments to run"
    )
    args.add_argument("--n_procs", type=int, default=1, help="Number of processes")
    args = args.parse_args()
    args.out_folder = out_folder / args.arena
    main(args)
