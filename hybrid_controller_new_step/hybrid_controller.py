import numpy as np
from flygym.examples.cpg_controller import CPGNetwork

from flygym import Fly, Camera, SingleFlySimulation
from flygym.examples import PreprogrammedSteps
from flygym.arena import FlatTerrain, GappedTerrain, BlocksTerrain, MixedTerrain
from dm_control.rl.control import PhysicsError

from tqdm import trange
import pickle


from scipy.interpolate import CubicSpline


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


def run_hybrid_simulation(sim, cpg_network, preprogrammed_steps, run_time):


    step_phase_multipler = {}

    for leg in preprogrammed_steps.legs:
        swing_start, swing_end = preprogrammed_steps.swing_period[leg]

        step_points = [swing_start, np.mean([swing_start, swing_end]), swing_end, np.mean([swing_end, 2*np.pi]), 2*np.pi]
        increment_vals = [0, 0.5, 0, 0.25, 0]

        step_phase_multipler[leg] = CubicSpline(step_points, increment_vals, bc_type="periodic")

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
    inf_list = []

    retraction_perisitance_counter = np.zeros(6)

    retraction_persistance_counter_hist = np.zeros((6, target_num_steps))

    adhesion_on_counter = np.zeros(6)
    
    for k in trange(target_num_steps):
        # retraction rule: does a leg need to be retracted from a hole?
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
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

        all_net_corrections = np.zeros(6)
        retraction_rule_on = np.zeros(6)
        stumbling_rule_on = np.zeros(6)

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
            all_net_corrections[i] = net_correction
            if leg[0] == "R":
                net_correction *= right_leg_inversion[i]
            
            net_correction *= step_phase_multipler[leg](cpg_network.curr_phases[i])

            my_joints_angles += net_correction * correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
                leg, cpg_network.curr_phases[i]
            )
            # stumbling_rule_leg = (force_proj < stumbling_force_threshold).any() and not(
                # leg_to_correct_retraction == i)
            
            # retraction_rule_leg = i == leg_to_correct_retraction or retraction_perisitance_counter[i] > 0
            # retraction_rule_on[i] = retraction_rule_leg
            # stumbling_rule_on[i] = stumbling_rule_leg


            # # No adhesion in stumbling or retracted
            # rule_active = np.logical_not(stumbling_rule_leg or retraction_rule_leg)
        #     my_adhesion_onoff *= rule_active
            
        #     # increment 
        #     adhesion_on_counter[i] += my_adhesion_onoff
        #     # reset if adhesion is off
        #     adhesion_on_counter[i] *= my_adhesion_onoff
            
        #     my_adhesion_onoff = min(1, float(adhesion_on_counter[i])/50)

            adhesion_onoff.append(my_adhesion_onoff)

        # if k >= 1874:
        #     adhesion_onoff = [0, 0, 0, 0, 0, 0]

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff),
        }

        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            obs["qacc"] = sim.physics.data.qacc.copy()
            obs["qfrc_act"] = sim.physics.data.qfrc_actuator.copy()
            info["net_corrections"] = all_net_corrections
            info["action_jnts"] = action["joints"]
            info["phase"] = cpg_network.curr_phases.copy()
            info["action_adhesion"] = action["adhesion"].copy()

            info["retraction_ruleon"] = retraction_rule_on
            info["stumbling_ruleon"] = stumbling_rule_on
            info["adhesion_on_counter"] = adhesion_on_counter.copy()
            obs_list.append(obs)
            inf_list.append(info)

            sim.render()
        except PhysicsError:
            print("Simulation was interupted because of a physics error")
            return obs_list, inf_list, True

    return obs_list, inf_list, False


if __name__ == "__main__":
    run_time = 1.5
    timestep = 1e-4

    id = 0
    
    np.random.seed(0)

    # Generate random positions
    max_x = 4.0
    shift_x = 2.0
    max_y = 4.0
    shift_y = 2.0
    positions = np.zeros((20, 3))
    positions[:, :2] = np.random.rand(20, 2)

    positions[:, 0] = positions[:, 0] * max_x - shift_x
    positions[:, 1] = positions[:, 1] * max_y - shift_y
    positions[:, 2] = 0.5

    preprogrammed_steps = PreprogrammedSteps()

    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in preprogrammed_steps.legs
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    for id in range(5):

        np.random.seed(id)

        fly = Fly(
            enable_adhesion=True,
            draw_adhesion=True,
            init_pose="stretch",
            control="position",
            contact_sensor_placements=contact_sensor_placements,
            actuator_forcerange = (-65.0, 65.0),
            spawn_pos=positions[id],
        )

    #     from pathlib import Path
    #     fly = Fly(
    #     enable_adhesion=True,
    #     draw_adhesion=True,
    #     contact_sensor_placements=contact_sensor_placements,
    #     xml_variant=Path("/Users/stimpfli/Desktop/flygym_other/flygym/data/mjcf/neuromechfly_seqik_kinorder_ypr_scaledmass.xml"),
    #     actuator_forcerange=[-65*1000, 65*1000],
    #     joint_damping= 0.15*1000,
    #     joint_stiffness= 0.15*1000,
    #     tarsus_damping=1e-2*1000,
    #     tarsus_stiffness=7.5*1000,
    #     non_actuated_joint_damping=1*1000,
    #     non_actuated_joint_stiffness=1*1000,
    #     actuator_gain=40*1000,
    #     adhesion_force=40*1000,
    #     #contact_solimp=(0.9, 0.95, 0.001, 0.5, 2),
    #     #contact_solref= (0.02, 1)
    # )

        terrain = GappedTerrain() if id % 2 == 0 else BlocksTerrain()

        cam_right = Camera(fly=fly, play_speed=0.1, camera_id="Animat/camera_right")
        cam_left = Camera(fly=fly, play_speed=0.1, camera_id="Animat/camera_left")
        cam_top = Camera(fly=fly, play_speed=0.1, camera_id="Animat/camera_top")
        sim = SingleFlySimulation(
            fly=fly,
            cameras=[cam_right, cam_top, cam_left],
            timestep=timestep,
            arena=terrain,
        )   

        cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            seed=id,
        )

        obs, info = sim.reset()
        print(f"Spawning fly at {obs['fly'][0]} mm")

        obs_list, inf_list, had_physics_error = run_hybrid_simulation(
            sim, cpg_network, preprogrammed_steps, run_time
        )

        x_pos = obs_list[-1]["fly"][0][0]
        print(f"Final x position: {x_pos:.4f} mm")

        # Save video
        cam_right.save_video(f"./outputs/hybrid_{id}_controller_right.mp4", 0)
        cam_top.save_video(f"./outputs/hybrid_{id}_controller_top.mp4", 0)
        cam_left.save_video(f"./outputs/hybrid_{id}_controller_left.mp4", 0)

        # save the physics error observations
        with open(f"./outputs/hybrid_{id}_obs_list.pkl", "wb") as f:
            pickle.dump(obs_list, f)
        with open(f"./outputs/hybrid_{id}_inf_list.pkl", "wb") as f:
            pickle.dump(inf_list, f)
    print("Saved obs_list.pkl")