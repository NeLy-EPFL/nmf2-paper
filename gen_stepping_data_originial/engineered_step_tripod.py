# run the kinematic replay inspired for initial neuromechfly
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import sys
cwd = Path.cwd()
sys.path.append(str(cwd.parent))

from nmf_mujoco import NeuroMechFlyMuJoCo
from data import mujoco_groundwalking_model_path, mujoco_clean_groundwalking_model_path
from config import all_leg_dofs, kin_replay_leg_dofs

from tqdm import tqdm

import matplotlib.pyplot as plt

random_state = np.random.RandomState(0)


def parse_args():
    """ Parse arguments from command line """
    parser = argparse.ArgumentParser()
    # parser.add_argument('-b', '--behavior', default='walking')^
    # parser.add_argument('--record', dest='record', action='store_true')
    parser.add_argument('--show_collisions', dest='show_collisions', action='store_true')
    parser.add_argument('-fly', '--fly_number', default='1')
    parser.add_argument('-c', '--camera', default='1')

    return parser.parse_args()


###### Ball Config Transformations Constants ########
# Betwen pyBullet and MuJoCo, the coordinate systems are different.
thorax_MuJoCo_rel_pos = [495.64525485038757, -18.091375008225441, 1296.6440916061401]
thorax_Pybullet_rel_pos = [0.48076672828756273, -5.750451226305131e-07, 1.2107710354030132]
reg = np.polyfit(thorax_Pybullet_rel_pos, thorax_MuJoCo_rel_pos, 1)
regress = lambda x: reg[0] * x + reg[1]

# Pybullet uses units
meters = 1000
kilograms = 1000

# Offset from pybullet (alreadyin meters)
pybullet_model_offset = [0., 0, 11.2e-3]


def format_ball_terrain_config(terrain_config):
    """ Format the terrain config for the ball terrain """

    terrain_config["mass"] = 54.6e-6 * kilograms
    mjc_pos = [regress(x * meters + off) for x, off in zip(terrain_config["position"], pybullet_model_offset)]
    terrain_config["position"] = (tuple(mjc_pos), (1, 0, 0, 0))
    terrain_config["radius"] = regress(terrain_config["radius"] * meters)
    terrain_config["fly_placement"] = ((0, 0, 0), (0, np.deg2rad(terrain_config["angle"]), 0))
    terrain_config["lateral_friction"] = 1.3

    return terrain_config


########## Joint Angles Matching ##########
joint_names_correspondence = {'ThC': 'Coxa',
                              'CTr': 'Femur',
                              'FTi': 'Tibia',
                              'TiTa': 'Tarsus1'}


def set_compliant_Tarsus(physics, all_actuators, acutated_joints, kp_val=5, stiff_val=0.0, damping_val=100):
    # Set the Tarsus2345 to be compliant by setting the kp stifness and damping to a low value
    for a in all_actuators:
        if "position" in a.name and "Tarsus" in a.name and not "Tarsus1" in a.name:
            physics.model.actuator('Animat/' + a.name).gainprm[0] = kp_val

    for j in acutated_joints:
        if j is None:
            continue
        if "Tarsus" in j and not "Tarsus1" in j:
            physics.model.joint("Animat/" + j).stiffness = stiff_val
            physics.model.joint("Animat/" + j).damping = damping_val


def load_joint_angles(path, actuated_joints, time_step, pybullet_time_step, starting_time, run_time):
    """
    Load joint angles from a file
    Read in the dictionary of joint angles and order them in an array ordered by actuated joints
    Crop the data between the starting and ending times
    Interpolate the joint angles to the simulation time step

    :param run_time:
    :param pybullet_time_step:
    :param actuated_joints:
    :param path:
    :param time_step:
    :param starting_time:
    :return:
    """
    pybullet_joint_angles = np.load(path, allow_pickle=True)
    n_joints_data = len([leg + joint for leg in pybullet_joint_angles for joint in pybullet_joint_angles[leg]])

    assert len(actuated_joints) == n_joints_data, (f"Number of joints in the data ({n_joints_data}) does not "
                                                   f"match the number of actuated joints ({len(actuated_joints)})")

    end_time = starting_time + run_time
    simulation_time = np.arange(0, run_time, time_step)
    pybullet_time = np.arange(0, run_time, pybullet_time_step)

    joint_angles = np.zeros((len(actuated_joints), int(run_time / time_step)))
    for leg in pybullet_joint_angles:
        for joint in pybullet_joint_angles[leg]:
            split_joint = joint.split('_')
            matching_joint = f"joint_{leg[:2]}{joint_names_correspondence[split_joint[0]]}"
            if not "pitch" in split_joint:
                matching_joint += f"_{split_joint[1]}"

            selected_joint_angles = pybullet_joint_angles[leg][joint][int(starting_time / pybullet_time_step):
                                                                      int(end_time / pybullet_time_step)]
            joint_angles[actuated_joints.index(matching_joint)] = np.interp(simulation_time,
                                                                            pybullet_time,
                                                                            selected_joint_angles)

    return joint_angles


def main(args):
    """ Main function """

    behavior = 'walking'
    terrain = 'flat'
    #camera = f"Animat/camera_{args.camera}" if not args.camera == '1' else 1
    camera = "Animat/camera_left_top"

    run_time = 1.0
    data_time_step = 5e-4
    time_step = 1e-4
    starting_time = 3.0

    # Paths for data
    data_path = Path.cwd() / f'data/joint_tracking/{behavior}/fly{args.fly_number}/df3d'
    print(data_path)

    angles_path = next(iter(data_path.glob('joint_angles*.pkl')))
    # velocity_path = glob.glob(data_path + '/joint_velocities*.pkl')[0]

    terrain_config = {}

    # At some point replace all contacts by only relevant ones
    # Implement contact visualization

    render_config = {"camera_id": camera, "playspeed": 0.1, "window_size": (1280, 720)}

    # Setting the fixed joint angles to default values, can be altered to
    # change the appearance of the fly

    # Setting up the paths for the SDF and POSE files
    model_path = mujoco_clean_groundwalking_model_path
    output_dir = Path.cwd() / f'output/kinematic_replay/{behavior}/fly{args.fly_number}'

    # Simulation options
    sim_options = {"render_mode": 'saved',
                   "render_config": render_config,
                   "model_path": model_path,
                   "actuated_joints": kin_replay_leg_dofs,
                   "timestep": time_step,
                   "output_dir": output_dir,
                   "terrain": terrain,
                   "terrain_config": terrain_config,
                   "control": 'position',
                   "init_pose": "stretch",
                   "show_collisions": args.show_collisions
                   }

    nmf = NeuroMechFlyMuJoCo(**sim_options)

    joint_angles = load_joint_angles(angles_path, nmf.actuated_joints, time_step, data_time_step, starting_time,
                                     run_time)
    friction = 1
    for g in nmf.all_geom:
        if "collision" in g:
            nmf.physics.model.geom("Animat/" + g).friction = [friction, 0.005, 0.0001]
    nmf.physics.model.geom("ground").friction = [friction, 0.005, 0.0001]

    stiffness = 2500
    for j in nmf.actuated_joints:
        if j is None:
            continue
        nmf.physics.model.joint("Animat/" + j).stiffness = stiffness

    grav = -9.810 * 1e5
    nmf.physics.model.opt.gravity = [0, 0, grav]

    """for geom in nmf.physics.model.geom:
        if "collision" in geom.name:
            nmf.physics.model.geom(geom).mass /= kilograms"""

    set_compliant_Tarsus(nmf.physics, nmf.actuators, nmf.all_joints)

    nmf.physics.reset()

    nmf.output_dir = nmf.output_dir.parent / f"friction{friction}_gravity{grav}_stiff{stiffness}"
    nmf.output_dir.mkdir(parents=True, exist_ok=True)

    zero_pose = {}
    for joint in nmf.actuated_joints:
        zero_pose[joint] = joint_angles[nmf.actuated_joints.index(joint), 0]
    for joint in nmf.all_joints:
        if joint in nmf.init_pose and joint in zero_pose and "M" in joint and "Tibia" in joint:
            nmf.init_pose[joint] = zero_pose[joint]

    nmf.set_init_pose(nmf.init_pose)

    n_timesteps = joint_angles.shape[1]

    # order is the same
    tarsus_geoms = ["Animat/" + geom for geom in nmf.all_geom if "collision" in geom and "Tarsus" in geom]
    tarsus_bodies = [geom[:-10] for geom in tarsus_geoms]
    print(tarsus_bodies)
    nmfid2arrayid = dict(zip([i for i, geom in enumerate(nmf.all_geom)
                              if "collision" in geom and "Tarsus" in geom], np.arange(len(tarsus_geoms))))
    arrayid2names = dict(zip(np.arange(len(tarsus_geoms)), tarsus_geoms))
    contact_array = np.zeros((len(tarsus_geoms), n_timesteps))

    position_array = np.zeros((len(tarsus_geoms)*3, n_timesteps))

    floor_id = nmf.all_geom.index("ground")

    for i in tqdm(range(n_timesteps)):
        obs, info = nmf.step({'joints': joint_angles[:, i]})
        if nmf.physics.data.ncon > 1:
            for collision in nmf.physics.data.contact:
                if ((collision.geom1 in nmfid2arrayid or collision.geom2 in nmfid2arrayid) and
                        (collision.geom1 == floor_id or collision.geom2 == floor_id)):
                    tarsal_geom = collision.geom1 if collision.geom1 in nmfid2arrayid else collision.geom2
                    contact_array[nmfid2arrayid[tarsal_geom], i] = 1
            for k, tar_body in enumerate(tarsus_bodies):
                position_array[k*3:(k+1)*3, i] = nmf.physics.named.data.xpos[tar_body]
        nmf.render()

    contact_data = pd.DataFrame(contact_array.T, columns=tarsus_bodies)
    pos_col = pd.MultiIndex.from_product([tarsus_bodies,
                                          ["x", "y", "z"]], names=["body", "coordinate"])
    position_data = pd.DataFrame(position_array.T, columns=pos_col)
    contact_data.to_csv(nmf.output_dir / "contact_data.csv")
    position_data.to_csv(nmf.output_dir / "position_data.csv")

    for i in range(len(tarsus_geoms)):
        plt.plot(np.arange(n_timesteps) * time_step, contact_array[i, :], label=[arrayid2names[i]])
    plt.legend()
    plt.show(block=True)

    nmf.close()


if __name__ == "__main__":
    """ Main """
    # parse cli arguments
    cli_args = parse_args()
    main(cli_args)
