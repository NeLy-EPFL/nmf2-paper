from df3dPostProcessing import df3dPostProcess
from df3dPostProcessing.df3dPostProcessing import prism_skeleton_LP3D, flytracker_skel, df3d_skeleton
from df3dPostProcessing.utils import utils_plots


from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_FPS = 360
OUTPUT_FPS = 1000

def load_csv(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    df = df.drop(columns=["frame_idx", "video_id"])

    joints_todf3d = {"Coxa":"ThC", "Femur":"CTr", "Tibia":"FTi", "Tarsus":"TiTa", "Claw":"Claw"}
    df3d_output_dict = {}

    assert len(df.columns)%3 == 0, "Number of columns in csv file is not a multiple of 3 (can not be xyz)"
    points_3d = np.zeros((len(df), len(flytracker_skel), 3))
    thorax_ref = df[["Th_x", "Th_y", "Th_z"]].values

    for i, joint in enumerate(flytracker_skel):
        seg = joint[:2]
        #inv_seg = seg.replace("R", "L") if "R" in seg else seg.replace("L", "R")
        
        joint_ikjrec = joints_todf3d[joint[2:]]
        points_3d[:, i, 0] = df[f"{seg}-{joint_ikjrec}_x"]
        points_3d[:, i, 1] = df[f"{seg}-{joint_ikjrec}_y"]
        points_3d[:, i, 2] = df[f"{seg}-{joint_ikjrec}_z"]

    df3d_output_dict["points3d"] = points_3d.copy()
    return points_3d, len(df)

colors_dict = {
        "RF": (0.0, 0.0, 1.0),
        "RM": (0.0, 0.0, 0.75),
        "RH": (0.0, 0.0, 0.5),
        "LF": (1.0, 0.0, 0.0),
        "LM": (0.75, 0.0, 0.0),
        "LH": (0.5, 0.0, 0.0),
    }

def plot_legs_array(ax, data, skeleton, t_id=30, block=False):
    # Plot raw pose
    for i in range(len(skeleton)):
        currpt_leg = skeleton[i][:2]
        if currpt_leg in colors_dict.keys():
            color = colors_dict[currpt_leg]
            if not i == len(skeleton)-1:
                nextpt_leg = skeleton[i+1][:2]
                if not i == len(skeleton)-1 and currpt_leg == nextpt_leg:
                    ax.plot(data[t_id, i:i+2, 0],
                            data[t_id, i:i+2, 1],
                            data[t_id, i:i+2, 2], color=color)
                else:
                    ax.scatter(data[t_id, i, 0],
                            data[t_id, i, 1],
                            data[t_id, i, 2], color=color, label=currpt_leg)
            else:
                ax.scatter(data[t_id, i, 0],
                        data[t_id, i, 1],
                        data[t_id, i, 2], color=color, label = currpt_leg)
        else:
            ax.scatter(data[t_id, i, 0],
                    data[t_id, i, 1],
                    data[t_id, i, 2], color="black")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if block:
        plt.show(block = True)
    return 

def plot_legs_dict(ax, data, t_id=30, block=True):
    # Plot pose before alignment
    t_id = 30
    for leg, leg_data in data.items():
        if not "leg" in leg:
            continue
        color = colors_dict[leg.split("_")[0]]
        joints = list(leg_data.keys())
        for i in range(len(joints)):  
            joint = joints[i]      
            if not "Claw" in joint:
                joint_data = np.vstack([leg_data[joint][t_id, :],
                                    leg_data[joints[i+1]][t_id, :]])
                ax.plot(joint_data[:, 0], joint_data[:, 1], joint_data[:, 2], color = color)
            else:
                joint_data = leg_data[joint][t_id, :]
                ax.scatter(joint_data[0], joint_data[1], joint_data[2], color = color, label=leg)
    plt.legend()
    if block:
        plt.show(block = True)
    return

def plot_legs(ax, data, t_id, linestyle="--"):    
    # Plot pose before alignment
    for leg, leg_data in data.items():
        if not "leg" in leg:
            continue
        color = colors_dict[leg.split("_")[0]]
        joints = list(leg_data.keys())
        for i in range(len(joints)):  
            joint = joints[i]

            if not "Claw" in joint:
                if "raw_pos_aligned" in leg_data[joint]:                    
                    joint_data = np.vstack([leg_data[joint]["raw_pos_aligned"][t_id, :],
                                    leg_data[joints[i+1]]["raw_pos_aligned"][t_id, :]])
                elif "raw_pos" in leg_data[joint]:
                    joint_data = np.vstack([leg_data[joint]["raw_pos"][t_id, :],
                                    leg_data[joints[i+1]]["raw_pos"][t_id, :]])
                else:
                    joint_data = np.vstack([leg_data[joint][t_id, :],
                                        leg_data[joints[i+1]][t_id, :]])
                ax.plot(joint_data[:, 0], joint_data[:, 1], joint_data[:, 2], color=color, linestyle=linestyle)
            else:
                if "raw_pos_aligned" in leg_data[joint]:
                    joint_data = leg_data[joint]["raw_pos_aligned"][t_id, :]
                elif "raw_pos" in leg_data[joint]:
                    joint_data = leg_data[joint]["raw_pos"][t_id, :]
                else:
                    joint_data = leg_data[joint][t_id, :]
                ax.scatter(joint_data[0], joint_data[1], joint_data[2], color = color, label=leg)

def plot_align_base(base, align, base_path, skeleton, azim=0, elev=-90):
    n_frames_base = len(base)
    n_frames_align = len(align["RF_leg"]["Coxa"]["raw_pos_aligned"])
    transfer = n_frames_align/n_frames_base
    scaling_factors = []
    for i in range(len(skeleton)-1):
        leg_joint = skeleton[i]
        next_leg_joint = skeleton[i+1]
        leg = leg_joint[:2]
        next_leg = next_leg_joint[:2]
        if leg == next_leg:
            joint = leg_joint[2:]
            next_joint = next_leg_joint[2:]
            #distance between this joint and the next
            align_lengths = np.linalg.norm(align[f"{leg}_leg"][joint]["raw_pos_aligned"]-
                                           align[f"{next_leg}_leg"][next_joint]["raw_pos_aligned"], axis=1)
            base_lengths = np.linalg.norm(base[:, i]-base[:, i+1], axis=1) 
            scaling_factors.append(np.mean(align_lengths)/np.mean(base_lengths))
    base_scaling = np.median(scaling_factors)

    #scale the base so sizes ared similar
    base = base*base_scaling
    save_path = base_path / "align_base_comparison"
    save_path.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames_base):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=azim, elev=elev)
        plot_legs_array(ax, base, skeleton, i)
        plot_legs(ax, align, int(i*transfer), linestyle="--")
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlim(-2.5, 2.5)
        plt.legend()
        plt.savefig(save_path / f"{i}.png")
        plt.close()

    return 0

def align_swing_stance_dict(swing_stance_dict, new_start):
    new_swing_stance_dict = {}
    for leg, leg_data in swing_stance_dict.items():
        new_swing_stance_dict[leg] = {}
        for phase, phase_data in leg_data.items():
            new_swing_stance_dict[leg][phase] = phase_data - new_start
    return new_swing_stance_dict

def plot_swing_stance_alignes(align, time):
    """
    plot tarsus 3d pose and label swing phases
    """
    legs = ["RF", "RM", "RH", "LF", "LM", "LH"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    for i, leg in enumerate(legs):
        for j, coord in enumerate(["x", "y", "z"]):
            axs[i].plot(time, align[f"{leg}_leg"]["Tarsus"]["raw_pos_aligned"][:, j], label=f"Tarsus {coord}")
        swing_starts = align["swing_stance_time"][leg]["swing"]
        stance_starts = align["swing_stance_time"][leg]["stance"]
        if swing_starts[0] > stance_starts[0]:
            swing_starts = np.insert(swing_starts, 0, 0)
        if stance_starts[-1] < swing_starts[-1]:
            stance_starts = np.append(stance_starts, time[-1])
        for swing_start, stance_start in zip(swing_starts, stance_starts):
            axs[i].axvspan(swing_start, stance_start, alpha=0.5, color="green", label="swing")
        axs[i].legend()
        axs[i].set_title(leg)
    return fig, axs
        

data_path = Path("data/3D_pose_alfie/clean_3d_best_ventral_best_side.csv")
skel = "flytracker"
data, data_len = load_csv(data_path)
output_path = data_path.parent.parent / "df3dpp_output"
# save to pickle 
formatted_data_pat = output_path / "df3dpostprocess_reshape.pkl"
formatted_data_pat.parent.mkdir(parents=True, exist_ok=True)
with open(formatted_data_pat, "wb") as f:
    pickle.dump(data, f)

old_time = np.arange(0, data_len/BASE_FPS, 1/BASE_FPS)
new_time = np.arange(0, data_len/BASE_FPS, 1/OUTPUT_FPS)

# Read pose results and calculate 3d positions from 2d estimations
df3dpp = df3dPostProcess(str(formatted_data_pat), calculate_3d=False, skeleton=skel)
skeleton = flytracker_skel

interpolate = True
smoothing = True
window_size = 10
conv_casting = "valid"
# Align and scale 3d positions using the NeuroMechFly skeleton as template, data is interpolated 
align = df3dpp.align_to_template(interpolate=interpolate, smoothing=smoothing, original_time_step=1/BASE_FPS,
                                  new_time_step=1/OUTPUT_FPS, window_length=window_size, convolution_casting=conv_casting)
plot_align_base(df3dpp.raw_data_3d, align, output_path, skeleton)

if smoothing and conv_casting == "valid":
    # use valid convlutuion to avoid border effects needs to accordingly crop new_time
    print(len(new_time), max(new_time), min(new_time))
    new_time = new_time[window_size//2:-window_size//2+1] - new_time[window_size//2]
    print(len(new_time), max(new_time), min(new_time))


# Need to adjust the swing and stance times accordingly
swing_stance_dict_path = Path("data/swing_stance.pkl")
with open(swing_stance_dict_path, "rb") as f:
    swing_stance_dict = pickle.load(f)
if smoothing and conv_casting == "valid":
    swing_stance_dict = align_swing_stance_dict(swing_stance_dict, new_time[0])
align["swing_stance_time"] = swing_stance_dict
fig, axs = plot_swing_stance_alignes(align, new_time)
fig.savefig(output_path / "swing_stance_align.png")
plt.close()

# save aligned pose
if interpolate:
    align["meta"] = {"timestep": 1/OUTPUT_FPS, "source":str(data_path), "status":"aligned"}
else:
    align["meta"] = {"timestep": 1/BASE_FPS, "source":str(data_path), "status":"aligned"}
align["meta"]["interpolated"] = interpolate
align["meta"]["smoothed"] = smoothing
if smoothing:
    align["meta"]["window_size"] = window_size
    align["meta"]["conv_casting"] = conv_casting
out = output_path / "aligned_df3dpp.pkl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(align, f)

# Calculate joint angles from the leg (ThC-3DOF, CTr-2DOF, FTi-1DOF, TiTa-1DOF)
#df3dpp.skeleton = "flytracker"
angles = df3dpp.calculate_leg_angles()

# beg, end = 0, -1
#
# utils_plots.plot_legs_from_angles(
#         angles = angles,
#         data_dict= align,
#         exp_dir = str(output_path),
#         begin=beg,
#         end=end,
#         plane='xy',
#         saveImgs=True,
#         dir_name='km',
#         extraDOF={},
#         ik_angles=False,
#         pause=False,
#         lim_axes=True)

# utils_plots.plot_legs_from_angles(
#         angles = angles,
#         data_dict= align,
#         exp_dir = str(output_path),
#         begin=beg,
#         plane='xz',
#         saveImgs=True,
#         dir_name='km',
#         extraDOF={},
#         ik_angles=False,
#         pause=False,
#         lim_axes=True)

angles_nmf = {}
angles_nmf["meta"] = align["meta"]

joints_to_nmf = {"ThC":"Coxa", "CTr":"Femur", "FTi":"Tibia", "TiTa":"Tarsus1", "Claw":"Claw"}

for leg, leg_data in angles.items():
    for joint, joint_data in leg_data.items():
        leg = leg.split("_")[0]
        splitted_joint = joint.split("_")
        if "pitch" in joint:
            angles_nmf[f"joint_{leg}{joints_to_nmf[splitted_joint[0]]}"] = joint_data
        else:
            angles_nmf[f"joint_{leg}{joints_to_nmf[splitted_joint[0]]}_{splitted_joint[1]}"] = joint_data
    
# save to pickle
with open(output_path / "joint_angles_df3dpp.pkl", "wb") as f:
    pickle.dump(angles_nmf, f)