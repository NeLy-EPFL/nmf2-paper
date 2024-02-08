# read turning_i.pkl files in a folder that contains a dictionary with the timestep
# run_time, fly_pos, fly_angs, fly_orientation, trurn_drive, turning_indices, cpg_phase_turn_start, turning_angle_change

# Create a dataframe with the following columns:
# - timestep
# - run_time
# - fly_pos
# - fly_angs
# - fly_orientation
# - trurn_drive
# - turning_indices
# - cpg_phase_turn_start
# - turning_angle_change
# - k

from pathlib import Path
import pickle
import pandas as pd
import numpy as np

benchmark_folder = Path("/Users/stimpfli/Desktop/nmf2-paper/revision_stepping/data/benchmark_turning_parallel_adhesion_R=2.0")
files = sorted(benchmark_folder.glob("*basic*.pkl"))
timesteps = []
run_times = []
l_drives = []
r_drives = []
turn_starts = []
turn_ends = []
cpg_phase_turn_starts = []
turning_angle_changes = []
ks = []

for file in files:
    with open(file, "rb") as f:
        data = pickle.load(f)
    timesteps.append(data["timestep"])
    run_times.append(data["run_time"])
    l_drives.append(data["l_drive"])
    r_drives.append(data["r_drive"])
    turn_starts.append(data["turn_start"])
    turn_ends.append(data["turn_ends"])
    cpg_phase_turn_starts.append(data["cpg_phase_turn_start"])
    turning_angle_changes.append(data["turning_angle_change"])
    ks.append(int(file.stem.split("_")[-1]))

df = pd.DataFrame({
    "timestep": timesteps,
    "run_time": run_times,
    "l_drive": l_drives,
    "r_drive": r_drives,
    "turn_start": turn_starts,
    "turn_ends": turn_ends,
    "cpg_phase_turn_start": cpg_phase_turn_starts,
    "turning_angle_change": turning_angle_changes,
    "k": ks
})

k_max = df["k"].max()
k_min = df["k"].min()

# save as csv file
out_file = benchmark_folder / f"turning_data_{k_min}-{k_max}_basic.csv"
df.to_csv(out_file, index=False)
assert out_file.exists()
print("Loaded basic data and saved as csv file")


files = sorted(benchmark_folder.glob("*full*.pkl"))
all_fly_pos = []
all_fly_angs = []
all_fly_orientations = []
exp_ids = []

for file in files:
    with open(file, "rb") as f:
        data = pickle.load(f)
    all_fly_pos.append(data["fly_pos"])
    all_fly_angs.append(data["fly_angs"])
    all_fly_orientations.append(data["fly_orientation"])
    n_pts = len(data["fly_pos"][0])
    exp_id = int(file.stem.split("_")[-1]) 
    exp_ids.append([exp_id]*n_pts)

all_fly_pos = np.transpose(np.array(all_fly_pos), (0, 2, 1)).reshape(-1, 2)
all_fly_angs = np.transpose(np.array(all_fly_angs), (0, 2, 1)).reshape(-1, 3)
all_fly_orientations = np.transpose(np.array(all_fly_orientations), (0, 2, 1)).reshape(-1, 2)
exp_ids = np.reshape(exp_ids, (-1, 1))

full_data = np.concatenate([all_fly_pos, all_fly_angs, all_fly_orientations, exp_ids], axis=1)

columns = [
    "fly_pos_x",
    "fly_pos_y",
    "fly_yaw",
    "fly_roll",
    "fly_pitch",
    "fly_orientation_x",
    "fly_orientation_y",
    "exp_ids"
]

df = pd.DataFrame(full_data, columns=columns)

k_max = df["exp_ids"].max().astype(int)
k_min = df["exp_ids"].min().astype(int)
out_file = benchmark_folder / f"turning_data_{k_min}-{k_max}_full.csv"
df.to_csv(out_file, index=False)
assert out_file.exists()
print("Loaded full data and saved as csv file")
