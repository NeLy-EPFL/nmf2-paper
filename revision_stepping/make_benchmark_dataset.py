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

benchmark_folder = Path("/home/stimpfling/nmf2-paper/revision_stepping/data/benchmark_turning_parallel_adhesion_R=2.0")
files = sorted(benchmark_folder.glob("*.pkl"))
timesteps = []
run_times = []
fly_poss = []
fly_angss = []
fly_orientations = []
turn_drives = []
turning_indices = []
cpg_phase_turn_starts = []
turning_angle_changes = []
ks = []

for file in files:
    with open(file, "rb") as f:
        data = pickle.load(f)
    timesteps.append(data["timestep"])
    run_times.append(data["run_time"])
    fly_poss.append(data["fly_pos"])
    fly_angss.append(data["fly_angs"])
    fly_orientations.append(data["fly_orientation"])
    turn_drives.append(data["trurn_drive"])
    turning_indices.append(data["turning_indices"])
    cpg_phase_turn_starts.append(data["cpg_phase_turn_start"])
    turning_angle_changes.append(data["turning_angle_change"])
    ks.append(int(file.stem.split("_")[-1]))

df = pd.DataFrame({
    "timestep": timesteps,
    "run_time": run_times,
    "fly_pos": fly_poss,
    "fly_angs": fly_angss,
    "fly_orientation": fly_orientations,
    "trurn_drive": turn_drives,
    "turning_indices": turning_indices,
    "cpg_phase_turn_start": cpg_phase_turn_starts,
    "turning_angle_change": turning_angle_changes,
    "k": ks
})

k_max = df["k"].max()
k_min = df["k"].min()

# save as csv file
out_file = benchmark_folder / f"turning_data_{k_min}-{k_max}.csv"
df.to_csv(out_file, index=False)
assert out_file.exists()
