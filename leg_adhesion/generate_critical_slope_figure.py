import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import pickle

base_data_path = Path("datapts_gainslope")
gains = []
seeds = []
critical_slopes = []
reached_end_all = []
reached_end_critical = []

is_critical = True

x_pos_thr = -6 #two body lengths

# Go through all pkl files
gain_folders = base_data_path.glob("seed_*/gain_*")
for gain_folder in gain_folders:
    gains.append(float(gain_folder.name.split("_")[1]))
    seeds.append(int(gain_folder.parent.name.split("_")[1]))
    metadata_file = gain_folder / "slope_metadata.pkl"
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    slope_files = sorted(gain_folder.glob("slope_[0123456789.]*.pkl"), key=lambda x: float(x.name.split("_")[1].split(".")[0]))
    for slope_file in slope_files:
        with open(slope_file, "rb") as f:
            try:
                obs_list = pickle.load(f)
            except EOFError:
                print("Error loading file:", slope_file)
                reached_end_all.append(False)
                continue
        # Check if the end was reached
        is_complete = len(obs_list) == np.ceil((metadata["run_time"]+metadata["stabilisation_dur"])/metadata["timestep"])
        reached_end_all.append(is_complete)

        if len(gains) == len(critical_slopes):
            #print(gain_folder.name, slope_file.name, "already processed")
            continue
        
        # When gravity is reversed, xpos is not zeros
        # check wether the x pos is threshold away from the position it was when gravity was reversed

        # In case something went wrong, check if the fly was flipped (maybe with low slopes, the fly flips over but does not go reverse x due to friciton)

        reverse_id = int(np.ceil(metadata["slope_reversal_time"]/metadata["timestep"]))

        fly_xvel = np.array([obs["fly"][1, 2] for obs in obs_list])
        fly_xpos = np.array([obs["fly"][0, 0] for obs in obs_list])
        fly_xpos_treverse_is_origin = fly_xpos[reverse_id:] - fly_xpos[reverse_id]
        is_critical = np.any(fly_xpos_treverse_is_origin < x_pos_thr)
        if not is_critical:
            # check wether the fly just flipped over
            fly_ang = np.array([obs["fly"][2][1:] for obs in obs_list])
            is_critical = np.any(np.abs(fly_ang) > np.pi/2)
            if is_critical:
                print("Fly flipped over")
        if is_critical:
            critical_slopes.append(float(slope_file.name.split("_")[1].split(".")[0]))
            reached_end_critical.append(is_complete)
            # DO NOT BREAK JUST TO CHECK IF SOME SIMULATION DID NOT REACH THE END

print(gains, critical_slopes, reached_end_critical)

fig, ax = plt.subplots(figsize=(12, 7))

ids_reached_end = np.where(reached_end_critical)[0]
ids_did_not_reach_end = np.where(np.logical_not(reached_end_critical))[0]

scatter = plt.scatter(np.array(gains)+np.random.rand(len(gains))*1.5, critical_slopes, c=seeds,
                       s=(np.array(reached_end_critical).astype(float)+0.5)*25)
ax.legend(["seed_{}".format(seed) for seed in seeds])
legend = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Seeds")
ax.add_artist(legend)

# produce a legend with a cross-section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, [True, False], loc="lower right", title="Had physics error")

ax.set_xlabel("Adhesion maximal force (mN)")
ax.set_ylabel("Critical slope (deg)")
ax.set_title("Critical slope vs adhesion gain")
fig.savefig(base_data_path / "critical_slope_vs_gain.png")

reached_end_all = np.array(reached_end_all).reshape((len(gains), len(list(gain_folder.glob("slope_[0123456789]*.pkl")))))
print("All reached end:", reached_end_all)