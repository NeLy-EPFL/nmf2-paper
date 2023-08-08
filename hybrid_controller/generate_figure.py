import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

base_path = Path("Data_points")
all_folders = base_path.glob("*pts*")

# Now in each folder for each pkl file open the observation list
# From the observation list get obs_list[-1]["fly"][0] the last fly position
# Generate a scatter plot for each condition with the last fly z position¨
# Chnage the color if the adhesion is on
# Save the figure in the base directory
kp_selector = "_kp40.0"

save_path = Path(f"panel{kp_selector}.png")

controller = ["CPG", "Decentralized"]
terrain = ["flat", "blocks", "gapped", "mixed"]
adhesion = [True, False]

conditions = [(c, t) for c in controller for t in terrain]
n_conditions = len(conditions)

fig, ax = plt.subplots(figsize=(20, 10))

all_data_pts = []
all_colors = np.tile(["r", "b"], n_conditions)

for controller, terrain in conditions:
    for adh in adhesion:
        path = base_path / f"{terrain}_{controller}pts_adhesion{adh}{kp_selector}"
        if not path.is_dir():
            print(f"Path {path} does not exist")
            continue
        all_pkl = list(path.glob("*.pkl"))
        assert len(all_pkl) > 1, f"Path {path} does not contain any pkl file"
        data_pts = []
        for pkl_file in all_pkl:
            with open(pkl_file, "rb") as f:
                obs_list = pickle.load(f)

            data_pts.append(obs_list[-1]["fly"][0][0] - obs_list[0]["fly"][0][0])

        all_data_pts.append(data_pts)

# Plot the data in a boxplot with visible points way

for i in range(n_conditions):
    if i == 0:
        ax.scatter(
            np.ones(len(all_data_pts[2 * i])) * i,
            all_data_pts[2 * i],
            c=all_colors[2 * i],
            label="adhesion ON",
        )
        ax.scatter(
            np.ones(len(all_data_pts[2 * i + 1])) * i,
            all_data_pts[2 * i + 1],
            c=all_colors[2 * i + 1],
            label="adhesion OFF",
        )
    else:
        ax.scatter(
            np.ones(len(all_data_pts[2 * i])) * i,
            all_data_pts[2 * i],
            c=all_colors[2 * i],
        )
        ax.scatter(
            np.ones(len(all_data_pts[2 * i + 1])) * i,
            all_data_pts[2 * i + 1],
            c=all_colors[2 * i + 1],
        )

# set the xticks to strings with the condition names
ax.set_xticks(np.arange(n_conditions))
ax.set_xticklabels([f"{c}_{t}" for c, t in conditions])

ax.set_xlim(-0.5, n_conditions + 0.5)
ax.set_ylabel("x distance travelled")
# Set the legend for red is adhesion and blue is not
ax.legend()

# save the figure
plt.savefig(save_path)
