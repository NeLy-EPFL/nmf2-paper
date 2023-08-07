from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from flygym.util.config import all_tarsi_links

from scipy.signal import find_peaks

import yaml

# Generate a boxplot like figure for each fly contact forces in the xyx direction
# Use a different color for each coordinate x,y,z
# Load the data points
base_path = Path(f"Data_points/slope_side")
# get all pkl files in the folder
pkl_files = list(base_path.glob("*.pkl"))

# load metadata.yaml
yml_path = base_path / "metadata.yml"
with open(yml_path, 'r') as f:
    metadata = yaml.safe_load(f)
n_stabilization_steps = metadata["n_stabilization_steps"]
gravity_switching_step = metadata["gravity_switching_step"]
slope_vector = []
contact_forces = []

for pkl_file in pkl_files:
    # The gravity direction is encoded in the file name
    slope = pkl_file.stem.split("_")[-2]
    slope_vector.append(slope)
    # Load the data points
    obs_list = np.load(pkl_file, allow_pickle=True)
    # Get the x,y,z positions
    xyz_contacts = np.array([obs["contact_forces"] for obs in obs_list[:gravity_switching_step]])
    contact_forces.append(xyz_contacts)

# pad the shorter contact forces with nan
max_length = max([len(c) for c in contact_forces])
for i, c in enumerate(contact_forces):
    if len(c) < max_length:
        contact_forces[i] = np.pad(c, ((0, max_length - len(c)), (0, 0), (0, 0)), mode="constant", constant_values=np.nan)

order = np.argsort(slope_vector)
slope_vector = np.array(slope_vector)[order]
contact_forces = np.array(contact_forces)[order]

#sum contact forces per leg
legs = ["LF", "LM", "LH", "RF", "RM", "RH"]
n_legs = len(legs)
legs_contact_sensors = [[i for i, seg in enumerate(all_tarsi_links) if seg.startswith(leg)] for leg in legs]
print(contact_forces.shape)
contact_forces = contact_forces[:, :, :, legs_contact_sensors].sum(axis=-1)
print(contact_forces.shape)


separate_legs = True
legs_subplots = True

scatter = 0.1 if separate_legs and not legs_subplots else 0.45
if legs_subplots:
    assert separate_legs, "Legs subplots can only be used if the legs are separated"

colors = ["red", "green", "blue"]
leg_offset = np.linspace(-0.3, 0.3, n_legs)

if not legs_subplots:
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharey=True)
else:
    fig, axs = plt.subplots(3, n_legs, figsize=(20, 10), sharey=True)

for j in range(3):
    data_array = []
    positions = []
    for i in range(0, len(slope_vector)):
        pos_leg_data = []
        neg_leg_data = []

        n_pts_legs = 0
        for k, leg in enumerate(legs):
            #y_array = np.abs(contact_forces[i][:, j, k].flatten())
            y_array = contact_forces[i][:, j, k].flatten()
            # drop nan values
            y_array = y_array[~np.isnan(y_array)]
            # peak force
            standardized_y = (y_array - np.mean(y_array))/(np.std(y_array))
            height = np.quantile(standardized_y, 0.95)
            pos_peaks, _ = find_peaks(standardized_y, prominence=1.0, width=50,
                                  height=height)
            height = np.quantile(-standardized_y, 0.95)
            neg_peaks, _ = find_peaks(-standardized_y, prominence=1.0, width=50,
                                  height=height)
            #peaks = np.concatenate([pos_peaks, neg_peaks])
            #peaks = pos_peaks
            """if i == 0:
                fig_int = plt.figure()
                plt.plot(standardized_y)
                plt.plot(peaks, standardized_y[peaks], "x")
                plt.show(block =True)"""
            #peak_y_array = y_array[peaks]
            pos_peaks = y_array[pos_peaks]
            neg_peaks = y_array[neg_peaks]
            peak_y_array = np.concatenate([pos_peaks, neg_peaks])
            n_points = len(peak_y_array)
            n_pts_legs += n_points
            x_array = np.ones(n_points) * i
            if separate_legs:
                if legs_subplots:
                    axs[j, k].scatter(x_array+(np.random.rand(n_points)*scatter - scatter/2), peak_y_array,
                               color=colors[j],
                               s=10,
                               alpha=0.2)
                    axs[j, k].boxplot([pos_peaks, neg_peaks], positions=[i , i],
                                      widths=scatter)
                    if i == 0:
                        axs[j, k].axhline(0, color="black", linestyle="--")
                else:
                    axs[j].scatter(x_array+(np.random.rand(n_points)*scatter - scatter/2) + leg_offset[k],
                                   peak_y_array,
                                   color=colors[j],
                                   s=10,
                                   alpha=0.2)
                    data_array.append(pos_peaks)
                    positions.append(i + leg_offset[k])
                    data_array.append(neg_peaks)
                    positions.append(i + leg_offset[k])
            else:
                pos_leg_data.extend(pos_peaks)
                neg_leg_data.extend(neg_peaks)
        if not separate_legs:
            leg_data = np.concatenate([pos_leg_data, neg_leg_data])
            axs[j].scatter(np.ones(n_pts_legs) * i + np.random.rand(n_pts_legs)*scatter - scatter/2,
                           leg_data,
                           color=colors[j],
                           s=10,
                           alpha=0.2)
            data_array.append(pos_leg_data)
            positions.append(i)
            data_array.append(neg_leg_data)
            positions.append(i)

    if not separate_legs:
        axs[j].boxplot(data_array, positions=positions, widths=scatter)
        axs[j].axhline(0, color="black", linestyle="--")
    #axs[j].bar(positions, [np.mean(d) for d in data_array],
    #           yerr=[np.std(d) for d in data_array],
    #           width=scatter, color=colors[j])
    #flat_data = np.concatenate(data_array)
    #min = np.quantile(flat_data, 1e-5) - 5
    #max = np.quantile(flat_data, 1-1e-5) + 5
    #axs[j].set_ylim([min, max])

if not legs_subplots:
    axs[0].set_title("x")
    axs[1].set_title("y")
    axs[2].set_title("z")
    # hide x axis for the first two plots
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks(range(len(slope_vector)))
    axs[2].set_xticklabels([f"{slope}" for slope in slope_vector])
else:
    for k, leg in enumerate(legs):
        axs[0, k].set_title(leg)
        axs[0, k].set_xticks([])
        axs[1, k].set_xticks([])
        axs[2, k].set_xticks(range(len(slope_vector)))
        axs[2, k].set_xticklabels([f"{slope}" for slope in slope_vector])
plt.tight_layout()
fig.savefig(base_path / f"contact_forces.png")
