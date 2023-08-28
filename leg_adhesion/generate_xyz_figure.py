from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dm_control.utils import transformations
import yaml

# Generate graphs of the fly Thorax x,y x,z, y,z positions over time for the front and side gravity vectors
# for each fly
# Colors for the front gravity red and getting darker as the fly is more inclined
# Colors for the side gravity green and getting darker as the fly is more inclined

# Load the data points
base_path = Path(f"data/slope_front")
# get all pkl files in the folder
pkl_files = list(base_path.glob("*.pkl"))

# load metadata.yaml
yml_path = base_path / "metadata.yml"
with open(yml_path, 'r') as f:
    metadata = yaml.safe_load(f)
n_stabilization_steps = metadata["n_stabilization_steps"]
gravity_switching_step = metadata["gravity_switching_step"]

xyz_positions_list = []
xyz_positions_list_rotated = []
slope_vector = []

if "front" in base_path.name:
    rotation_axis = "y"
elif "side" in base_path.name:
    rotation_axis = "x"

# Apply each rotation to the x,y,z positions
# get rotation matrix
if rotation_axis == "x":
    rotation_matrix = lambda slope: transformations.rotation_x_axis(np.deg2rad(slope))
elif rotation_axis == "y":
    rotation_matrix = lambda slope: transformations.rotation_y_axis(np.deg2rad(slope))

for pkl_file in pkl_files:
    # The gravity direction is encoded in the file name
    slope = int(pkl_file.stem.split("_")[-2])
    slope_vector.append(slope)
    # Load the data points
    obs_list = np.load(pkl_file, allow_pickle=True)
    # Get the x,y,z positions
    xyz_positions = np.array([obs["fly"][0] for obs in obs_list[n_stabilization_steps:]])
    xyz_positions_list.append(xyz_positions)
    # Rotate the x,y,z positions if the gravtiy is changed i > gravity_switching_step-n_stabilization_steps
    xyz_positions_rotated = np.array(
        [np.dot(rotation_matrix(-1*slope), xyz_position )
         for xyz_position in xyz_positions]
    )
    #xyz_positions_rotated += xyz_positions[gravity_switching_step - n_stabilization_steps] - xyz_positions_rotated[gravity_switching_step - n_stabilization_steps]
    xyz_positions_rotated[: gravity_switching_step - n_stabilization_steps] = xyz_positions[
        : gravity_switching_step - n_stabilization_steps
    ]
    xyz_positions_rotated[gravity_switching_step - n_stabilization_steps:] += xyz_positions_rotated[
        gravity_switching_step - n_stabilization_steps - 1
    ] - xyz_positions_rotated[gravity_switching_step - n_stabilization_steps]

    """for i, coord in enumerate(["x", "y", "z"]):
        plt.figure()
        plt.plot(xyz_positions[:, i], label="original")
        plt.plot(xyz_positions_rotated[:, i], label="rotated")
        plt.title(f"{coord} position, slope {slope}")
        plt.legend()
        plt.show(block=True)"""
    xyz_positions_list_rotated.append(xyz_positions_rotated)

# pad the shorter contact forces with nan
max_length = max([len(p) for p in xyz_positions_list])
for i, p in enumerate(xyz_positions_list):
    if len(p) < max_length:
        xyz_positions_list[i] = np.pad(p, ((0, max_length - len(p)), (0, 0)),
                                               mode="constant", constant_values=np.nan)
for i, pr in enumerate(xyz_positions_list_rotated):
    if len(pr) < max_length:
        xyz_positions_list_rotated[i] = np.pad(pr, (
        (0, max_length - len(pr)), (0, 0)), mode="constant",
                                       constant_values=np.nan)
order = np.argsort(slope_vector)
slope_vector = np.array(slope_vector)[order]
xyz_positions_list = np.array(xyz_positions_list)[order]
xyz_positions_list_rotated = np.array(xyz_positions_list_rotated)[order]

if "side" in base_path.name:
    slope_vector = slope_vector[:-3]

# Plot the x,y,z positions over time for each fly
# Make the base color light red and make it darker as the fly is more inclined
if "front" in base_path.name:
    base_color = np.array([1, 0, 0])
elif "side" in base_path.name:
    base_color = np.array([0, 1, 0])

coord_corresp = {"x": 0, "y": 1, "z": 2}

for coord1, coord2 in [["x", "y"], ["x", "z"], ["y", "z"]]:
    fig, axs = plt.subplots(1, 2, figsize=(25, 7))
    coord1_values = xyz_positions_list[:, :, coord_corresp[coord1]]
    coord2_values = xyz_positions_list[:, :, coord_corresp[coord2]]

    coord1_values_rotated = xyz_positions_list_rotated[:, :, coord_corresp[coord1]]
    coord2_values_rotated = xyz_positions_list_rotated[:, :, coord_corresp[coord2]]

    for i, slope in enumerate(slope_vector):
        if slope not in [0, 30, 60, 90]:
            continue
        color = base_color * (1 - i / len(slope_vector))
        axs[0].plot(
            coord1_values[i][5000:10000],
            coord2_values[i][5000:10000],
            label=f"{slope}, {rotation_axis}",
            color=color,
        )
        axs[1].plot(
            coord1_values_rotated[i][5000:10000],
            coord2_values_rotated[i][5000:10000],
            label=f"{slope}, {rotation_axis}",
            color=color,
        )

    # No labels on ax 0
    print(f"Length after 5000 steps: {coord1_values[0][5000:].size} steps")

    axs[0].set_title("Original")
    axs[0].set_xlabel(f"{coord1} position (mm)")
    axs[0].set_ylabel(f"{coord2} position (mm)")
    axs[0].set_ylim([0, 2])
    axs[0].legend()
    axs[0].set_aspect('equal')
    axs[1].set_title("Rotated")
    axs[1].set_xlabel(f"{coord1} position (mm)")
    axs[1].set_ylabel(f"{coord2} position (mm)")
    axs[1].legend()
    axs[1].set_aspect('equal')

    fig.savefig(base_path / f"{coord1}_{coord2}.png")
    fig.savefig(base_path / f"{coord1}_{coord2}.pdf")
    print(f"Saved {base_path / f'{coord1}_{coord2}.png/pdf'}")
