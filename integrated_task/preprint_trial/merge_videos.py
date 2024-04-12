import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

spawn_positions = [
    (-1, -1, 0.2),
    (-1, 0, 0.2),
    (-1, 1, 0.2),
    (0, -1, 0.2),
    (0, 0, 0.2),
    (0, 1, 0.2),
    (1, -1, 0.2),
    (1, 0, 0.2),
    (1, 1, 0.2),
]
width = None
height = None
fps = None
# fourcc = None
num_train_steps = 266000
out_path = "outputs/navigation_task_merged.mp4"


# Merge video
all_frames = []
for spawn_pos in spawn_positions:
    in_path = f"outputs/{num_train_steps}_{spawn_pos[0]}_{spawn_pos[1]}_{spawn_pos[2]}/video.mp4"
    print(f"Reading {in_path}...")
    cap = cv2.VideoCapture(in_path)
    if width is None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    else:
        assert width == int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        assert height == int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert fps == int(cap.get(cv2.CAP_PROP_FPS))
        # assert fourcc == int(cap.get(cv2.CAP_PROP_FOURCC))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    for _ in range(fps // 2):  # add a little pause
        all_frames.append(all_frames[-1])
    cap.release()

print(f"Writing to {out_path}...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
for frame in all_frames:
    out.write(frame)
out.release()


## Make trajectory figure
spawn_positions = [
    (-1, -1, 0.2),
    (-1, 0, 0.2),
    (-1, 1, 0.2),
    (0, -1, 0.2),
    (0, 0, 0.2),
    (0, 1, 0.2),
    (1, -1, 0.2),
    (1, 0, 0.2),
    (1, 1, 0.2),
]
num_train_steps = 266000

trajectories = []
print(f"Making trajectories figure...")
for spawn_pos in spawn_positions:
    in_path = f"outputs/{num_train_steps}_{spawn_pos[0]}_{spawn_pos[1]}_{spawn_pos[2]}/info_hist.pkl"
    with open(in_path, "rb") as f:
        info_hist = pickle.load(f)
    trajectory = [np.array(spawn_pos[:2])] + [x["fly_pos"] for x in info_hist[1:]]
    trajectories.append(np.array(trajectory))

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
for trajectory in trajectories:
    last_pos_dist = np.linalg.norm(trajectory[-1, :] - np.array([15, 0]))
    if last_pos_dist < 3:
        color = "tab:blue"
        end_marker = "v"
    else:
        color = "tab:red"
        end_marker = "x"
    plt.plot(
        [trajectory[0, 0]], trajectory[0, 1], marker="o", color=color, markersize=3
    )
    plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.5, color=color)
    plt.plot(
        [trajectory[-1, 0]],
        trajectory[-1, 1],
        marker=end_marker,
        color=color,
        markersize=3,
    )
target_range = patches.Circle(
    (15, 0),
    radius=3,
    edgecolor="tab:orange",
    facecolor="none",
    linestyle="--",
    linewidth=0.5,
)
target = patches.Circle(
    (15, 0), radius=0.5, edgecolor="tab:orange", facecolor="tab:orange"
)
obstacle = patches.Circle((7.5, 0), radius=1, edgecolor="black", facecolor="black")
ax.add_patch(target_range)
ax.add_patch(target)
ax.add_patch(obstacle)
ax.set_aspect("equal")
ax.set_xlim([-2, 19])
ax.set_ylim([-7, 7])
fig.savefig("outputs/trajectories.pdf", transparent=True)
