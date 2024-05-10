import matplotlib.pyplot as plt
import numpy as np
import imageio
from pathlib import Path
from tempfile import gettempdir
from shutil import rmtree
from tqdm import trange
from subprocess import run


# Configs
data_path = Path("../../flygym/notebooks/Tarsus")
glob_pattern = "changeang{}_Ftars1kp{}_jointstiffness0.0.mp4"
y_var_name = "Modify joint angles (+30deg):"
y_var_values = {"YES":"True", "NO":"False"}
x_var_name = "Tarsus1 Kp:"
x_var_values = {"18.0":"18.0", "45.0":"45.0"}
num_trials = 1

temp_dir = Path(gettempdir()) / "tpm"
output_path = Path("outputs/kpt1_jamodif_comparison.mp4")
output_path.parent.mkdir(exist_ok=True)
video_paths = {}

# Index video files
video_paths = {}
for y, tag_y in y_var_values.items():
    for x, tag_x in x_var_values.items():
        matches = list(data_path.glob(glob_pattern.format(tag_y, tag_x)))
        matches = sorted(matches)
        assert (
                len(matches) == num_trials
        ), f"Found {len(matches)} videos for {y_var_name}{tag_y}" \
           f", {x_var_name}{tag_x}, {glob_pattern.format(tag_y, tag_x)}"
        video_paths[(y, x)] = matches

# Load videos
fps = None
num_frames = None
all_videos = {}
for (y, x), video_path in video_paths.items():
    print(f"Loading videos for  {y_var_name}{y}, {x_var_name}{x}")
    all_videos[(y, x)] = []
    for path in video_path:
        vid = imageio.get_reader(path)
        metadata = vid.get_meta_data()
        if fps is None:
            fps = metadata["fps"]
        else:
            assert fps == metadata["fps"], "FPS is not the same"
        frames = []
        while True:
            try:
                frame = vid.get_next_data()
                frames.append(frame)
            except IndexError:
                break
        if num_frames is None:
            num_frames = len(frames)
        else:
            assert num_frames == len(frames), "Number of frames is not the same"
        all_videos[(y, x)].append(frames)

def draw_frame(curr_video, frame_within_video):
    fig, axs = plt.subplots(len(y_var_values), len(x_var_values), figsize=(20, 12))
    fig.subplots_adjust(
        left=0.1, right=0.95, top=0.9, bottom=0.05, hspace=0.03, wspace=0.03
    )

    kp_label_config = []
    modif_label_config = []
    for i, y in enumerate(y_var_values):
        for j, x in enumerate(x_var_values):
            ax = axs[i, j]
            ax.imshow(all_videos[(y, x)][curr_video][frame_within_video])
            ax.axis("off")
            ax.text(7, 60, f"Trial {curr_video + 1}", fontname="Arial", fontsize=20)

            pos = ax.get_position()
            y_pos_label = pos.y0 + 0.5 * pos.height - 0.05
            kp_label_config.append((f"{y_var_name}\n {y}", y_pos_label, "#4e79a7"))

            x_pos_label = pos.x0 + 0.5 * pos.width
            modif_label_config.append((f"{x_var_name}\n {x}", x_pos_label, "#e15759"))

    for text, y_pos, color in kp_label_config:
        fig.text(
            0.0606,
            y_pos - 0.05,
            text,
            transform=fig.transFigure,
            fontsize=26.0,
            color=color,
            weight="bold",
            fontname="Arial",
            rotation=90.0,
            ha="center",
        )

    for text, x_pos, color in modif_label_config:
        fig.text(
            x_pos,
            0.92,
            text,
            transform=fig.transFigure,
            fontsize=26.0,
            color=color,
            weight="bold",
            fontname="Arial",
            ha="center",
        )

    fig.text(
        0.8461,
        0.0220,
        "0.1x speed",
        transform=fig.transFigure,
        fontsize=26.0,
        color=color,
        fontname="Arial",
        ha="center",
    )
    return fig


# Draw individual frames
temp_dir.mkdir(exist_ok=True)
print(f"Saving frames to {temp_dir}")

pause_time = 0.5
global_counter = 0

for curr_video in range(num_trials):
    for frame_within_video in trange(num_frames, desc=f"Trial {curr_video + 1}"):
        fig = draw_frame(curr_video, frame_within_video)
        plt.savefig(temp_dir / f"{global_counter:05d}.jpg")
        plt.close(fig)
        global_counter += 1

    # add brief pause after each trial
    for i in range(int(pause_time * fps)):
        fig = draw_frame(curr_video, frame_within_video)
        plt.savefig(temp_dir / f"{global_counter:05d}.jpg")
        plt.close(fig)
        global_counter += 1

run(
    [
        "ffmpeg",
        "-r",
        str(fps),
        "-i",
        str(temp_dir / r"%05d.jpg"),
        "-vcodec",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(output_path),
    ]
)
# rmtree(temp_dir)
# print("removing {temp_dir}")