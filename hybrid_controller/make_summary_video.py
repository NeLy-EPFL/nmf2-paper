import matplotlib.pyplot as plt
import numpy as np
import imageio
from pathlib import Path
from tempfile import gettempdir
from shutil import rmtree
from tqdm import trange
from subprocess import run


# Configs
data_path = Path("data")
controller_paths = {"CPG": "CPG", "Rule-based": "Decentralized", "Hybrid": "hybrid"}
terrain_types = ["flat", "gapped", "blocks", "mixed"]
enable_adhesion = True
num_trials = 10
output_path = Path("outputs/controller_comparison.mp4")


# Index video files
video_paths = {}
for controller, tag in controller_paths.items():
    for terrain in terrain_types:
        matches = list(
            data_path.glob(f"{terrain}_{tag}pts_adhesion{enable_adhesion}*/*.mp4")
        )
        matches = sorted(matches)
        assert (
            len(matches) == num_trials
        ), f"Found {len(matches)} videos for {controller} on {terrain} terrain"
        video_paths[(controller, terrain)] = matches


# Load videos
fps = None
num_frames = None
all_videos = {}
for (controller, terrain), video_path in video_paths.items():
    print(f"Loading videos for {controller} on {terrain}")
    all_videos[(controller, terrain)] = []
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
        all_videos[(controller, terrain)].append(frames)


def draw_frame(curr_video, frame_within_video):
    fig, axs = plt.subplots(len(controller_paths), len(terrain_types), figsize=(20, 12))
    fig.subplots_adjust(
        left=0.1, right=0.95, top=0.9, bottom=0.05, hspace=0.03, wspace=0.03
    )

    for i, controller in enumerate(controller_paths):
        for j, terrain in enumerate(terrain_types):
            ax = axs[i, j]
            frame = all_videos[(controller, terrain)][curr_video][frame_within_video]
            ax.imshow(frame[30:, ::-1, :])
            ax.axis("off")
            ax.text(7, 60, f"Trial {curr_video + 1}", fontname="Helvetica", fontsize=20)

    controller_label_config = [
        ("CPG\ncontroller", 0.7216, "#4e79a7"),
        ("Rule-based\ncontroller", 0.4266, "#f28e2b"),
        ("Hybrid\ncontroller", 0.1494, "#7a653b"),
    ]
    for text, y_pos, color in controller_label_config:
        fig.text(
            0.0606,
            y_pos - 0.05,
            text,
            transform=fig.transFigure,
            fontsize=26.0,
            color=color,
            weight="bold",
            fontname="Helvetica",
            rotation=90.0,
            ha="center",
        )

    terrain_label_config = [
        ("Flat terrain", 0.2039, "#000000"),
        ("Gapped terrain", 0.4180, "#000000"),
        ("Blocks terrain", 0.6320, "#000000"),
        ("Mixed terrain", 0.8461, "#000000"),
    ]
    for text, x_pos, color in terrain_label_config:
        fig.text(
            x_pos,
            0.92,
            text,
            transform=fig.transFigure,
            fontsize=26.0,
            color=color,
            weight="bold",
            fontname="Helvetica",
            ha="center",
        )

    fig.text(
        0.8461,
        0.0220,
        "0.1x speed",
        transform=fig.transFigure,
        fontsize=26.0,
        color=color,
        fontname="Helvetica",
        ha="center",
    )

    return fig


# Draw individual frames
temp_dir = Path(gettempdir()) / "controller_comparison_video"
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