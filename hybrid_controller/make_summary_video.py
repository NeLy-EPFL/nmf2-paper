from itertools import product
from pathlib import Path
from subprocess import run

import imageio
import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader
from joblib import Parallel, delayed

controllers = ["cpg", "rule_based", "hybrid"]
terrains = ["flat", "gapped", "blocks", "mixed"]
n_trials = 20
n_frames = 450

s_ = np.s_[30:, ::-1]
video_dir = Path("outputs/videos")


def get_video_reader(trial_id, controller, terrain):
    path = video_dir / f"{controller}_{terrain}_{trial_id}.mp4"
    return VideoReader(path.as_posix())


def get_video_props():
    video_reader = get_video_reader(0, "cpg", "flat")
    return (*video_reader[0].asnumpy()[s_].shape[:2], video_reader.get_avg_fps())


img_height, img_width, fps = get_video_props()
empty_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)


def init_figure(trial_id):
    fig, axs = plt.subplots(len(controllers), len(terrains), figsize=(20, 12))
    fig.subplots_adjust(
        left=0.1, right=0.95, top=0.9, bottom=0.05, hspace=0.03, wspace=0.03
    )

    images = np.empty((len(controllers), len(terrains)), object)

    for i in range(len(controllers)):
        for j in range(len(terrains)):
            ax = axs[i, j]
            images[i, j] = ax.imshow(empty_img)
            ax.axis("off")
            ax.text(7, 60, f"Trial {trial_id + 1}", fontname="Arial", fontsize=20)

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
            fontsize=26,
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
    return fig, images.ravel()


output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


def write_trial_video(trial_id, n_frames=450):
    it = list(product(controllers, terrains))
    video_readers = [get_video_reader(trial_id, c, t) for c, t in it]
    fig, images = init_figure(trial_id)

    with imageio.get_writer(output_dir / f"{trial_id:02d}.mp4", fps=fps) as writer:
        for frame_id in range(n_frames):
            if frame_id < 5:
                continue
            for image, video_reader in zip(images, video_readers):
                if frame_id < len(video_reader):
                    img = video_reader[frame_id].asnumpy()[s_]
                else:
                    img = empty_img
                image.set_data(img)

            fig.canvas.draw()
            frame = np.array(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)


Parallel(n_jobs=-1)(
    delayed(write_trial_video)(trial_id) for trial_id in range(n_trials)
)
video_paths = [output_dir / f"{trial_id:02d}.mp4" for trial_id in range(n_trials)]
video_list = output_dir / "video_list.txt"
with open(video_list, "w") as f:
    for path in video_paths:
        f.write(f"file {path.relative_to(output_dir)}\n")

run(
    [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "outputs/video_list.txt",
        "-c",
        "copy",
        "outputs/controller_comparison.mp4",
        "-y",
    ]
)

video_list.unlink()
for path in video_paths:
    path.unlink()
