from itertools import product
from pathlib import Path
from subprocess import run

import imageio
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

controllers = ["cpg", "rule_based", "hybrid"]
terrains = ["flat", "gapped", "blocks", "mixed"]
n_trials = 20
n_frames = 450

s_ = np.s_[30:]
s_yx = np.s_[25:]
video_dir = Path("outputs/videos")


def get_video_reader(trial_id, controller, terrain):
    path = video_dir / f"{controller}_{terrain}_{trial_id}.mp4"
    return imageio.get_reader(path)


def get_video_props():
    with get_video_reader(0, "cpg", "flat") as reader:
        metadata = reader.get_meta_data()
        return (*metadata["size"][::-1], metadata["fps"])


img_height, img_width, fps = get_video_props()
empty_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)


def init_figure(trial_id):
    fig, axs = plt.subplots(len(controllers), len(terrains), figsize=(19.2, 10.72))
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.93, bottom=0.045, hspace=0, wspace=0.03
    )

    images = np.empty((len(controllers), len(terrains)), object)

    for i in range(len(controllers)):
        for j in range(len(terrains)):
            ax = axs[i, j]
            images[i, j] = ax.imshow(empty_img[s_yx])

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])

            if i + j == 0:
                ax.text(
                    -0.21,
                    1,
                    f"Trial {trial_id + 1}",
                    fontsize=26,
                    fontname="Arial",
                    va="bottom",
                    transform=ax.transAxes,
                    color=(0.4,) * 3,
                )

    controller_label_config = [
        ("CPG\ncontroller", "#4e79a7"),
        ("Rule-based\ncontroller", "#f28e2b"),
        ("Hybrid\ncontroller", "#7a653b"),
    ]

    for i, (text, color) in enumerate(controller_label_config):
        axs[i, 0].set_ylabel(text, color=color, size=26, fontname="Arial", labelpad=10)

    terrain_labels = [
        "Flat terrain",
        "Gapped terrain",
        "Blocks terrain",
        "Mixed terrain",
    ]

    for j, text in enumerate(terrain_labels):
        axs[0, j].set_title(text, size=26, fontname="Arial")

    axs[-1, -1].text(
        1,
        -0.02,
        "0.1x speed",
        color="k",
        transform=axs[-1, -1].transAxes,
        fontsize=26,
        ha="right",
        va="top",
        fontname="Arial",
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
                    video_reader.set_image_index(frame_id)
                    img = video_reader.get_next_data()[s_]
                else:
                    img = empty_img
                image.set_data(img[s_yx])

            fig.canvas.draw()
            frame = np.array(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)

    for reader in video_readers:
        reader.close()


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
