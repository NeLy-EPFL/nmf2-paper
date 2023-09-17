import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import stable_baselines3 as sb3
import imageio
import torch_geometric as pyg
from pathlib import Path
from tqdm import trange

import flygym.util.vision as vision
from flygym.arena.mujoco_arena import OdorArena, FlatTerrain, GappedTerrain, BlocksTerrain, MixedTerrain
from flygym.util.data import color_cycle_rgb

from arena import ObstacleOdorArena
from rl_navigation import NMFNavigation
from vision_model import VisualFeaturePreprocessor


base_dir = Path("/home/sibwang/nmf2-paper/integrated_task/preprint_trial")


## Load vision model =====
vision_model_path = base_dir / "data/vision/visual_preprocessor.pt"
vision_model = VisualFeaturePreprocessor.load_from_checkpoint(vision_model_path).cpu()
ommatidia_graph_path = base_dir / "data/vision/ommatidia_graph.pkl"
with open(ommatidia_graph_path, "rb") as f:
    ommatidia_graph_nx = pickle.load(f)
ommatidia_graph = pyg.utils.from_networkx(ommatidia_graph_nx).cpu()

## Fix random seed =====
np.random.seed(0)
sb3.common.utils.set_random_seed(0, using_cuda=True)
torch.manual_seed(0)

def add_insets(
    viz_frame,
    visual_input,
    odor_intensities,
    odor_color,
    odor_gain=400,
    panel_height=150
):
    final_frame = np.zeros(
        (viz_frame.shape[0] + panel_height + 5, viz_frame.shape[1], 3), dtype=np.uint8
    )
    final_frame[: viz_frame.shape[0], :, :] = viz_frame

    img_l = vision.hex_pxls_to_human_readable(visual_input[0, :, :]).astype(np.uint8)
    img_r = vision.hex_pxls_to_human_readable(visual_input[1, :, :]).astype(np.uint8)
    vision_inset_size = np.array(
        [panel_height, panel_height * (img_l.shape[1] / img_l.shape[0])]
    ).astype(np.uint16)

    img_l = cv2.resize(img_l, vision_inset_size[::-1])
    img_r = cv2.resize(img_r, vision_inset_size[::-1])
    mask = cv2.resize(
        (vision.ommatidia_id_map > 0).astype(np.uint8), vision_inset_size[::-1]
    ).astype(bool)
    img_l[~mask] = 0
    img_r[~mask] = 0
    img_l = np.repeat(img_l[:, :, np.newaxis], 3, axis=2)
    img_r = np.repeat(img_r[:, :, np.newaxis], 3, axis=2)
    vision_inset = np.zeros(
        (panel_height, vision_inset_size[1] * 2 + 10, 3), dtype=np.uint8
    )
    vision_inset[:, : vision_inset_size[1], :] = img_l
    vision_inset[:, vision_inset_size[1] + 10 :, :] = img_r
    col_start = int((viz_frame.shape[1] - vision_inset.shape[1]) / 2)
    final_frame[
        -panel_height:, col_start : col_start + vision_inset.shape[1], :
    ] = vision_inset

    cv2.putText(
        final_frame,
        f"L",
        org=(col_start, viz_frame.shape[0] + 27),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.8,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    cv2.putText(
        final_frame,
        f"R",
        org=(col_start + vision_inset.shape[1] - 17, viz_frame.shape[0] + 27),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.8,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
        thickness=1,
    )

    # Odor info
    assert np.array(odor_intensities).shape == (1, 4)
    odor_intensities = np.average(
        np.array(odor_intensities).reshape(1, 2, 2), axis=1, weights=[9, 1],
    ).flatten()
    unit_size = panel_height // 5

    for i_side in range(2):
        row_start = unit_size * 2 + viz_frame.shape[0] + 5
        row_end = row_start + unit_size
        width = int(odor_intensities[i_side] * odor_gain)
        if i_side == 0:
            col_start = 0
            col_end = width
        else:
            col_start = final_frame.shape[1] - width
            col_end = final_frame.shape[1]
        final_frame[row_start:row_end, col_start:col_end, :] = odor_color

    return final_frame


def make_arena():
    terrain_arena = MixedTerrain(
        height_range=(0.3, 0.3), gap_width=0.2, ground_alpha=1
    )
    odor_arena = ObstacleOdorArena(
        terrain=terrain_arena,
        obstacle_positions=np.array([(7.5, 0)]),
        obstacle_radius=1,
        odor_source=np.array([[15, 0, 2]]),
        marker_size=0.5,
        obstacle_colors=(0, 0, 0, 1),
    )
    return odor_arena

def run_and_visualize(
    model_path, out_path, spawn_pos=[-0.02, 0, 0.2], spawn_orient=(0, 0, np.pi / 2)
):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    sim = NMFNavigation(
        arena_factory=make_arena,
        vision_model=vision_model,
        ommatidia_graph=ommatidia_graph,
        test_mode=True,
        debug_mode=True,
    )

    model = sb3.SAC.load(model_path)

    reward_hist = []
    action_hist = []
    obs, info = sim.reset(spawn_pos=spawn_pos, spawn_orient=spawn_orient)
    obs_hist = [obs]
    info_hist = [info]
    for i in trange(100):
        action, _ = model.predict(obs, deterministic=True)    
        obs, reward, terminated, truncated, info = sim.step(action)
        action_hist.append(action)
        obs_hist.append(obs)
        reward_hist.append(reward)
        info_hist.append(info)
        if info["fly_tgt_dist"] < 2.5:
            print("distance < 3, stopping")
            break
        if terminated:
            print("terminated")
            break
    with open(out_path / "info_hist.pkl", "wb") as f:
        pickle.dump(info_hist, f)

    obs_hist = np.array(obs_hist)
    reward_hist = np.array(reward_hist)
    action_hist = np.array(action_hist)

    with imageio.get_writer(
        out_path / "video.mp4", fps=sim.sim_params.render_fps
    ) as writer:
        for viz_frame, vision_input, odor_input in zip(
            sim.controller._frames, sim.vision_hist, sim.odor_hist
        ):
            frame = add_insets(
                viz_frame,
                vision_input,
                odor_input,
                odor_color=color_cycle_rgb[1],
                odor_gain=400,
                panel_height=150
            )
            writer.append_data(frame)
    # sim.controller.save_video(out_path / "video.mp4")

    # Save snapshot frames
    individual_frames_dir = out_path / "individual_frames"
    individual_frames_dir.mkdir(parents=True, exist_ok=True)
    snapshot_interval_frames = 30
    snapshots = np.array(
        [sim.controller._frames[i] for i in range(0, len(sim.controller._frames), snapshot_interval_frames)]
    )
    background = np.median(snapshots, axis=0)

    for i in trange(0, snapshots.shape[0]):
        img = snapshots[i, :, :, :]
        is_background = np.isclose(img, background, atol=1).all(axis=2)
        img_alpha = np.ones((img.shape[0], img.shape[1], 4)) * 255
        img_alpha[:, :, :3] = img
        img_alpha[is_background, 3] = 0
        img_alpha = img_alpha.astype(np.uint8)
        # break
        imageio.imwrite(
            individual_frames_dir / f"frame_{i}.png", img_alpha
        )

    imageio.imwrite(individual_frames_dir / "background.png", background.astype(np.uint8))


if __name__ == "__main__":
    spawn_positions = [
        (-1, -1, 0.2), (-1, 0, 0.2), (-1, 1, 0.2), 
        (0, -1, 0.2), (0, 0, 0.2), (0, 1, 0.2), 
        (1, -1, 0.2), (1, 0, 0.2), (1, 1, 0.2),     
    ]
    num_train_steps = 266000
    model_path = f"data/rl/rl_model.zip"
    for spawn_pos in spawn_positions:
        run_and_visualize(
            model_path,
            f"outputs/{num_train_steps}_{spawn_pos[0]}_{spawn_pos[1]}_{spawn_pos[2]}", 
            np.array(spawn_pos),
        )
