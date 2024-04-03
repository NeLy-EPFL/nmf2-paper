import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange

from flygym import SingleFlySimulation
from flygym.arena.tethered import Tethered
from nmf_grooming import all_groom_dofs, GroomingCamera, GroomingFly

import cv2

timestep = 1e-4


def get_data(xml_variant):
    data_path = Path(f"/Users/stimpfli/Desktop/nmf2-paper/grooming/data/220807_aJO-Gal4xUAS-CsChr_fly002_beh002/pose-3d/{xml_variant}")
    with open(data_path / "leg_joint_angles.pkl", "rb") as f:
        leg_data = pickle.load(f)

    with open(data_path / "head_joint_angles.pkl", "rb") as f:
        head_data = pickle.load(f)

    # merge the dict (have different keys)
    data_raw = {**leg_data, **head_data}

    data_raw["meta"] = {"timestep":1e-2}
    # reformat the data
    joint_corresp = {"ThC":"Coxa", "CTr":"Femur", "FTi":"Tibia", "TiTa":"Tarsus1"}

    data = {}
    for key, value in data_raw.items():
        if key == "meta":
            data[key] = value
            continue
        elif "head" in key:
            _, joint, dof = key.split("_")
            side = ""
            joint = "Head"
        elif "antenna" in key:
            _, joint, dof, side = key.split("_")
            joint = "Pedicel"
        else:
            _, side, joint, dof = key.split("_")
            joint = joint_corresp[joint]
        new_key = f"joint_{side}{joint}"
        if not dof == "pitch":
            new_key += f"_{dof}"
        
        data[new_key] = value

    target_num_steps = int(len(data["joint_LFCoxa"])/timestep*data["meta"]["timestep"])
    data_block = np.zeros((len(all_groom_dofs), target_num_steps))
    input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
    output_t = np.arange(target_num_steps) * timestep
    for i, joint in enumerate(all_groom_dofs):
        if "RPedicel_yaw" in joint:
                # data[joint] = np.ones_like(data[joint])*np.mean(data[joint])*-1
                data[joint] *= -1
        data_block[i, :] = np.unwrap(np.interp(output_t, input_t, data[joint]))

    # swap head roll and head yaw
    hroll_id = all_groom_dofs.index("joint_Head_roll")
    hyaw_id = all_groom_dofs.index("joint_Head_yaw")
    hroll = data_block[hroll_id, :].copy()
    data_block[hroll_id, :] = data_block[hyaw_id, :]

    return data_block

def get_replay_frames(data_block_focus, video_fps):

    fly = GroomingFly(xml_variant=xml_variant)
    cam = GroomingCamera(fly, camera_id="Animat/camera_front")
    cam._eff_render_interval = 1 / video_fps
    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        arena=Tethered(),
    )

    timestamps = []
    _ = sim.reset()

    for i in range(50):
        sim.step({"joints": data_block_focus[:, 0]})

    sim.curr_time = 0.0
    for i in trange(target_num_steps):
        # here, we simply use the recorded joint angles as the target joint angles
        joint_pos = data_block_focus[:, i]
        action = {"joints": joint_pos}
        _ = sim.step(action)
        frame = sim.render()
        if frame:
            timestamps.append(sim.curr_time)

    return cam._frames, timestamps

def get_original_frames(start_t, end_t, video_fps):
    video = Path("/Users/stimpfli/Desktop/nmf2-paper/grooming/data/220807_aJO-Gal4xUAS-CsChr_fly002_beh002/videos/camera_3.mp4")
    # Extract frames between start and end time

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
    num_frames = int((end_t - start_t) * video_fps)

    base_video_frames = []
    timestamp = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        base_video_frames.append(frame)
        timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC)*1000)

    cap.release()
    return base_video_frames, timestamp

def write_side_by_side_video(order, output_path, replay_frames, replay_timestamps, video_fps):
    assert len(replay_frames[order[0]]) == len(replay_frames[order[1]]), f"Length mismatch between {order[0]} and {order[1]} ({replay_timestamps[order[0]]} vs {replay_timestamps[order[1]]}"
    assert len(replay_frames[order[0]]) == len(replay_frames[order[2]]), f"Length mismatch between {order[0]} and {order[2]} ({replay_timestamps[order[0]]} vs {replay_timestamps[order[2]]}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # write a video with the three frames side by side
    out_size = [0, 0]
    for key in order:
        frame_shape = replay_frames[key][0].shape
        out_size[0] += frame_shape[1]
    out_size[1] = replay_frames["real_fly"][0].shape[0] # assume all frames have the same height
    
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (out_size[0], out_size[1]))
    for i in range(len(replay_frames[order[0]])):

        frame_frames = []
        for key in order:
            if order == "real_fly":
                frame_frames.append(replay_frames[key][i])
            else:
                frame_frames.append(replay_frames[key][i][:, :, ::-1])
        frame = cv2.hconcat(frame_frames)
        out.write(frame)
    
    out.release()


# run the main
if __name__ == "__main__":
    replay_frames = {}
    replay_timestamps = {}
    for xml_variant in ["deepfly3d_original", "deepfly3d"]:
        data_block = get_data(xml_variant)
        video_fps = 100

        start_t = 3.5
        end_t = 4.0#6
        start_step = int(start_t / timestep)
        end_step = int(end_t / timestep)
        target_num_steps = end_step - start_step
        data_block_focus = data_block[:, start_step:end_step]

        replay_frames[xml_variant], replay_timestamps[xml_variant] = get_replay_frames(data_block_focus, video_fps)

    replay_frames["real_fly"], replay_timestamps["real_fly"] = get_original_frames(start_t, end_t, video_fps)

    write_side_by_side_video(["deepfly3d_original", "real_fly", "deepfly3d"],
                                Path(f"/Users/stimpfli/Desktop/nmf2-paper/grooming/outputs/final_sidebyside.mp4"),
                                replay_frames, replay_timestamps, video_fps)

