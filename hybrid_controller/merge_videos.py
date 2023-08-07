from pathlib import Path
from glob import glob

import cv2
import numpy as np

from tqdm import tqdm

# This script aims at reading and merging all the videos from a given folder

# Path to the folder containing the datapts
path = Path("Data_points")
all_videos_folders = path.glob("*pts*")

for video_folder in all_videos_folders:
    print(f"Processing {video_folder}")
    assert video_folder.is_dir(), "The path provided is not a folder"
    # Get the name of the video folder
    concat_video_name = video_folder.name + "_concat.mp4"
    all_videos = list(video_folder.glob("**/*.mp4"))

    concat_video_path = video_folder / concat_video_name
    if concat_video_path.is_file():
        print(f"Video {concat_video_path} already exists")
        continue

    if len(all_videos) > 1:
        # Open a video to get its specs
        first_video = str(all_videos[0])
        video = cv2.VideoCapture(first_video)
        # Get the specs of the video
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        video.release()

        # Build the video output with cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(concat_video_path), fourcc, fps, (width, height))
        for video in tqdm(all_videos):
            # open all of them and merge them into a single video
            video = cv2.VideoCapture(str(video))
            while video.isOpened():
                ret, frame = video.read()
                if ret:
                    out.write(frame)
                else:
                    break
            video.release()
        out.release()
        print(f"Video {concat_video_path} has been saved")
    else:
        print(f"No video found in {video_folder}")
