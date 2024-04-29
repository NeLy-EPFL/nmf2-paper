import cv2
import numpy as np
from pathlib import Path
import imageio


base_path = Path("data/slope_front")
slope_degrees = np.unique(
    [int(i.stem.split("_")[2]) for i in base_path.glob("CPG_gravity_*.mp4")]
)
ignore_after = {
    False: {i: 3.5 for i in range(30, 190, 10)},
    True: {i: 3.5 for i in range(140, 190, 10)},
}
ignore_after[False][30] = 4
# ignore_after[True][140] = 4

fps = None
# gravity change time - stab duratin + initial duration dropped in rendering
warm_up_period = 0.4 - 0.2 + 0.05
frame_shape = None
playspeed = 0.1
pause_time = 0.5

frames_all = []

for deg in slope_degrees:
    # Read video
    frames_by_adhesion = {True: [], False: []}

    for adhesion in [True, False]:
        path = base_path / f"CPG_gravity_{deg}_y_adhesion{adhesion}.mp4"
        assert path.is_file()
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), f"Error opening video {path}"
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            assert fps == cap.get(
                cv2.CAP_PROP_FPS
            ), "fps is not the same for all videos"
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Finished reading {path}")
                break
            frame = np.array(frame[35:, ::-1, ::-1])
            frame_count += 1
            if frame_count / fps * playspeed < warm_up_period:
                continue
            if (
                deg in ignore_after[adhesion]
                and frame_count / fps > ignore_after[adhesion][deg]
            ):
                frames_by_adhesion[adhesion].append(np.zeros_like(frame))
                continue
            if frame_count >= 200 + warm_up_period * fps:
                print(f"Finished reading {path}")
                break

            # add labels
            cv2.putText(
                frame,
                "With adhesion" if adhesion else "Without adhesion",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(0, 0, 0),
                lineType=cv2.LINE_AA,
                thickness=1,
            )
            cv2.putText(
                frame,
                f"{playspeed}x",
                org=(560, 40),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(0, 0, 0),
                lineType=cv2.LINE_AA,
                thickness=1,
            )

            frames_by_adhesion[adhesion].append(frame)
        cap.release()

    for on_frame, off_frame in zip(
        frames_by_adhesion[True][:90], frames_by_adhesion[False][:90]
    ):
        frame_merged = np.zeros(
            (on_frame.shape[0] + 50, on_frame.shape[1] * 2, 3), dtype=np.uint8
        )
        if off_frame is not None:
            frame_merged[50:, : on_frame.shape[1]] = off_frame
        frame_merged[50:, on_frame.shape[1] :] = on_frame
        frame_merged = cv2.putText(
            frame_merged,
            f"{deg} degrees",
            org=(550, 30),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
            thickness=1,
        )
        frames_all.append(frame_merged)

    for i in range(int(pause_time * fps)):
        frames_all.append(frame_merged)

# Write video
Path("outputs").mkdir(exist_ok=True)
writer = imageio.get_writer("outputs/climbing.mp4", fps=fps)
for frame in frames_all:
    writer.append_data(frame)

writer.close()
print("Finished writing merged video")
