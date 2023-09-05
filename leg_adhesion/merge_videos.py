import cv2
from pathlib import Path


base_path = Path("data/slope_front")
slope_degrees = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
fps = None
warm_up_period = 3
frame_shape = None

frames_all = []

for deg in slope_degrees:
    path = base_path / f"CPG_gravity_{deg}_y.mp4"
    assert path.is_file()

    # Read video
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"Error opening video {path}"
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        assert fps == cap.get(cv2.CAP_PROP_FPS), "fps is not the same for all videos"
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished reading {path}")
            break
        frame_count += 1
        if frame_count / fps < warm_up_period:
            continue

        # add labels
        cv2.putText(
            frame,
            f"{deg} degrees",
            org=(20, 70),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(0, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=1,
        )
        cv2.putText(
            frame,
            f"0.1x speed",
            org=(540, 465),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            color=(0, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=1,
        )

        frames_all.append(frame)
        frame_shape = frame.shape
    cap.release()

# Write video
out = cv2.VideoWriter(
    str(base_path / "climbing.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_shape[1], frame_shape[0]),
)
for frame in frames_all:
    out.write(frame)
out.release()
print("Finished writing merged video")
