import cv2

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
w = None
h = None
fps = None
num_train_steps = 500000
out_path = "outputs/navigation_task_merged.mp4"


if __name__ == "__main__":
    # Merge video
    all_frames = []
    for spawn_pos in spawn_positions:
        in_path = f"outputs/{num_train_steps}_{spawn_pos[0]}_{spawn_pos[1]}_{spawn_pos[2]}/video.mp4"
        print(f"Reading {in_path}...")
        cap = cv2.VideoCapture(in_path)
        if w is None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        else:
            assert w == int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            assert h == int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    import imageio

    with imageio.get_writer(out_path, fps=fps) as writer:
        for frame in all_frames:
            writer.append_data(frame[..., ::-1])
