import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

base_path = Path("Data_points")

controllers = ["CPG", "hybrid"]
terrains = ["flat"]

for c in controllers:
    for t in terrains:
        path = base_path / f"{t}_{c}pts_adhesionTrue_kp30.0"
        if not path.is_dir():
            print(f"Path {path} does not exist")
            continue
        all_pkl = list(path.glob("*.pkl"))
        assert len(all_pkl) > 1, f"Path {path} does not contain any pkl file"
        data_pts = []
        for pkl_file in all_pkl:
            with open(pkl_file, "rb") as f:
                obs_list = pickle.load(f)
            data_pts.append(obs_list[-1]["fly"][0][0] - obs_list[0]["fly"][0][0])
            print(pkl_file.name, data_pts[-1])

        n_files = len(all_pkl)
        print(f"Controller {c} terrain {t} mean {np.mean(data_pts[-n_files:])}"
              f" std {np.std(data_pts[-n_files:])}")