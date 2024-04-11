import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Iterable
from matplotlib.lines import Line2D


def matplotlib_figure_to_array(fig: plt.Figure):
    fig.canvas.draw()
    return np.array(fig.canvas.buffer_rgba())[..., :3]


def merge_frame_and_time_series(
    fig: plt.Figure, lines: Iterable[Line2D], t: float, frame: np.ndarray
):
    for line in lines:
        line.set_xdata([t, t])

    time_series_img = matplotlib_figure_to_array(fig)
    assert frame.shape[0] >= time_series_img.shape[0]
    width = frame.shape[1]
    ts_height = int(width * time_series_img.shape[0] / time_series_img.shape[1])
    time_series_img = cv2.resize(time_series_img, (width, ts_height))
    merged_frame = np.concatenate([frame[..., :3], time_series_img[..., :3]], axis=0)
    return merged_frame
