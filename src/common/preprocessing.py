import numpy as np
from scipy.signal.windows import hann

from src.common import config


def frames_from_wav(wav: np.ndarray) -> list[np.ndarray]:
    """Defines overlapping frames for provided audio, see documentation for more details"""

    frames = []
    i = 0
    while i + config.FRAME_SIZE <= len(wav):
        range = wav[i : i + config.FRAME_SIZE]
        frames.append(range)
        i += config.FRAME_HOP

    return frames


def frames_from_stream(prev :np.ndarray, curr: np.ndarray) -> list[np.ndarray]:
    '''This will define all the overlapping frames between prev and curr read buffers as well as the entirety of the current read'''
    curr = curr.flatten()
    prev = prev.flatten()

    frames = []
    drift = config.FRAME_HOP

    while drift <= config.FRAME_SIZE:
        frame = np.concatenate([prev[drift:config.FRAME_SIZE], curr[0:drift]])
        frames.append(frame)
        drift += config.FRAME_HOP

    return frames


HANN_WINDOW = hann(config.FRAME_SIZE, sym=False)

def to_hann_window(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Applies the hann window on the given frames."""
    windows = []
    for i in range(0, len(frames)):
        windows.append(frames[i] * HANN_WINDOW)

    return windows
