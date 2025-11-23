import numpy as np
import scipy.signal.windows as sp
import librosa

import config
    
def define_frames(prev :np.ndarray, curr: np.ndarray) -> list[np.ndarray]:
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


def to_hann_windows(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Applies the hann window on the given frames."""
    hann_window = sp.hann(config.FRAME_SIZE, sym=False)
    windows = []
    for i in range(0, len(frames)):
        windows.append(frames[i] * hann_window)

    return windows

def to_log_mel(windows: list[np.ndarray]) -> list[np.ndarray]:
    """Takes in a list of windows, applies the FFT, the Mel filterbank and then takes the log to produce the log-mel spectrographs"""
    log_mels = []
    for window in windows:
        magnitude_spectrum = abs(np.fft.rfft(window))       # calculating the Euclidean Norm to convert to magnitude-frequency spectrogram
        power_spectrum = abs(magnitude_spectrum ** 2)       # apparently power-frequency spectrograms are the shit

        mel_filterbank = librosa.filters.mel(sr=config.SAMPLE_RATE, n_fft = config.FRAME_SIZE)

        mel_energies = mel_filterbank @ power_spectrum
        log_mel = np.log(mel_energies + 1e-6)               # adding a negligible value to avoid log(0)
        log_mels.append(log_mel)

    return log_mels