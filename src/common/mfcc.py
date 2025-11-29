import numpy as np
from scipy.signal.windows import hann
from scipy.fftpack import dct
import librosa

from src.common import config

HANN_WINDOW = hann(config.FRAME_SIZE, sym=False)
MEL_FILTERBANK = librosa.filters.mel(sr=config.SAMPLE_RATE, n_fft = config.FRAME_SIZE)
    
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


def to_hann_window(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Applies the hann window on the given frames."""
    windows = []
    for i in range(0, len(frames)):
        windows.append(frames[i] * HANN_WINDOW)

    return windows


def to_log_mel(windows: list[np.ndarray]) -> list[np.ndarray]:
    """Takes in a list of windows and returns Corresponding Log-Mel spectrograms"""
    log_mels = []
    
    for window in windows:
        magnitude_spectrum = abs(np.fft.rfft(window))       # calculating the Euclidean Norm to convert to magnitude-frequency spectrogram
        power_spectrum = magnitude_spectrum ** 2            # apparently power-frequency spectrograms are the shit

        mel_energies = MEL_FILTERBANK @ power_spectrum
        log_mel = np.log(mel_energies + 1e-6)               # adding a negligible value to avoid log(0)

        log_mels.append(log_mel)
        
    return log_mels


def to_mfcc(log_mels: list[np.ndarray]) -> list[np.ndarray]:
    """Takes in Log-Mel spectrograms and returns the Mel Frequency Cepstral Coefficients"""
    mfccs = []

    for log_mel in log_mels:
        mfcc = dct(log_mel, type=2, norm='ortho')[:config.N_MFCC]
        mfccs.append(mfcc)
    
    return mfccs
