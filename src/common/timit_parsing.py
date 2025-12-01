import os
import regex
import librosa
import numpy as np

from src.common import config
from src.common import preprocessing as pre
from src.common import mfcc


def get_files(folder: str) -> list[str]:
    """Scan the directory and get the absolute paths to every single PHN, WAV pair"""
    file_names = set()
    files = sorted(os.listdir(folder))
    for i in files:
        matches = regex.findall(r"(.*)\.(?:PHN)", i.upper())

        if matches:
            file_names.add(matches[0])

    return sorted(file_names)


def extract_phonemes(dir: str, file: str) -> list[tuple[int, int, str]]:
    """Read and return list of Phonemes for each sample range"""
    raw_phonemes: str
    with open(os.path.join(dir, file + ".PHN")) as f:
        raw_phonemes = f.read()

    phonemes: list[tuple[int, int, str]] = []
    for line in raw_phonemes.split("\n"):
        if not line:
            continue
        start_sample, end_sample, phoneme = line.strip().split()
        phonemes.append((int(start_sample), int(end_sample), phoneme))
    return phonemes


def extract_mfccs(dir: str, file: str) -> list[np.ndarray]:
    """Read and return all MFCCs for given file after defining overlapping frames"""
    
    wav, _ = librosa.load(os.path.join(dir, file + ".WAV"), sr=config.SAMPLE_RATE)

    frames = pre.frames_from_wav(wav)
    windows = pre.to_hann_window(frames)
    log_mels = mfcc.to_log_mel(windows)
    mfccs = mfcc.to_mfcc(log_mels)
    
    return mfccs
