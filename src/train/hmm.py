import os
import regex
import librosa
import numpy as np

from src.common import config
from src.common import mfcc
from src.models import hmm


def define_frames(wav: np.ndarray) -> list[np.ndarray]:
    """Defines overlapping frames for provided audio, see documentation for more details"""

    frames = []
    i = 0
    while i + config.FRAME_SIZE <= len(wav):
        frames.append(wav[i : i + config.FRAME_SIZE])
        i += config.FRAME_HOP

    return frames


def extract_data(dir: str, file: str) -> dict[str, list[list[np.ndarray]]]:
    """Return a mapping from phoneme to its corresponding MFCC sequences"""

    print(os.path.join(dir, file))

    # get MFCCs
    wav, _ = librosa.load(os.path.join(dir, file + ".WAV"), sr=config.SAMPLE_RATE)
    mfccs = mfcc.to_mfcc(mfcc.to_log_mel(mfcc.to_hann_window(define_frames(wav))))

    # get Phonemes
    raw_phonemes: str
    with open(os.path.join(dir, file + ".PHN")) as f:
        raw_phonemes = f.read()

    phonemes: list[tuple[int, int, str]] = []
    for line in raw_phonemes.split("\n"):
        if not line:
            continue
        start_sample, end_sample, phoneme = line.strip().split()
        phonemes.append((int(start_sample), int(end_sample), phoneme))

    # group MFCCs by Phoneme
    
    grouped: dict[str, list[list[np.ndarray]]] = {i: [] for i in config.PHONEMES}
    for start_sample, end_sample, phoneme in phonemes:
        range_start = round(start_sample / config.FRAME_HOP)
        range_end = round(end_sample / config.FRAME_HOP)
        
        if range_start >= len(mfccs):
            continue

        grouped[phoneme].append(mfccs[range_start: range_end])
        
    return grouped


def merge_phonemes(
    phonemes: dict[str, list[list[np.ndarray]]], curr_phonemes: dict[str, list[list[np.ndarray]]]
) -> dict[str, list[list[np.ndarray]]]:
    for k, v in curr_phonemes.items():
        phonemes[k].extend(v)
    return phonemes


def fit_models(
    models: dict[str, hmm.hmm.GMMHMM], phonemes: dict[str, list[list[np.ndarray]]]
):
    """Fit all the models for acceptably long clusters of phonemes"""
    for phoneme, sequences in phonemes.items():
        flattened : list[np.ndarray] = [frame for seq in sequences for frame in seq]
        if not flattened: continue
        stacked = np.vstack(flattened)
        models[phoneme].fit(stacked, [len(seq) for seq in sequences])

    print("Updated Model")


def get_files(folder: str) -> list[str]:
    """Scan the directory and get the absolute paths to every single PHN, WAV pair"""
    file_names = set()
    files = sorted(os.listdir(folder))
    for i in files:
        matches = regex.findall(r"(.*)\.(?:PHN)", i.upper())

        if matches:
            file_names.add(matches[0])

    return sorted(file_names)


def get_next_indices(last_file_trained: dict[str, int]) -> tuple[int, int, int]:
    """Given the last file from the previous session, it tells you where to start"""

    i = last_file_trained.get("DR-IND", 0)
    j = last_file_trained.get("SPKR-IND", 0)
    k = last_file_trained.get("FILE-IND", -1)

    drs = sorted(
        dr
        for dr in os.listdir(config.TIMIT_TRAIN)
        if os.path.isdir(os.path.join(config.TIMIT_TRAIN, dr))
    )
    DR_PATH = os.path.join(config.TIMIT_TRAIN, drs[i])

    speakers = sorted(
        spkr
        for spkr in os.listdir(DR_PATH)
        if os.path.isdir(os.path.join(DR_PATH, spkr))
    )
    SPKR_PATH = os.path.join(DR_PATH, speakers[j])

    files = get_files(SPKR_PATH)

    k += 1
    if k >= len(files):
        k = 0
        j += 1
        if j >= len(speakers):
            j = 0
            i += 1
            if i >= len(drs):
                i = 0

    return i, j, k


def main() -> None:
    """Train the HMM+GMM Models"""

    models = {}
    last_file_trained = {}

    # loading
    if not os.path.exists(config.MODELS_PATH):
        confirmation = input(
            f"{config.MODELS_PATH} doesn't exist, do you want to create a new set of HMM models? [Y/n]: "
        )

        if confirmation != "Y":
            print("Okay, I can't really do anything then")
            return

        models = hmm.create()
        hmm.persist(models, {}, config.MODELS_PATH)
    else:
        models, last_file_trained = hmm.load(config.MODELS_PATH)

    # training
    i_old, j_old, k_old = get_next_indices(last_file_trained)
    i, j, k = 0, 0, 0

    drs = sorted(
        dr
        for dr in os.listdir(config.TIMIT_TRAIN)
        if os.path.isdir(os.path.join(config.TIMIT_TRAIN, dr))
    )
    try:
        for i in range(i_old, len(drs)):
            DR_PATH = os.path.join(config.TIMIT_TRAIN, drs[i])
            speakers = sorted(
                spkr
                for spkr in os.listdir(DR_PATH)
                if os.path.isdir(os.path.join(DR_PATH, spkr))
            )

            for j in range(j_old, len(speakers)):
                SPKR_PATH = os.path.join(DR_PATH, speakers[j])
                files = get_files(SPKR_PATH)

                phonemes = {phn: [] for phn in config.PHONEMES}
                for k in range(k_old, len(files)):
                    curr_phonemes = extract_data(SPKR_PATH, files[k])
                    phonemes = merge_phonemes(phonemes, curr_phonemes)

                fit_models(models, phonemes)
                k_old = 0
            j_old = 0
    except KeyboardInterrupt:
        last_file_trained = {"DR-IND": i, "SPKR-IND": j, "FILE-IND": k}
        print("Trained until: ", last_file_trained)
        hmm.persist(models, last_file_trained, config.MODELS_PATH)


if __name__ == "__main__":
    main()
