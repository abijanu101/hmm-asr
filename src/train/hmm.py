import os
import numpy as np

from src.common import config
from src.common import timit_parsing as tp
from src.common import hmm


def parse_utterance(dir: str, file: str) -> dict[str, list[list[np.ndarray]]]:
    """Given a valid Utterance ID and Directory, return a mapping from phoneme to MFCC sequence list for the utterance"""

    mfccs = tp.extract_mfccs(dir, file)
    phonemes = tp.extract_phonemes(dir, file)

    grouped: dict[str, list[list[np.ndarray]]] = {i: [] for i in config.PHONEMES}
    for start_sample, end_sample, phoneme in phonemes:
        range_start = round(start_sample / config.FRAME_HOP)
        range_end = round(end_sample / config.FRAME_HOP)

        if range_start >= len(mfccs) or not mfccs[range_start:range_end]:
            continue

        grouped[phoneme].append(mfccs[range_start:range_end])

    return grouped


def fit_models(
    models: dict[str, hmm.hmm.GMMHMM], phonemes: dict[str, list[list[np.ndarray]]]
):
    """Fit all the models for acceptably long clusters of phonemes"""
    for phoneme, sequences in phonemes.items():
        flattened: list[np.ndarray] = [frame for seq in sequences for frame in seq]
        if not flattened:
            continue
        stacked = np.vstack(flattened)
        print("Currently Running EM Algorithm for: ", phoneme)
        models[phoneme].fit(stacked, [len(seq) for seq in sequences])

    print("Updated Model")


def merge_phonemes(
    phonemes: dict[str, list[list[np.ndarray]]],
    curr_phonemes: dict[str, list[list[np.ndarray]]],
) -> dict[str, list[list[np.ndarray]]]:
    for k, v in curr_phonemes.items():
        phonemes[k].extend(v)
    return phonemes


def main() -> None:
    """Train the HMM+GMM Models"""

    models = {}

    # loading
    if not os.path.exists(config.HMM_MODEL_PATH):
        confirmation = input(
            f"{config.HMM_MODEL_PATH} doesn't exist, do you want to create a new set of HMM models? [Y/n]: "
        )

        if confirmation != "Y":
            return print("Okay, I can't really do anything then")

        models = hmm.create()
        hmm.persist(models)
    else:
        models = hmm.load(config.HMM_MODEL_PATH)

    # training
    drs = sorted(
        dr
        for dr in os.listdir(config.TIMIT_TRAIN)
        if os.path.isdir(os.path.join(config.TIMIT_TRAIN, dr))
    )

    phonemes = {phn: [] for phn in config.PHONEMES}

    for dr in drs:
        print("Scanning through", dr)
        DR_PATH = os.path.join(config.TIMIT_TRAIN, dr)
        speakers = sorted(
            spkr
            for spkr in os.listdir(DR_PATH)
            if os.path.isdir(os.path.join(DR_PATH, spkr))
        )

        for spkr in speakers:
            SPKR_PATH = os.path.join(DR_PATH, spkr)
            files = tp.get_files(SPKR_PATH)

            for rec in files:
                curr_phonemes = parse_utterance(SPKR_PATH, rec)
                phonemes = merge_phonemes(phonemes, curr_phonemes)

            print("Completed Scanning through Speaker", spkr)

    fit_models(models, phonemes)
    hmm.persist(models)


if __name__ == "__main__":
    main()
