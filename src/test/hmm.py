import os
import numpy as np

from src.common import config
from src.common import timit_parsing as tp
from src.common import hmm


in_top_1_phonesF = 0
in_top_2_phonesF = 0
in_top_3_phonesF = 0
in_top_5_phonesF = 0

in_top_1_phones = 0
in_top_2_phones = 0
in_top_3_phones = 0
in_top_5_phones = 0

total_phones = 0
total_frames = 0

models = {}


def test_file(dir: str, file: str) -> None:
    """Increments correct_phones and incorrect_phones based on predictions from the HMM"""

    phonemes = tp.extract_phonemes(dir, file)
    mfccs = tp.extract_mfccs(dir, file)

    for start_sample, end_sample, phoneme in phonemes:
        start_range = round(start_sample / config.FRAME_HOP)
        end_range = round(end_sample / config.FRAME_HOP)



        answers = []
        # get answer
        X = mfccs[start_range:end_range]
        if len(X) < config.N_STATES:
            continue
        
        for PHONEME in config.PHONEMES:
            answers.append((PHONEME, models[PHONEME].score(X)))

        # evaluate
        global total_phones, in_top_5_phones, in_top_3_phones, in_top_2_phones, in_top_1_phones
        global total_frames, in_top_5_phonesF, in_top_3_phonesF, in_top_2_phonesF, in_top_1_phonesF
        answers = sorted(answers, key=lambda x: x[1], reverse=True)

        total_phones += 1
        total_frames += len(X)
        if phoneme == answers[0][0]:
            in_top_1_phones += 1
            in_top_2_phones += 1
            in_top_3_phones += 1
            in_top_5_phones += 1

            in_top_1_phonesF += len(X)
            in_top_2_phonesF += len(X)
            in_top_3_phonesF += len(X)
            in_top_5_phonesF += len(X)
        elif phoneme == answers[1][0]:
            in_top_2_phones += 1
            in_top_3_phones += 1
            in_top_5_phones += 1

            in_top_2_phonesF += len(X)
            in_top_3_phonesF += len(X)
            in_top_5_phonesF += len(X)
        elif phoneme == answers[2][0]:
            in_top_3_phones += 1
            in_top_5_phones += 1

            in_top_3_phonesF += len(X)
            in_top_5_phonesF += len(X)
        elif phoneme == answers[3][0] or phoneme == answers[4][0]:
            in_top_5_phones += 1
            in_top_5_phonesF += len(X)


def main() -> None:
    """Test the HMM+GMM Models"""

    # loading
    if not os.path.exists(config.HMM_MODEL_PATH):
        print(f"{config.HMM_MODEL_PATH} doesn't exist...")
        return
    global models
    models = hmm.load(config.HMM_MODEL_PATH)

    # testing
    drs = sorted(
        dr
        for dr in os.listdir(config.TIMIT_TEST)
        if os.path.isdir(os.path.join(config.TIMIT_TEST, dr))
    )

    try:
        for dr in drs:
            print("Scanning through", dr)
            DR_PATH = os.path.join(config.TIMIT_TEST, dr)
            speakers = sorted(
                spkr
                for spkr in os.listdir(DR_PATH)
                if os.path.isdir(os.path.join(DR_PATH, spkr))
            )

            for spkr in speakers:
                SPKR_PATH = os.path.join(DR_PATH, spkr)
                files = tp.get_files(SPKR_PATH)

                for rec in files:
                    test_file(SPKR_PATH, rec)

                print("\nPhones:\t", total_phones, '\t', 
                    in_top_5_phones / total_phones, '\t',
                    in_top_3_phones / total_phones, '\t',
                    in_top_2_phones / total_phones, '\t',
                    in_top_1_phones / total_phones
                )
                print("Frames:\t", total_frames, '\t', 
                    in_top_5_phonesF / total_frames, '\t',
                    in_top_3_phonesF / total_frames, '\t',
                    in_top_2_phonesF / total_frames, '\t',
                    in_top_1_phonesF / total_frames
                )

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
