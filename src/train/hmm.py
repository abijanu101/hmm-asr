import os
import pathlib
import regex
import librosa

from src.common import config
from src.models import hmm

MODELS_PATH = os.path.join(
    pathlib.Path(__file__).parent.parent.parent, "models", "hmm.pk1"
)
TIMIT_PATH = os.path.join(
    pathlib.Path(__file__).parent.parent.parent, "resources", "TIMIT", "TRAIN"
)


def get_files(folder: str) -> list[str]:
    """Scan the directory and get the absolute paths to every single PHN, WAV pair"""
    file_names = set()
    files = os.listdir(folder)
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

    drs = sorted(os.listdir(TIMIT_PATH))
    speakers = sorted(os.listdir(os.path.join(TIMIT_PATH, drs[i])))
    files = sorted(os.listdir(os.path.join(TIMIT_PATH, drs[i], speakers[j])))

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


def train(models: dict[str, hmm.hmm.GMMHMM], dir: str, file: str) -> None:
    """Train the HMM+GMM models on the file specified"""

    print(os.path.join(dir, file + '.WAV'))
    wav = librosa.load(os.path.join(dir, file + '.WAV'), sr=config.SAMPLE_RATE)
    phones: str
    with open(os.path.join(dir,file + '.PHN')) as f:
        phones = f.read()


        
    print(wav, phones)

def main() -> None:
    """Train the HMM+GMM Models"""

    # loading
    models = {}
    last_file_trained = {}

    if not os.path.exists(MODELS_PATH):
        confirmation = input( f"{MODELS_PATH} doesn't exist, do you want to create a new set of HMM models? [Y/n]: ")

        if confirmation != "Y":
            print("Okay, I can't really do anything then")
            return

        models = hmm.create()
        hmm.persist(models, {}, MODELS_PATH)
    else:
        models, last_file_trained = hmm.load(MODELS_PATH)

    # training
    i_old, j_old, k_old = get_next_indices(last_file_trained)
    i,j,k = 0,0,0

    drs = sorted(os.listdir(os.path.join(TIMIT_PATH)))
    try:
        for i in range(i_old, len(drs)):
            speakers = sorted(os.listdir(os.path.join(TIMIT_PATH, drs[i])))

            for j in range(j_old, len(speakers)):
                files = get_files(os.path.join(TIMIT_PATH, drs[i], speakers[j]))

                for k in range(k_old, len(files)):
                    train(models, os.path.join(TIMIT_PATH, drs[i], speakers[j]), files[k])
                    
                k_old = 0
            j_old = 0
            
    except KeyboardInterrupt:
        last_file_trained = {"DR-IND": i, "SPKR-IND": j, "FILE-IND": k}
        print("Trained until: ", last_file_trained)
        hmm.persist(models, last_file_trained, MODELS_PATH)


if __name__ == "__main__":
    main()
