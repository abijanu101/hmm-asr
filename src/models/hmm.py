import os
import joblib

from src.common import config
from hmmlearn import hmm


def create() -> dict[str, hmm.GaussianHMM]:
    """Create HMM+GMM Models"""

    models = {}
    for phn in config.PHONEMES:
        models[phn] = hmm.GMMHMM(
            n_components=3,  # start, middle, end
            n_mix=config.N_GAUSSIANS,
            algorithm="viterbi",
            n_iter=config.N_EM_ITER,  # MAX EM Iterations
        )

    print(f"A total of {len(config.PHONEMES)} new HMM+GMM models created.")
    return models


def persist(
    models: dict[str, hmm.GaussianHMM], last_file_trained: dict[str, int], path: str
) -> None:
    """Persist HMM Models along with information about the last file successfully trained on in the dataset"""

    print("Saving...")
    data = {"models": models, "last_file_trained": last_file_trained}
    joblib.dump(data, path)

    print(f"Saved models as file '{path}' successfully.")


def load(path: str) -> tuple[dict[str, hmm.GaussianHMM], dict[str, int]]:
    """Load HMM Models along with the information about the last file successfully trained on in the dataset"""

    if not os.path.exists(path):
        raise RuntimeError(f"{path} does not exist")

    print("Loading HMM+GMM Models...")
    data = joblib.load(path)
    last_file_trained: dict[str, int] = data.get("last_file_trained", {})

    models = data.get("models", None)
    if not models:
        raise RuntimeError(f"{path} is not a valid models file")

    print("Loaded Model Successfully.")
    return models, last_file_trained
