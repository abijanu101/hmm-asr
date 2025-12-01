import os
import joblib

from src.common import config
from hmmlearn import hmm


def create() -> dict[str, hmm.GMMHMM]:
    """Create HMM+GMM Models"""

    models = {}
    for phn in config.PHONEMES:
        models[phn] = hmm.GMMHMM(
            n_components=config.N_STATES,  # start, middle, end
            n_mix=config.N_GAUSSIANS,
            algorithm="viterbi",
            n_iter=config.N_EM_ITER,  # MAX EM Iterations
            init_params='',
            covariance_type='diag',
            verbose=True
        )

    print(f"A total of {len(config.PHONEMES)} new HMM+GMM models created.")
    return models


def persist(models: dict[str, hmm.GMMHMM]) -> None:
    """Persist HMM Models along with information about the last file successfully trained on in the dataset"""

    print("Saving...")
    joblib.dump(models, config.HMM_MODEL_PATH)

    print(f"Saved models as file '{config.HMM_MODEL_PATH}' successfully.")


def load(path: str) -> dict[str, hmm.GMMHMM]:
    """Load HMM Models along with the information about the last file successfully trained on in the dataset"""

    if not os.path.exists(path):
        raise RuntimeError(f"{path} does not exist")

    print("Loading HMM+GMM Models...")
    models = joblib.load(path)
    
    print("Loaded Model Successfully.")
    return models
