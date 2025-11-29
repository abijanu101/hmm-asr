import os
import pathlib

from src.common import config
from src.models import hmm

def main() -> None:
    models_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent, "models", "hmm.pk1"
    )

    # loading
    models = {}
    last_file_trained = None
    
    if not os.path.exists(models_path):
        confirmation = input(f"{models_path} doesn't exist, do you want to create a new set of HMM models? [Y/n]: ")

        if confirmation != 'Y':
            print("Okay, I can't really do anything then")
            return

        models = hmm.create()
        hmm.persist(models, None, models_path)
    else:
        models, last_file_trained = hmm.load(models_path)

    
    # training
    
        



if __name__ == "__main__":
    main()
