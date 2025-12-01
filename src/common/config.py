import os
import pathlib

"""Audio Processing Stuff"""
SAMPLE_RATE = 16000             # this is usually enough for speech

FRAME_SIZE = SAMPLE_RATE // 50  # 20ms (1s = 1000ms; 1/50s = 20ms)
FRAME_HOP = FRAME_SIZE // 4     # 5ms

N_MFCC = 12                     # 12 dimensional MFCC embeddings


"""Paths"""

MODELS_PATH = os.path.join(
    pathlib.Path(__file__).parent.parent.parent, "models"
)
HMM_MODEL_PATH = os.path.join(MODELS_PATH, 'hmm.pk1')

TIMIT_ROOT = os.path.join(
    pathlib.Path(__file__).parent.parent.parent, "resources", "TIMIT"
)
TIMIT_TRAIN = os.path.join(TIMIT_ROOT, "TRAIN")
TIMIT_TEST = os.path.join(TIMIT_ROOT, "TEST")

CMU_PATH = os.path.join(
    pathlib.Path(__file__).parent.parent.parent, "resources", "CMUdict"
)
CMU_DICT = os.path.join(CMU_PATH, 'cmudict.dict')


"""GMM + HMM Stuff"""
N_EM_ITER = 50                  # max amount of Expectation-Maximation iterations
N_GAUSSIANS = 6                 # no of gaussians for GMM
N_STATES = 3                    # no of HMM states

PHONEMES = [                    # all of the phonemes the TIMIT dataset contains
    "aa", "ae", "ah", "ao", "aw", "ax",
    "ax-h", "axr", "ay", "b", "bcl", "ch",
    "d", "dcl", "dh", "dx", "eh", "el",
    "em", "en", "eng", "epi", "er", "ey",
    "f", "g", "gcl", "h#", "hh", "hv", 
    "ih", "ix", "iy", "jh", "k", "kcl",
    "l", "m", "n", "ng", "nx", "ow", "oy",
    "p", "pau", "pcl", "q", "r", "s",
    "sh", "t", "tcl", "th", "uh", "uw", 
    "ux", "v", "w", "y", "z", "zh",
]