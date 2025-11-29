"""Audio Processing Stuff"""
SAMPLE_RATE = 16000             # this is usually enough for speech

FRAME_SIZE = SAMPLE_RATE // 5   # 20ms (1s = 100ms; 1/5s = 20ms)
FRAME_HOP = FRAME_SIZE // 4     # 5ms

N_MFCC = 12                     # 12 dimensional MFCC embeddings


"""HMM Stuff"""
N_EM_ITER = 50                  # max amount of Expectation-Maximation iterations

PHONEMES = [                    # all of the phonemes the TIMIT dataset contains
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ax",
    "ax-h",
    "axr",
    "ay",
    "b",
    "bcl",
    "ch",
    "d",
    "dcl",
    "dh",
    "dx",
    "eh",
    "el",
    "em",
    "en",
    "eng",
    "epi",
    "er",
    "ey",
    "f",
    "g",
    "gcl",
    "h#",               # utterance boundary
    "hh",
    "hv",
    "ih",
    "ix",
    "iy",
    "jh",
    "k",
    "kcl",
    "l",
    "m",
    "n",
    "ng",
    "nx",
    "ow",
    "oy",
    "p",
    "pau",
    "pcl",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "tcl",
    "th",
    "uh",
    "uw",
    "ux",
    "v",
    "w",
    "y",
    "z",
    "zh",
]
