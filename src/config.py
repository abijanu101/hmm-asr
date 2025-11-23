SAMPLE_RATE = 16000                             # this is usually enough for speech

FRAME_SIZE = SAMPLE_RATE // 5               # 20ms (1s = 100ms; 1/5s = 20ms)
FRAME_HOP = FRAME_SIZE // 4

N_MFCC = 12