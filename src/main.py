import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading

import config as config
import preprocessing as pre

transliterating = False


def handle_asr() -> None:
    """This will take in mic input and manage the higher level pipeline of the ASR"""
    stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1)

    stream.start()
    curr = np.zeros(config.FRAME_SIZE)

    while transliterating:
    # for i in range(0,3):
        
        prev = curr
        curr, _ = stream.read(config.FRAME_SIZE)

        # preprocessing
        frames = pre.define_frames(prev, curr)
        windows = pre.to_hann_window(frames)
        log_mels = pre.to_log_mel(windows)
        mfccs = pre.to_mfcc(log_mels)

        for mfcc in mfccs:
            print(mfcc)

    stream.close()
    

def main() -> int:
    if not sd.query_devices(kind='input'):
        print("No Input Device Detected...")
        return 1

    global transliterating
    transliterating = True

    asr_thread = threading.Thread(target=handle_asr)
    asr_thread.start()

    # use main thread for UI and as soon as it closes, cleanup and exit

    # transliterating = False
    asr_thread.join()
    return 0

if __name__ == '__main__':
    main()