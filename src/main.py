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
    curr = np.zeros(config.FRAME_SIZE)      # COULD cause confusion at startup

    # while transliterating:
    for i in range(0,3):
        
        prev = curr
        curr, _ = stream.read(config.FRAME_SIZE)

        frames = pre.define_frames(prev, curr)
        windows = pre.to_hann_windows(frames)
        pre.to_log_mel(windows)

    stream.close()
    

def main() -> int:
    if not sd.query_devices(kind='input'):
        print("No Input Device Detected...")
        return 1

    global transliterating
    transliterating = True

    asr_thread = threading.Thread(target=handle_asr)
    asr_thread.start()

    # UI call here

    asr_thread.join()    
    return 0

if __name__ == '__main__':
    main()