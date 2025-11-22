import sounddevice as sd
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import threading

SAMPLE_RATE = 16000                     # this is usually enough for speech

FRAME_SIZE = int(SAMPLE_RATE / 5)       # 20ms (1s = 100ms; 1/5s = 20ms)
FRAME_HOP = int(FRAME_SIZE / 4)

transliterating = False

def processFrames(curr_frames: list[np.ndarray]):

    hann_window = sp.get_window('hann', FRAME_SIZE)
    windows = []
    for i in range(0, 4):
        windows.append(curr_frames[i] * hann_window)
    

def handleASR() -> None:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1)

    stream.start()
    curr = np.zeros(FRAME_SIZE)

    # while transliterating:
    for i in range(0,3):
        curr_frames = []
        drift = FRAME_HOP
        
        prev = curr
        curr, _ = stream.read(FRAME_SIZE)
        curr = curr.flatten()
        while drift <= FRAME_SIZE:
            frame = np.concatenate([prev[drift:FRAME_SIZE], curr[0:drift]])
            curr_frames.append(frame)
            drift += FRAME_HOP

        processFrames(curr_frames)

    stream.close()
    

def main() -> int:
    if not sd.query_devices(kind='input'):
        print("No Input Device Detected...")
        return 1

    global transliterating
    transliterating = True
    asr_thread = threading.Thread(target=handleASR)
    asr_thread.start()

    asr_thread.join()
    
    return 0

if __name__ == '__main__':
    main()