# ASR Project Documentation

Hopefully this will be helpful for the writeup

## 22/11/25: Getting Started, Signal Processing, etc.

### Input
For audio input, I've used the _sounddevice_ python library. I initially used .rec() and .play() to get started and later switched to the InputStream method for recordings of non-fixed durations.

Since 16kHz seems to be enough and I want to make a responsive-feeling real-time ASR system, I am going with that instead of 48kHz.

### Virtual Environment and Shell Script
I was reading the documentation for the  _sounddevice_ python library and it mentioned that I should set this sort of structure up for portability and keeping the global packages clean and conflict-free. So, I ended up writing a .bat so that my teammates can just double click and get started.

### Framing
The InputStream solution has both async and sync options. But since the frames must be overlapping, I need to maintain a _prev_. I am doing this so that I can have all of the frames defined by the overlap of two blocks of size _n_ after reading each block of size _n_ in an easy to implement loop.

Apparently a frame size of 20-25ms and a frame drift of 5-10ms is the established sweet-spot, so I chose 20ms and 5ms respectively for these constants.

### MFCC
* Applied the Hann Window to avoid Spectral Leakage before applying the Fourier Transform
To apply the Mel Filterbank, DCT, etc.