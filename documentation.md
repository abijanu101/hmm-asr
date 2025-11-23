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

### Windowing
* Applied the Hann Window to avoid Spectral Leakage before applying the Fourier Transform

## 22/11/25: Preprocessing Complete

### File Organization
Improved the file organization by separating logic into smaller files

### Fourier Transformer
Initially I went with the Fast Fourier Transform and got a 3200 element list containing complex numbers. Here, the complex number stored as the k^th element embeds information about the magnitude and phase of the frequency k / FRAME_SIZE * SAMPLING_RATE.

Apparently, the phase is hardly relevant and we typically just care about the frequencies and their magnitude so we just use abs() to calculate the Euclidean Norm for each of the elements, giving us an array of reals (representing Magnitudes for each frequency) as opposed to complex numbers.

### Mel
Going into this, I already knew what Mel is for, but not so much the how. To begin, the Mel Filterbank is a matrix of dimensions BAND_COUNT x FRAME_SIZE // 2 + 1.

Neglecting the n // 2 + 1 for now, lets talk about what each row represents.

Each of these rows represents a band. A band is essentially a row of weights such that plotting it along the frequencies shows a single triangular spike for a certain range of frequencies.

By increasing the energies for the earlier frequencies in a narrow and steep manner and a wide and softer manner for larger frequencies, we basically add more significance to lower frequencies broadly sort of like how tf-idf raises the relative importance for rare words.

As for the columns, I learned when trying to multiplying my FFT matrix by the Mel Filterbank that even if I explicitly tell the Mel filterbank the dimensions of n_fft = FRAME_SIZE; it always gives a BAND_COUNT x (FRAME_SIZE // 2 + 1) matrix rather than a BAND_COUNT x FRAME_SIZE matrix.

I looked into this and learned that this was due to something called the conjugate symmetric property of the FFT's result 

### Conjugate Symmetricity of FFT and Why n // 2 + 1
The expanded fourier transform can be used to prove that for all k, X[n-k] and X[k] are conjugate-symmetric (meaning X[n-k]* = X[k]*). So, we simply discard the last half of frequency bins since they are inferable.

The rfft() is relevant here since. For real-valued inputs, this function basically just gives you quickly calculated fft() results with only the first n // 2 + 1 frequency bins, which is exactly what we want

### Log-Mel Proper
All of this is to say, I took a rfft()
I then squared it to construct it into a power spectrum (known to give better results).
Finally, I multiplied it with a mel filterbank to give me the log-mel vector