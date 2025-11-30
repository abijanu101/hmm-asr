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

Applied the Hann Window to avoid Spectral Leakage before applying the Fourier Transform

## 23/11/25: Preprocessing Complete

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

### Finishing All Pre-processing

With all this theory knowledge, I just wrote like 5 lines of code and now I have working feature extraction ready for my HMM to be trained on.

## 29/11/25: HMM+GMM

### TIMIT Dataset

I've locally downloaded the TIMIT dataset. This dataset contains already split TEST and TRAIN partitions, each containing 8 Dialectical Regions. One potential problem with the dataset is audio quality; it is all highly polished with no background noise, but this is the best we can do in the time provided.

One potential solution here is to include a catch-all UNK phoneme for room-tone and ambient noises reasonably acceptable in speech audio. We could train it on a bunch of room-tone sound files, but I am gonna hold off on that for now.

### File Structure

Learned what ```__init__.py``` files do and restructured the project.

### HMM+GMM Setup

I learned that we actually set up multiple HMM+GMM models; one for each phoneme. Here, each model has 3 states representing the start, middle and end of each respective phoneme utterance. So, I used a short python script to extract every single phoneme used in TIMIT and initialized the models in ```init/hmm.py```.

While combing through the list of phonemes and reading about how sometimes people merge certain phonemes into one to reduce the 61 phonemes to around 39, I realized a potential problem; What if the CMUdict input is expected to be different. Turns out it is, but thats not really that big of a problem since frequently used mappings exist regardless.

One might stop and think that this means we only need 39 HMM models rather than the 61 for each TIMIT phoneme. This is not necesassarily wrong, but I decided against it because its my first time training and I dont have the time to come back and redo everything if my mappings end up losing information or causing troubles later on.

Next, I used ```joblib``` to persist and load the HMM models. This was necessary so we could have continue-where-you-left-off behavior for both training and also just to load weights from a file for the actual deployment.

## 30/11/25: Training

### Frame Definitions

Unlike in realtime, we now have a finite ending. However, it is possible that the recording is not perfectly representable with the fixed FRAME_SIZE and FRAME_HOP we have chosen. To handle this boundary case, I previously thought to insert zeroes (i.e., silence) for the few frames that end beyond the buffer size to avoid loss of information, however I reviewed the dataset and each file ends and starts with h# anyways (a special silence indicator for utterance boundaries)

So, I will let all samples except the last few be seen 4 times each with the last sample appearing either 0 or 1 times in total.

### Overlapping Frames

For the sake of example, consider a frame_size=3 and frame_hop=1 with the following .phn
0 2 sh
2 3 g

The frames for this will be {s[0],s[1],s[2]}, {s[1],s[2],s[3]}. Maybe even add {s[2],s[3],NULL}, {s[3],NULL,NULL}

In this case, the problem is that {s[1],s[2],s[3]} streches across the 2nd sample in a way that it lies within both the first and second phoneme. Such frames are called ambiguous frames and they can be dealt with in many ways.

The way I chose to deal with them is that any given will be included in the frame if there is an overlap of 50% or more. In practice, this could mean that if a frame is perfectly centered at the phoneme boundary, the same frame will be fed to both of the HMMs - once for the first 50% and second for the next 50%.

My hope is that this will help register co-articulation better.

### Fitting the GMM+HMMs

Eventually, I finished writing ```src/train/hmm.py```. However, upon running it, I ran into the following error:

```File "D:\Coding\Projects\hmm-asr\asr-env\Lib\site-packages\sklearn\cluster_kmeans.py", line 871, in _check_params_vs_input
    raise ValueError(
        f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
    )
ValueError: n_samples=1 should be >= n_clusters=3
```

I plugged this into GPT and apparently, the number of samples for each phoneme must be greater than or equal to the amount of Gaussians for the GMM. So, even though I grouped each frame by phoneme in the form grouped = {'sh': [], 'g': [mfccs[0], mfccs[3]], 'h#': [mfccs[1], mfccs[2]]}.

One option was to simply add a condition to avoid triggering this error, but this risks losing data. A solution I thought of here was to only fit after processing ALL files for each speaker to maximize exploitation. But before trying that I thought to run the same program with the much quicker condition method to see a proof of concept.

### Debugging Hell

Turns out this too gave me the same error, even though i did ```len(frames) < config.N_GAUSSIANS```. Even after deleting my model and recreating it with less Gaussians, I got the exact same error and the ```n_clusters=3``` persisted even if my N_GAUSSIANS was far away from 3. Eventually, I just plugged in 10 Gaussians instead just to see if the n_clusters was independent of it and if GPT had misled me.

I ran it and it somehow seemed to have fixed the issue actually? I say this since the next few files ran without raising any warnings or anything, but this too had issues.

```
D:\Coding\Projects\hmm-asr\src\common\mfcc.py:9: UserWarning: Empty filters detected in mel frequency basis. Some channels will produce empty responses. Try increasing your sampling rate (and fmax) or reducing n_mels.
MEL_FILTERBANK = librosa.filters.mel(sr=config.SAMPLE_RATE, n_fft = config.FRAME_SIZE)
Loading HMM+GMM Models...
Loaded Model Successfully.
D:\Coding\Projects\hmm-asr\resources\TIMIT\TRAIN\DR1\FCJF0\SA1
D:\Coding\Projects\hmm-asr\resources\TIMIT\TRAIN\DR1\FCJF0\SA2
D:\Coding\Projects\hmm-asr\resources\TIMIT\TRAIN\DR1\FCJF0\SI1027
Fitting a model with 737 free scalar parameters with only 48 data points will result in a degenerate solution.
D:\Coding\Projects\hmm-asr\asr-env\Lib\site-packages\hmmlearn\hmm.py:809: RuntimeWarning: invalid value encountered in divide
self.covars_ = c_n / c_d
D:\Coding\Projects\hmm-asr\asr-env\Lib\site-packages\hmmlearn\_emissions.py:208: RuntimeWarning: divide by zero encountered in log
log_cur_weights = np.log(self.weights_[i_comp])
D:\Coding\Projects\hmm-asr\resources\TIMIT\TRAIN\DR1\FCJF0\SI1657
Fitting a model with 737 free scalar parameters with only 48 data points will result in a degenerate solution.
```

Well, why did this even run? Well, it was actually because I still had the ```len(frames) < config.N_GAUSSIANS``` condition. Basically, what was happening was that it never fit() anything into the model since all of the phonemes had less than 10 corresponding frames.

After figuring that out and removing the condition for testing purposes, I saw the same

```
File "D:\Coding\Projects\hmm-asr\asr-env\Lib\site-packages\sklearn\cluster\_kmeans.py", line 871, in _check_params_vs_input
raise ValueError(
f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
)
ValueError: n_samples=1 should be >= n_clusters=3.
```

It was clear now that this n_clusters was something entirely unrelated to the amount of Gaussians. Eventually, I found out it's actually the number of STATES. (I confirmed this by changing the state count and n_clusters was no longer equal to 3...)

However, running it now with a modified condition ```len(frames) < config.N_STATES```, STILL yielded Runtime warnings about division by zero in the hmmlearn API's code

```
D:\Coding\Projects\hmm-asr\asr-env\Lib\site-packages\hmmlearn\hmm.py:809: RuntimeWarning: invalid value encountered in divide
  self.covars_ = c_n / c_d
D:\Coding\Projects\hmm-asr\resources\TIMIT\TRAIN\DR1\FCJF0\SI1027
Even though the 'startprob_' attribute is set, it will be overwritten during initialization because 'init_params' contains 's'
Even though the 'transmat_' attribute is set, it will be overwritten during initialization because 'init_params' contains 't'
Fitting a model with 299 free scalar parameters with only 48 data points will result in a degenerate solution.
Even though the 'weights_' attribute is set, it will be overwritten during initialization because 'init_params' contains 'w'
Even though the 'means_' attribute is set, it will be overwritten during initialization because 'init_params' contains 'm'
Even though the 'covars_' attribute is set, it will be overwritten during initialization because 'init_params' contains 'c'
D:\Coding\Projects\hmm-asr\asr-env\Lib\site-packages\hmmlearn\hmm.py:809: RuntimeWarning: divide by zero encountered in divide
  self.covars_ = c_n / c_d
D:\Coding\Projects\hmm-asr\asr-env\Lib\site-packages\hmmlearn\hmm.py:809: RuntimeWarning: invalid value encountered in divide
  self.covars_ = c_n / c_d
```

### The Grouping Solution
I didn't quite know why all these warnings were coming up, but one of them was quite explicit; I needed more datapoints. So, the next step was to try the grouping solution. And lo-and-behold, it really just worked with no problems as soon as i grouped it.

However, I now realized another mistake I made. I had treated the ``lengths`` param in ``.fit()`` as the amount of frames while it was supposed to be an array representing the lengths of the contiguous frames for each phoneme utterance.

Turns out I need to use a ```dict[str, list[list[np.ndarray]]]``` So i could have a list of sequences for each phoneme so i could fit properly.


### Debugging Grouping
I found a logical error that made it such that i only inserted only the first frame in my ```mfccs``` for each labelled phoneme utterance even though there were tens of other ones in that range that I should've inserted that led to really impossible results.