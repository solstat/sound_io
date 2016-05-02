




import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
WAV_RATE = 44100
import numpy as np
import os
from scipy.io import wavfile


def getSnippets(track, len_secs = 30):
    len_bytes = len_secs * WAV_RATE
    num_snippets = len(track) / len_bytes #rounds down
    snippets = [None for i in range(num_snippets)]
    for i in range(num_snippets):
        idx = [i*len_bytes, (i+1)*len_bytes]
        snippets[i] = channel0[idx[0], idx[1]]

def snippetToSpectrogram(snippet, window_len = WAV_RATE / 20):
    num_windows = len(snippet) / window_len
    windows = np.zeroes(shape = (num_windows, window_len))
    for i in range(num_windows):
        idx = [i*window_len, (i+1)*window_len]
        windows[i, :] = dct(snippet[idx[0], idx[1]])
    return windows

def spectrogramToSnippet(spectrogram):
    snippet = np.zeros(shape = (spectrogram.shape[0]* spectrogram.shape[1]))
    num_windows = spectrogram.shape[0]
    for i in range(num_windows):
        idx = [i*window_len, (i+1)*window_len]
        snippet[idx[0], idx[1]] = idct(windows[i, :])
    return snippet



DEMO_BERLIOZ_PATH = "data/Symphonie_Fantastique/wav"
MOVEMENTS = [ "aux_champs3.wav",
	"nuit_sabbath5.wav",
	"reveries1.wav",
	"supplice_oh_shit4.wav",
	"un_bal2.wav"]
DEMO_BACH_PATH = ????

rate, data = wavfile.read(os.path.join(DEMO_BERLIOZ_PATH, MOVEMENTS[0]))
assert rate == WAV_RATE
track = data.T[0]
snippets = getSnippets(track)
for snippet in snippets:
    data_rectangle = snippetToSpectrogram(snippet)
    # tf.SGD_step_or_whatever(data_rectangle)
    return


data_rectangle = # tf.predict_or_hallucinate_or_whatever()
# TO DO: get bach snippet, feed it through cnn, convert to WAV
wavfile.write(os.path.join(DEMO_BERLIOZ_PATH, "synthetic_extra_mvt.wav"), WAV_RATE, data)
