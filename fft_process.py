import numpy as np
import pyeeg as pe


band = [4, 8, 12, 16, 25, 45]  # 5 bands
window_size = 500  # Averaging band power of 2 sec
step_size = 31  # Each 0.125 sec update once
sample_rate = 250  # Sampling rate of 250 Hz

meta = []

dir_path = "openbci_emotion_eeg/"
filename = "happy2"

data = np.loadtxt(dir_path + filename + ".txt", delimiter=",", skiprows=5, usecols=(1, 2, 4, 5, 6, 7, 8))
data = data.T

print(data.shape)

start = 0

while start + window_size < data.shape[1]:
    meta_data = []  # meta vector for analysis
    for i in range(len(data)):
        X = data[i][start: start + window_size]  # Slice raw data over 2 sec, at interval of 0.125 sec
        Y = pe.bin_power(X, band, sample_rate)  # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
        meta_data = meta_data + list(Y[0])

    meta.append(np.array(meta_data))
    start = start + step_size

meta = np.array(meta)
print(meta.shape)
np.save("out\\fft-" + filename, meta, allow_pickle=True, fix_imports=True)
