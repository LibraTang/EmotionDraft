import numpy as np
import pyeeg as pe


band = [4, 8, 12, 16, 25, 45]  # 5 bands
window_size = 500  # Averaging band power of 2 sec
step_size = 31  # Each 0.125 sec update once
sample_rate = 250  # Sampling rate of 250 Hz


def fft_process(raw_data):
    start = 0
    meta = []
    while start + window_size < raw_data.shape[1]:
        meta_data = []  # meta vector for analysis
        for i in range(len(raw_data)):
            x = raw_data[i][start: start + window_size]  # Slice raw data over 2 sec, at interval of 0.125 sec
            y = pe.bin_power(x, band, sample_rate)  # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
            meta_data = meta_data + list(y[0])
        meta.append(np.array(meta_data))
        start = start + step_size
    meta = np.array(meta)
    return meta
