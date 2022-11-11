from mne.io import read_raw_edf


tempo_sequence = [4, 5, 1]  # 按节奏重排文件顺序

for j in tempo_sequence:
    eeg_path = f'music_emotion_eeg/music1/S02/S02E{j:02d}_filtered.edf'
    raw = read_raw_edf(eeg_path, preload=False)
    # 4-45Hz的带通滤波
    raw.load_data().filter(4., 45.)
    # 300Hz下的功率谱密度图
    raw.compute_psd(fmin=4, fmax=45, picks=['EEG Fp1-Pz', 'EEG Fp2-Pz', 'EEG F3-Pz', 'EEG F4-Pz', 'EEG T3-Pz', 'EEG T4-Pz', 'EEG P3-Pz', 'EEG P4-Pz']).plot()
    # 降采样至128Hz
    raw.resample(128)
    raw.compute_psd(fmin=4, fmax=45, picks=['EEG Fp1-Pz', 'EEG Fp2-Pz', 'EEG F3-Pz', 'EEG F4-Pz', 'EEG T3-Pz', 'EEG T4-Pz', 'EEG P3-Pz', 'EEG P4-Pz']).plot()

