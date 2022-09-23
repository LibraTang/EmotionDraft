import numpy as np
import mne
import matplotlib.pyplot as plt
import emotion_model as em

from sklearn.preprocessing import normalize
from fft_process import fft_process
from mne.io import read_raw_edf


def predict(data, model):
    output = model.predict(data)
    return np.mean(output)


tempo_sequence = [4, 5, 1]  # 按节奏重排文件顺序
tone_sequence = [2, 6, 3]  # 按音调重排文件顺序

tempo_x = [90, 120, 150]  # tempo x轴
tone_x = ['C', 'E', 'G']  # tone x轴

# 划分子图
fig, axes = plt.subplots(2, 2)
valence_tempo_ax = axes[0, 0]
valence_tone_ax = axes[0, 1]
arousal_tempo_ax = axes[1, 0]
arousal_tone_ax = axes[1, 1]

# 图片大小
fig.set_size_inches(16, 9)

# 坐标轴名称
valence_tempo_ax.set_xlabel('tempo(bpm)')
valence_tempo_ax.set_ylabel('valence')
valence_tone_ax.set_xlabel('tone')
valence_tone_ax.set_ylabel('valence')
arousal_tempo_ax.set_xlabel('tempo(bpm)')
arousal_tempo_ax.set_ylabel('arousal')
arousal_tone_ax.set_xlabel('tone')
arousal_tone_ax.set_ylabel('arousal')

# 读取所有受试者的数据
for i in range(1, 19):
    # 音乐2的数据需要跳过受试者10、11、17
    if i in [10, 11, 17]:
        continue

    # 记录valence和arousal的数值
    valence_tempo_arr = []
    arousal_tempo_arr = []
    valence_tone_arr = []
    arousal_tone_arr = []

    # 读取节奏相关的eeg数据
    for j in tempo_sequence:
        eeg_path = f'music_emotion_eeg/music2/S{i:02d}/S{i:02d}E{j:02d}_filtered.edf'
        raw = read_raw_edf(eeg_path, preload=False)

        events_from_annot, event_dict = mne.events_from_annotations(raw)

        start = events_from_annot[0][0]
        end = events_from_annot[1][0]

        rawEEG = raw.get_data()  # 读取原始信息
        music_eeg = rawEEG[[0, 1, 5, 6, 10, 11, 14, 15], start:end+1]  # 获取指定通道听音乐时段的eeg

        fft_data = fft_process(music_eeg)
        fft_data = normalize(fft_data)

        score_valence = predict(fft_data, em.Val_R)
        score_arousal = predict(fft_data, em.Aro_R)

        valence_tempo_arr.append(score_valence)
        arousal_tempo_arr.append(score_arousal)

        print("Valence: %f" % score_valence)
        print("Arousal: %f" % score_arousal)

    valence_tempo_ax.plot(tempo_x, valence_tempo_arr)
    arousal_tempo_ax.plot(tempo_x, arousal_tempo_arr)

    # 读取音调相关的eeg数据
    for k in tone_sequence:
        eeg_path = f'music_emotion_eeg/music2/S{i:02d}/S{i:02d}E{k:02d}_filtered.edf'
        raw = read_raw_edf(eeg_path, preload=False)

        events_from_annot, event_dict = mne.events_from_annotations(raw)

        start = events_from_annot[0][0]
        end = events_from_annot[1][0]

        rawEEG = raw.get_data()  # 读取原始信息
        music_eeg = rawEEG[[0, 1, 5, 6, 10, 11, 14, 15], start:end+1]  # 获取指定通道听音乐时段的eeg

        fft_data = fft_process(music_eeg)
        fft_data = normalize(fft_data)

        score_valence = predict(fft_data, em.Val_R)
        score_arousal = predict(fft_data, em.Aro_R)

        valence_tone_arr.append(score_valence)
        arousal_tone_arr.append(score_arousal)

        print("Valence: %f" % score_valence)
        print("Arousal: %f" % score_arousal)

    valence_tone_ax.plot(tone_x, valence_tone_arr)
    arousal_tone_ax.plot(tone_x, arousal_tone_arr)

plt.show()
