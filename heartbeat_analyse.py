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

# 受试者按心率分组
subject_list_60 = [7, 9, 12, 14]
subject_list_65 = [5, 13, 15]
subject_list_70 = [2, 16]
subject_list_75 = [1, 6, 18]
subject_list_80 = [3, 4, 8]

# 按心率排列文件
heartbeat_sequence_60 = [7, 4, 5]
heartbeat_sequence_65 = [7, 8, 9]
heartbeat_sequence_70 = [7, 8, 9]
heartbeat_sequence_75 = [7, 8, 1]
heartbeat_sequence_80 = [7, 5, 8]


heartbeat_x = ['1x', '1.5x', '2x']  # heartbeat x轴

# 划分子图
fig, axes = plt.subplots(2, 1)
valence_ax = axes[0]
arousal_ax = axes[1]

# 图片大小
fig.set_size_inches(16, 9)

# 坐标轴名称
valence_ax.set_xlabel('heartbeat')
valence_ax.set_ylabel('valence')
arousal_ax.set_xlabel('heartbeat')
arousal_ax.set_ylabel('arousal')


def start(subject_list, heartbeat_sequence):
    for i in subject_list:
        # 记录valence和arousal的数值
        valence_arr = []
        arousal_arr = []

        # 按时间顺序读取相关的eeg数据
        for j in heartbeat_sequence:
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

            valence_arr.append(score_valence)
            arousal_arr.append(score_arousal)

            print("Valence: %f" % score_valence)
            print("Arousal: %f" % score_arousal)

        valence_ax.plot(heartbeat_x, valence_arr)
        arousal_ax.plot(heartbeat_x, arousal_arr)


start(subject_list_60, heartbeat_sequence_60)
start(subject_list_65, heartbeat_sequence_65)
start(subject_list_70, heartbeat_sequence_70)
start(subject_list_75, heartbeat_sequence_75)
start(subject_list_80, heartbeat_sequence_80)

plt.show()
