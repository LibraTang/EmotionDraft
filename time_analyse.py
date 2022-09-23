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


time_sequence = [1, 2, 3, 4, 5, 6]  # 按时间顺序排列文件

time_x = [150, 'C', 'G', 90, 120, 'E']  # time x轴

# 划分子图
fig, axes = plt.subplots(2, 1)
valence_ax = axes[0]
arousal_ax = axes[1]

# 图片大小
fig.set_size_inches(16, 9)

# 坐标轴名称
valence_ax.set_xlabel('factor')
valence_ax.set_ylabel('valence')
arousal_ax.set_xlabel('factor')
arousal_ax.set_ylabel('arousal')

# 读取所有受试者的数据
for i in range(1, 19):
    # 音乐2的数据需要跳过受试者10、11、17
    if i in [10, 11, 17]:
        continue

    # 记录valence和arousal的数值
    valence_arr = []
    arousal_arr = []

    # 按时间顺序读取相关的eeg数据
    for j in time_sequence:
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

    valence_ax.plot(time_x, valence_arr)
    arousal_ax.plot(time_x, arousal_arr)

plt.show()
