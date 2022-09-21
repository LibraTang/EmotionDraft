import joblib
import numpy as np
import mne

from sklearn.preprocessing import normalize
from fft_process import fft_process
from mne.io import read_raw_edf

def predict(data, model):
    output = model.predict(data)
    return np.mean(output)


# load trained model
Val_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/val_model.pkl")
Aro_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/aro_model.pkl")
Dom_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/dom_model.pkl")
Lik_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/lik_model.pkl")

raw = read_raw_edf("music_emotion_eeg/S01E01_filtered.edf", preload=False)

events_from_annot, event_dict = mne.events_from_annotations(raw)

start = events_from_annot[0][0]
end = events_from_annot[1][0]

rawEEG = raw.get_data()  # 读取原始信息
music_eeg = rawEEG[[0, 1, 5, 6, 10, 11, 14, 15], start:end+1]  # 获取听音乐时段的eeg

fft_data = fft_process(music_eeg)
fft_data = normalize(fft_data)

score_valence = predict(fft_data, Val_R)
score_arousal = predict(fft_data, Aro_R)

print("Valence: %f" % score_valence)
print("Arousal: %f" % score_arousal)

if 4.5 <= score_valence <= 5.5 and 4.5 <= score_arousal <= 5.5:
    print("Emotion prediction: Calm")
elif score_valence >= 5 and score_arousal >= 5:
    print("Emotion prediction: Happy/Excited")
elif score_valence <= 5 and score_arousal >= 5:
    print("Emotion prediction: Angry/Frustrated")
elif score_valence <= 5 and score_arousal <= 5:
    print("Emotion prediction: Depressed/Tired")
elif score_valence >= 5 and score_arousal <= 5:
    print("Emotion prediction: Relaxed/Calm")
