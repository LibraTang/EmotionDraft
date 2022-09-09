import time
import joblib
import numpy as np

from pylsl import StreamInlet, resolve_stream
from sklearn.preprocessing import normalize
from fft_process import fft_process


def predict(data, model):
    output = model.predict(data)
    return np.mean(output)


# load trained model
Val_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/val_model.pkl")
Aro_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/aro_model.pkl")
Dom_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/dom_model.pkl")
Lik_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/lik_model.pkl")


print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

inlet = StreamInlet(streams[0])

tmp = []
while True:
    sample, timestamp = inlet.pull_sample()
    tmp.append(sample)
    # 每收集到60s的EEG信号做一次情绪识别
    if len(tmp) >= 256 * 60:
        data = np.array(tmp)
        data = data.T  # channels * samples
        # fft处理
        fft_data = fft_process(data)
        fft_data = normalize(fft_data)

        # start prediction
        score_valence = predict(fft_data, Val_R)
        score_arousal = predict(fft_data, Aro_R)

        current_time = time.strftime("%H:%M:%S", time.localtime())
        print("*******************************\n"
              "Current time: %s" % current_time)
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

        tmp.clear()
