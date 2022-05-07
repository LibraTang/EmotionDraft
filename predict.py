import joblib
import numpy as np
import threading
from sklearn.preprocessing import normalize
from queue import Queue
from fft_process import fft_process
from get_data import get_data


# load trained model
Val_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/val_model.pkl")
Aro_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/aro_model.pkl")
Dom_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/dom_model.pkl")
Lik_R = joblib.load("C:/Users/Libra/OneDrive - std.uestc.edu.cn/UR/BCI/model/DEAP_Emotion/lik_model.pkl")

# 线程通信队列
queue = Queue()


# 线程1，实时收集EEG数据
def thread_collect_data():
    print('Thread %s is running...' % threading.current_thread().name)
    while True:
        data = get_data()
        queue.put(data)


def predict(data, model):
    output = model.predict(data)
    return np.mean(output)


print('Thread %s is running...' % threading.current_thread().name)
# 启动收集EEG线程
t = threading.Thread(target=thread_collect_data, name='CollectThread')
t.start()

while True:
    # 从队列中取出data，若没有data则阻塞
    raw_data = queue.get()
    print("Get latest EEG data...")

    # fft处理
    fft_data = fft_process(raw_data[1:9, :])
    fft_data = normalize(fft_data)

    # start prediction
    score_valence = predict(fft_data, Val_R)
    score_arousal = predict(fft_data, Aro_R)

    print("Valence: %f" % score_valence)
    print("Arousal: %f" % score_arousal)

    if score_valence >= 5 and score_arousal >= 5:
        print("Current emotion prediction: Happy/Excited")
    elif score_valence <= 5 and score_arousal >= 5:
        print("Current emotion prediction: Angry/Frustrated")
    elif score_valence <= 5 and score_arousal <= 5:
        print("Current emotion prediction: Depressed/Tired")
    else:
        print("Current emotion prediction: Relaxed/Calm")


