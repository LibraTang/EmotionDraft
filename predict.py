import time

import joblib
import numpy as np
import fft_process
import get_data
import threading
from sklearn.preprocessing import normalize
from queue import Queue


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
        data = get_data.get_data()
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
    # while len(shared_variable.data) == 0:
    #     print("EEG data is empty...")
    #     time.sleep(30)
    #     continue

    print("Get latest EEG data...")
    # print(raw_data.shape)
    # 取出最新收集的EEG并清空共享数组
    # raw_data = shared_variable.data[-1]
    # shared_variable.data.clear()

    # fft处理
    fft_data = fft_process.fft_process(raw_data[1:9, :])
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


