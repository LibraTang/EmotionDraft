import joblib
import numpy as np
import fft_process
import get_data
import threading
import shared_variable
from sklearn.preprocessing import normalize


# 线程1，实时收集EEG数据
def thread_collect_data():
    print('thread %s is running...' % threading.current_thread().name)
    while True:
        get_data.get_data()

def predict(data_test, model):
    """
    arguments:  data_test: testing dataset
                model: scikit-learn model

    return:     float
    """
    output = model.predict(data_test)

    return np.mean(output)


print('thread %s is running...' % threading.current_thread().name)
# 启动收集EEG线程
t = threading.Thread(target=thread_collect_data(), name='CollectThread')
t.start()

# 轮询共享数组
while shared_variable.data.size == 0:
    continue

# 取出收集的EEG
raw_data = shared_variable.data[0]

# fft处理
fft_data = fft_process.fft_process(raw_data)
data_test = normalize(fft_data)

# load trained model
Val_R = joblib.load("model/val_model.pkl")
Aro_R = joblib.load("model/aro_model.pkl")
Dom_R = joblib.load("model/dom_model.pkl")
Lik_R = joblib.load("model/lik_model.pkl")


# start prediction
print("Valence: %f" % predict(data_test, Val_R))

print("Arousal: %f" % predict(data_test, Aro_R))

print("Domain: %f" % predict(data_test, Dom_R))

print("Like: %f" % predict(data_test, Lik_R))
