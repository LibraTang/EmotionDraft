import joblib
import numpy
import numpy as np
from sklearn.preprocessing import normalize


def predict(data_test, model):
    """
    arguments:  data_test: testing dataset
                model: scikit-learn model

    return:     float
    """
    output = model.predict(data_test)

    return numpy.mean(output)


filename = "fft-happy2.npy"

# load testing dataset
with open("out/" + filename, "rb") as file:
    data_test = np.load(file)
data_test = normalize(data_test)

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
