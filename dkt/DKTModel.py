from keras.models import Sequential
from keras.layers.core import Masking
from keras.layers import LSTM, Dense
from keras.backend import binary_crossentropy

import theano.tensor as Th


def get_DKT_model(lstm_units, dense_units):
    model = Sequential()

    model.add(Masking(-1.0))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dense(units=dense_units, activation='sigmoid'))
    return model


def loss_function(y_true, y_pred):
    skill = y_true[:, :, 0:num_skills]
    obs = y_true[:, :, num_skills]
    print(f"---in loss_function")
    print(f"--y_true: {y_true}")
    print(f"evaluaging y_true: {y_true.eval()}")
    print(f"--y_pred: {y_pred}")
    print(f"--skill: {skill}")
    print(f"--obs: {obs}")
    rel_pred = Th.sum(float(y_pred) * float(skill), axis=2)  # converted to float

    # keras implementation does a mean on the last dimension (axis=-1) which
    # it assumes is a singleton dimension. But in our context that would
    # be wrong.
    return binary_crossentropy(rel_pred, obs)