from keras.layers.core import Masking
from keras.layers import LSTM, Dense
from keras.losses import binary_crossentropy
from keras import Input

import tensorflow as tf


def get_target(y_true, y_pred):
    """this is used in the loss function"""
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, -1), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred


class DKTModel(tf.keras.Model):
    """ Subclass of a Keras model that can be used to instantiate a DKT model
    """

    def __init__(self, features_depth, skillss_depth, lstm_units):

        inputs = Input(shape=(None, features_depth), name='inputs')

        x = Masking(mask_value=-1)(inputs)

        x = LSTM(lstm_units, return_sequences=True)(x)

        outputs = Dense(skillss_depth, activation='sigmoid', name='dense_outputs')(x)

        super(DKTModel, self).__init__(inputs=inputs,
                                       outputs=outputs,
                                       name="DKTModel")

    def compile(self, optimizer, metrics=None):
        """defines and uses our custom loss function during compilation"""

        def custom_loss(y_true, y_pred):
            y_true, y_pred = get_target(y_true, y_pred)
            return binary_crossentropy(y_true, y_pred)

        super(DKTModel, self).compile(
            loss=custom_loss,
            optimizer=optimizer,
            metrics=metrics,
            experimental_run_tf_function=False)
