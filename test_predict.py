'''
temp script to load a trained model, transform a single sequence representing a single user's problem attempts
and save the predictions next to the original inputs in a csv
'''
from dkt.model import DKTModel
from dkt.data_prep import read_file, transform_data

import dkt.metrics as metrics

import tensorflow as tf
import pandas as pd


def main():

    # params are hard coded from initial model setup.
    features_depth = 312
    skills_depth = 156
    max_sequence = 499
    num_students = 9423

    lstm_units = 100
    dropout_rate = .2

    weights_dir = 'weights/bestmodel'

    print("\n-- LOADING MODEL --\n")

    # reinit the model, but load weights from training run
    model = DKTModel(
        features_depth=features_depth,
        skills_depth=skills_depth,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )

    model.compile(
        optimizer='rmsprop',
        metrics=[
            metrics.BinaryAccuracy(),
            metrics.AUC(),
            metrics.Precision(),
            metrics.Recall(),
        ]
    )

    model.load_weights(weights_dir)
    # print(model.summary())

    print("\n-- SINGLE PREDICTION EXAMPLE --\n")

    df, _, _, _ = read_file('data/khan/single_ka_user.txt')
    pred_sequence_length = len(df)

    dataset, _, _, _ = transform_data(df, num_students, skills_depth, max_sequence, features_depth=features_depth)

    X_test = tf.convert_to_tensor(list(map(lambda x: x[0], dataset)))
    single_entry = X_test[:1][0][:1]
    prediction = model.predict(single_entry)[0]

    print("\n-- SAVING OUTPUT --\n")

    '''
    for i in range(len(df)):
        print(f"X={df.loc[i]}, Predicted={prediction[0][i]}")
    '''
    preds = pd.DataFrame(
        data=prediction,
        columns=['Sk_' + str(i + 1)for i in range(prediction.shape[1])]
    ).head(pred_sequence_length)

    final = pd.concat([df, preds], axis=1)
    final.to_csv('data/khan/test_pred_output.csv', index=False)


if __name__ == '__main__':
    main()
