import argparse

import tensorflow as tf

from dkt.model import DKTModel
from dkt.data_prep import read_file, transform_data, split_dataset
import dkt.metrics as metrics


def parse_args():
    parser = argparse.ArgumentParser(prog="Khan DKT Prototype")

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/toy.txt",
        help="the path to the data on disk")

    model_group = parser.add_argument_group(title="Model arguments.")
    model_group.add_argument(
        "--lstm_units",
        type=int,
        default=100,
        help="number of units of the LSTM layer.")

    model_group.add_argument(
        "--dropout_rate",
        type=float,
        default=.2,
        help="fraction of the units to drop.")

    train_group = parser.add_argument_group(title="Training arguments.")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train.")

    train_group.add_argument(
        "--test_split",
        type=float,
        default=.2,
        help="fraction of data to be used for testing (0, 1).")

    train_group.add_argument(
        "--val_split",
        type=float,
        default=.2,
        help="fraction of data to be used for validation (0, 1).")

    return parser.parse_args()


def run(dataset, lstm_units, dropout_rate, epochs, test_split, val_split):
    """simple wrapper to load data, train a DKT model, test the model
    and then an example for making a prediction for a single student by taking a row
    from the test_set"""

    print("-- APPLYING CONFIGURATION --\n")

    # most configuration we get from args, but here's some additional config we use
    # during prototyping
    time_shift = True  # shift the data of engineered feature and truth (see data_prep.transform_data)
    verbosity = 2
    log_dir = 'logs'
    weights_dir = 'weights/bestmodel'

    # load data
    print("-- LOADING DATA --\n")
    print(f"Loading Dataset {dataset}")
    dataset, num_students, num_skills, max_sequence = read_file(dataset)

    print(f"Total observations: {len(dataset)}")
    print(f"Total num_students: {num_students}")
    print(f"Total num_skills: {num_skills}")
    print(f"Max sequence:  {max_sequence}")

    print("\n-- TRANSFORMING DATA --\n")

    trans_dataset, n_zero_batches, features_depth, skills_depth = transform_data(
        dataset, num_students=num_students, num_skills=num_skills, max_sequence=max_sequence, time_shift=time_shift)

    print(f"number of batches (zero indexed): {n_zero_batches}")
    print(f"features_depth: {features_depth}")
    print(f"skills_depth: {skills_depth}")

    # For use in testing to review the transformed_data
    # elem = next(iter(trans_dataset))
    # print(elem)

    print("\n-- SPLITTING DATA --\n")

    train_set, test_set, val_set = split_dataset(
        dataset=trans_dataset,
        n_zero_batches=n_zero_batches,
        test_fraction=test_split,
        val_fraction=val_split
    )

    print("\n-- COMPILING MODEL --\n")

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

    print(model.summary())
    print("\n-- COMPILING DONE --\n")

    print("\n-- TRAINING MODEL --\n")

    model.fit(
        train_set,
        epochs=epochs,
        verbose=verbosity,
        validation_data=val_set,
        callbacks=[
            tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
            tf.keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ]
    )
    print("\n-- TRAINING DONE --\n")

    print("\n-- TESTING MODEL --\n")

    model.load_weights(weights_dir)
    model.evaluate(x=test_set, verbose=verbosity, batch_size=1)

    print("\n-- TESTING DONE --\n")

    print("\n-- SINGLE PREDICTION EXAMPLE --\n")

    X_test = tf.convert_to_tensor(list(map(lambda x: x[0], test_set)))
    single_entry = X_test[:1][0][:1]
    print(f"\nX for prediction shape: {single_entry.shape}")
    print("X to predict: ")
    print(single_entry)
    prediction = model.predict(single_entry)
    print(f"\nprediction shape: {prediction.shape}")
    print("Prediction: ")
    print(prediction)


if __name__ == '__main__':
    args = parse_args()

    run(
        dataset=args.dataset,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        test_split=args.test_split,
        val_split=args.val_split
    )
