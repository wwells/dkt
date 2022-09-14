from dkt.DKTModel import DKTModel
from dkt.data_prep import read_file, transform_data, split_dataset

import tensorflow as tf


def run():
    # config
    print("-- APPLYING CONFIGURATION --\n")
    lstm_units = 32
    time_shift = False
    dataset = 'data/toy.txt'
    # dataset = 'data/assistments.txt'
    test_split = .2
    val_split = .2

    epochs = 10
    verbosity = 2
    log_dir = 'logs'
    weights_dir = 'weights/bestmodel'


    # load data
    print("-- LOADING DATA --\n")
    print(f"Loading Dataset {dataset}")
    dataset, num_students, num_exercises = read_file(dataset)

    print(f"Total observations: {len(dataset)}")
    print(f"Total num_students: {num_students}")
    print(f"Total num_problems: {num_exercises}")

    print("\n-- TRANSFORMING DATA --\n")

    trans_dataset, n_zero_batches, features_depth, exercise_depth = transform_data(
        dataset, num_students=num_students, num_exercises=num_exercises, time_shift=time_shift)

    print(f"number of batches (zero indexed): {n_zero_batches}")
    print(f"features_depth: {features_depth}")
    print(f"exercise_depth: {exercise_depth}")

    # For use in testing to review the transformed_data
    elem = next(iter(trans_dataset))
    print(elem)

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
        exercises_depth=exercise_depth,
        lstm_units=lstm_units
    )

    model.compile(
        optimizer='rmsprop'
    )

    print(model.summary())
    print("\n-- COMPILING DONE --\n")

    print("\n-- TRAINING MODEL --\n")
    model.fit(
        dataset=train_set,
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
    model.evaluate(test_set, verbose=verbosity)

    print("\n-- TESTING DONE --\n")


if __name__ == '__main__':
    run()
