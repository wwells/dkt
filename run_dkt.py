from dkt.DKTModel import get_DKT_model, loss_function
from dkt.data_prep import load_dataset, transform_data, read_file_alt, transform_data_alt


def run():
    # config
    print("-- APPLYING CONFIGURATION --\n")
    lstm_units = 32
    dense_units = 32
    dataset = 'data/toy.txt'
    #split_file = 'data/toy_split.txt'
    logs_dir = 'logs'
    weights = 'weights/bestmodel'
    shuffle = True

    # load data
    print("-- LOADING DATA --\n")
    print(f"Loading Dataset {dataset}")
    dataset, num_students, num_exercises = read_file_alt(dataset)
    #print(dataset)
    print(f"Total observations: {len(dataset)}")
    print(f"Total num_students: {num_students}")
    print(f"Total num_problems: {num_exercises}")
    #training_seqs, testing_seqs, num_skills, num_students = load_dataset(dataset, split_file)
    #print(f"Training Sequences: {len(training_seqs)}")
    #print(f"---training_seqs: {training_seqs}")
    #print(f"Testing Sequences: {len(testing_seqs)}")
    #print(f"Total Number of skills: {num_skills}")
    #print(f"Total Number of Students in Dataset: {num_students}")

    print("\n-- TRANSFORMING DATA --\n")

    trans_dataset, length, features_depth, exercise_depth = transform_data_alt(dataset, num_students=num_students, num_exercises=num_exercises)

    print(f"trans_dataset: {trans_dataset}")
    print(f"length: {length}")
    print(f"features_depth: {features_depth}")
    print(f"exercise_depth: {exercise_depth}")

    elem = next(iter(trans_dataset))
    print(elem)

    elem2 = next(iter(trans_dataset))
    print(elem2)

    elem3 = next(iter(trans_dataset))
    print(elem3)

    elem4 = next(iter(trans_dataset))
    print(elem4)



    # set up functions to transform train/test data in data_prep.py
    '''
    dataset, length, features_depth, skill_depth = transform_data(
        training_seqs,
        num_skills=num_skills,
        num_students=num_students
    )
    elem = next(iter(dataset))
    list(elem[0])
    '''
    # get num_skills and input_shape from data
    input_shape = (5, 100, 5)

    # compile model
    '''
    print("\n-- COMPILING MODEL --\n")

    model = get_DKT_model(
        lstm_units=lstm_units,
        dense_units=dense_units
    )

    model.compile(
        loss=loss_function,
        optimizer='rmsprop'
    )
    model.build(
        input_shape=input_shape
    )
    print(model.summary())
    print("\n-- COMPILING DONE --\n")


    #print("\n-- TRAINING MODEL --\n")
    model.fit(
        dataset=train_set,
        epochs=args.epochs,
        verbose=args.v,
        validation_data=val_set,
        callbacks=[
            tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
            tf.keras.callbacks.ModelCheckpoint(w,
                                                save_best_only=True,
                                                save_weights_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ])

    #print("\n-- TRAINING DONE --\n")


    #print("\n-- TESTING MODEL --\n")

    #print("\n-- TESTING DONE --\n")

    '''

if __name__ == '__main__':
    run()
