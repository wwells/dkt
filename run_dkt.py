from dkt.DKTModel import get_DKT_model, loss_function


def run():
    # config
    print("-- APPLYING CONFIGURATION --\n")
    lstm_units = 32
    dense_units = 32

    # load data
    print("-- LOADING DATA --\n")

    print("-- TRANSFORMING DATA --\n")

    # get num_skills and input_shape from data
    input_shape = (5, 100, 5)

    # compile model
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

    #print("\n-- TRAINING DONE --\n")


    #print("\n-- TESTING MODEL --\n")

    #print("\n-- TESTING DONE --\n")


if __name__ == '__main__':
    run()
