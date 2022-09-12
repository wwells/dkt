#
#   Deep Knowledge Tracing (DKT) Implementation
#   Mohammad M H Khajah <mohammad.khajah@colorado.edu>
#   Copyright (c) 2016 all rights reserved.
#
#   How to use:
#       python dkt.py dataset.txt dataset_split.txt
#
#   Script saves 3 files:
#       dataset.txt.model_weights trained model weights
#       dataset.txt.history training history (training LL, test AUC)
#       dataset.txt.preds predictions for test trials
#

import numpy as np
from keras.models import Sequential
from keras.layers.core import Masking
from keras.layers import LSTM, Dense
from keras import backend as K
from sklearn.metrics import roc_auc_score
import theano.tensor as Th
import random
import math
import argparse


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='Dataset file', default='assistments.txt')
    parser.add_argument('--splitfile', type=str, help='Split file', default='assistments_split.txt')
    parser.add_argument('--lstmunits', type=int, help='Number of LSTM hidden units.',
                        default=32, required=False)
    parser.add_argument('--batchsize', type=int, help='Number of sequences to process in a batch.',
                        default=5, required=False)
    parser.add_argument('--timewindow', type=int, help='Number of timesteps to process in a batch.',
                        default=100, required=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs.',
                        default=50, required=False)
    args = parser.parse_args()

    dataset = args.dataset
    split_file = args.splitfile
    lstm_units = args.lstmunits
    batch_size = args.batchsize
    time_window = args.timewindow
    epochs = args.epochs

    model_file = dataset + '.model_weights'
    history_file = dataset + '.history'
    preds_file = dataset + '.preds'

    overall_loss = [0.0]
    preds = []
    history = []

    # load dataset
    training_seqs, testing_seqs, num_skills = load_dataset(dataset, split_file)
    print(f"Training Sequences: {len(training_seqs)}")
    print(f"Testing Sequences: {len(testing_seqs)}")
    print(f"Number of skills: {num_skills}")

    # Our loss function
    # The model gives predictions for all skills so we need to get the
    # prediction for the skill at time t. We do that by taking the column-wise
    # dot product between the predictions at each time slice and a
    # one-hot encoding of the skill at time t.
    # y_true: (nsamples x nsteps x nskills+1)
    # y_pred: (nsamples x nsteps x nskills)
    def loss_function(y_true, y_pred):
        skill = y_true[:, :, 0:num_skills]
        obs = y_true[:, :, num_skills]
        rel_pred = Th.sum(float(y_pred) * float(skill), axis=2)  # converted to float

        # keras implementation does a mean on the last dimension (axis=-1) which
        # it assumes is a singleton dimension. But in our context that would
        # be wrong.
        return K.binary_crossentropy(rel_pred, obs)

    # build model
    model = Sequential()

    # ignore padding
    model.add(Masking(-1.0, batch_input_shape=(batch_size, time_window, num_skills * 2)))

    # lstm configured to keep states between batches
    model.add(LSTM(
        units=lstm_units,
        return_sequences=True
    ))
    '''
        # the original settings for LSTM
        # pinning keras, these are no longer available on the api

        input_dim=num_skills * 2,
        output_dim=hidden_units,
        return_sequences=True,
        batch_input_shape=(batch_size, time_window, num_skills * 2),
        stateful=True
    '''

    # readout layer. TimeDistributedDense uses the same weights for all
    # time steps.
    # KA python upgrade note:   pushing to Dense, based on this thread: https://stackoverflow.com/a/52092176
    model.add(Dense(
        units=32,  # NOTE: HARD CODING TO 32 during upgrade
        activation='sigmoid')
        # these are the original values from TimeDistributedDense
        # not options as params in the `Dense` api
        #
        # input_dim=hidden_units
        # output_dim=num_skills,
    )

    # optimize with rmsprop which dynamically adapts the learning
    # rate of each weight.
    model.compile(
        loss=loss_function,
        optimizer='rmsprop',
        # these are the original values
        # no longer available in updated compile api
        #
        # class_mode="binary"
    )

    # training function
    def trainer(X, Y):
        overall_loss[0] += model.train_on_batch(X, Y)[0]

    # prediction
    def predictor(X, Y):
        batch_activations = model.predict_on_batch(X)
        skill = Y[:, :, 0:num_skills]
        obs = Y[:, :, num_skills]
        y_pred = np.squeeze(np.array(batch_activations))

        rel_pred = np.sum(y_pred * skill, axis=2)

        for b in range(0, X.shape[0]):
            for t in range(0, X.shape[1]):
                if X[b, t, 0] == -1.0:
                    continue
                preds.append((rel_pred[b][t], obs[b][t]))

    # call when prediction batch is finished
    # resets LSTM state because we are done with all sequences in the batch
    def finished_prediction_batch(percent_done):
        model.reset_states()

    # similiar to the above
    def finished_batch(percent_done):
        print(f"%{percent_done} - {overall_loss[0]}")
        model.reset_states()

    # run the model
    for e in range(0, epochs):
        model.reset_states()

        # train
        run_func(training_seqs, num_skills, trainer, batch_size, time_window, finished_batch)

        model.reset_states()

        # test
        run_func(testing_seqs, num_skills, predictor, batch_size, time_window, finished_prediction_batch)

        # compute AUC
        auc = roc_auc_score([p[1] for p in preds], [p[0] for p in preds])

        # log
        history.append((overall_loss[0], auc))

        # save model
        model.save_weights(model_file, overwrite=True)
        print(f"==== Epoch: {e}, Test AUC: {auc}")

        # reset loss
        overall_loss[0] = 0.0

        # save predictions
        with open(preds_file, 'w') as f:
            f.write('was_heldout\tprob_recall\tstudent_recalled\n')
            for pred in preds:
                f.write('1\t%f\t%d\n' % (pred[0], pred[1]))

        with open(history_file, 'w') as f:
            for h in history:
                f.write('\t'.join([str(he) for he in h]))
                f.write('\n')

        # clear preds
        preds = []


def run_func(seqs, num_skills, f, batch_size, time_window, batch_done=None):

    assert(min([len(s) for s in seqs]) > 0)

    # randomize samples
    seqs = seqs[:]
    random.shuffle(seqs)

    processed = 0
    for start_from in range(0, len(seqs), batch_size):
        end_before = min(len(seqs), start_from + batch_size)
        x = []
        y = []
        for seq in seqs[start_from:end_before]:
            x_seq = []
            y_seq = []
            xt_zeros = [0 for i in range(0, num_skills * 2)]
            ct_zeros = [0 for i in range(0, num_skills + 1)]
            xt = xt_zeros[:]
            for skill, is_correct in seq:
                x_seq.append(xt)

                ct = ct_zeros[:]
                ct[skill] = 1
                ct[num_skills] = is_correct
                y_seq.append(ct)

                # one hot encoding of (last_skill, is_correct)
                pos = skill * 2 + is_correct
                xt = xt_zeros[:]
                xt[pos] = 1

            x.append(x_seq)
            y.append(y_seq)

        maxlen = max([len(s) for s in x])
        maxlen = round_to_multiple(maxlen, time_window)
        # fill up the batch if necessary
        if len(x) < batch_size:
            for e in range(0, batch_size - len(x)):
                x_seq = []
                y_seq = []
                for t in range(0, time_window):
                    x_seq.append([-1.0 for i in range(0, num_skills * 2)])
                    y_seq.append([0.0 for i in range(0, num_skills + 1)])
                x.append(x_seq)
                y.append(y_seq)

        X = pad_sequences(x, padding='post', maxlen=maxlen, dim=num_skills * 2, value=-1.0)
        Y = pad_sequences(y, padding='post', maxlen=maxlen, dim=num_skills + 1, value=-1.0)

        for t in range(0, maxlen, time_window):
            f(X[:, t:(t + time_window), :], Y[:, t:(t + time_window), :])

        processed += end_before - start_from

        # reset the states for the next batch of sequences
        if batch_done:
            batch_done((processed * 100.0) / len(seqs))


def round_to_multiple(x, base):
    return int(base * math.ceil(float(x) / base))


def load_dataset(dataset, split_file):
    seqs, num_skills = read_file(dataset)

    with open(split_file, 'r') as f:
        student_assignment = f.read().split(' ')

    training_seqs = [seqs[i] for i in range(0, len(seqs)) if student_assignment[i] == '1']
    testing_seqs = [seqs[i] for i in range(0, len(seqs)) if student_assignment[i] == '0']

    return training_seqs, testing_seqs, num_skills


def read_file(dataset_path):
    seqs_by_student = {}
    problem_ids = {}
    next_problem_id = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            student, problem, is_correct = line.strip().split(' ')
            student = int(student)
            if student not in seqs_by_student:
                seqs_by_student[student] = []
            if problem not in problem_ids:
                problem_ids[problem] = next_problem_id
                next_problem_id += 1
            seqs_by_student[student].append((problem_ids[problem], int(is_correct == '1')))

    sorted_keys = sorted(seqs_by_student.keys())
    return [seqs_by_student[k] for k in sorted_keys], next_problem_id


# https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
def pad_sequences(sequences, maxlen=None, dim=1, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


if __name__ == "__main__":
    main()
