import tensorflow as tf
import pandas as pd
import numpy as np


def read_file(dataset_path):
    student_list = []
    exercise_list = []
    is_correct_list = []

    with open(dataset_path, 'r') as f:
        for line in f:
            student, problem, is_correct = line.strip().split(' ')
            student_list.append(student)
            exercise_list.append(problem)
            is_correct_list.append(is_correct)

    df_dict = {
        "student": student_list,
        "exercise": exercise_list,
        "is_correct": is_correct_list
    }

    # calculate the number of students and problems in the dataset
    num_students = len(set(student_list))
    num_exercises = len(set(exercise_list))

    # create a pandas df
    df = pd.DataFrame(df_dict)
    df['student'] = df['student'].astype('int')
    df['exercise'] = df['exercise'].astype('int')
    df['is_correct'] = df['is_correct'].astype('int')
    return df, num_students, num_exercises


def transform_data(df, num_students, num_exercises, batch_size=32, time_shift=True, mask_value=-1.0, shuffle=True):

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('student').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id (transforms dtype to an int64)
    df['exercise'], _ = pd.factorize(df['exercise'], sort=True)

    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['feat_exercise_with_answer'] = df['exercise'] * 2 + df['is_correct']

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep

    # this is the magic transform...
    # it takes data that was in a format like this:
    #    student  exercise  is_correct  feat_exercise_with_answer
    #      3         3           0                          6
    #      3         4           1                          9
    #      3         3           1                          7
    #
    # and if the time_shift is applied, transforms it into something like this:
    #    student
    #    3               ([6, 9], [4, 3], [1, 1])
    #
    # of particular note is the time shifting the sequence done here to move exercise
    # while keeping exercise / is_correct in place via slicing
    if time_shift:
        seq = df.groupby('student').apply(
            lambda r: (
                r['feat_exercise_with_answer'].values[:-1],
                r['exercise'].values[1:],
                r['is_correct'].values[1:],
            )
        )
    # and if the time_shift is not applied, transforms it into something like this:
    #    student
    #    3               ([6, 9, 7], [3, 4, 3], [0, 1, 1])
    #
    else:
        seq = df.groupby('student').apply(
            lambda r: (
                r['feat_exercise_with_answer'].values,
                r['exercise'].values,
                r['is_correct'].values,
            )
        )
    print(seq)
    # Step 5 - Get Tensorflow Dataset
    #
    # this is where we generate tensors for each feature
    # so for student 3 above it becomes:
    #
    # (<tf.Tensor: shape=(2,), dtype=int32, numpy=array([6, 9], dtype=int32)>,
    #  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([4, 3], dtype=int32)>,
    #  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 1.], dtype=float32)>)
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_students)

    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142
    #
    # this is where we one-hot encode existing features based on the skill_depth
    # for each exercise, we generate two encodings representing correct/not_correct
    # an example for a dataset with only three questions, we would now have
    # now we have two tensors that represent the one-hot encoded exercise
    #
    # (<tf.Tensor: shape=(2, 13), dtype=float32, numpy=
    #    array([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]], dtype=float32)>,
    #  <tf.Tensor: shape=(2, 8), dtype=float32, numpy=
    #    array([[0., 0., 0., 0., 1., 0., 0., 1.],
    #    [0., 0., 0., 1., 0., 0., 0., 1.]], dtype=float32)>)
    #
    # for more information see: https://www.tensorflow.org/versions/r2.8/api_docs/python/tf/one_hot
    features_depth = df['feat_exercise_with_answer'].max() + 1
    exercise_depth = df['exercise'].max() + 1  # making it plus two here since we factorized this in step two which made it 0 indexed

    dataset = dataset.map(
        lambda feat, exercises, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(exercises, depth=exercise_depth),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )

    # Step 7 - Pad sequences per batch
    # https://www.tensorflow.org/versions/r2.8/api_docs/python/tf/data/Dataset#padded_batch
    #
    # here we create batches, all while using the Dataset.padded_batch function
    # make the sequences uniform in size base on the dataset seen
    # since student 1 may have only tried 10 problems
    # and student 2 may have tried 50
    # the padded_shapes=([None, None], [None, None])
    # indicates that we're generating X and Y that to the "smallest size that fits"
    # https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/python/data/ops/dataset_ops.py#L1811-L1817
    #
    # in addition, any values that are padded are done so with default `mask_value`, which in this case is -1.
    #
    # at the end we have a batch to be submitted to the model that looks roughly like:
    #
    # ( # First Tensor is X
    #   tf.Tensor: shape=(n_users_in_batch, number of sequences in batch, features_depth), dtype=float32,
    #   numpy=array (...like above shape.... ))
    #   # Second Tensor is Y
    #   tf.Tensor: shape=(n_users_in_batch, number of sequences in batch, exercise_depth + (since zero indexed)), dtype=float32,
    #   nump=array (...like above shape.... ))
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(mask_value, mask_value),
        padded_shapes=([None, None], [None, None]),
    )

    # zero indexed number of batches in generator
    n_zero_batches = num_students // batch_size

    return dataset, n_zero_batches, features_depth, exercise_depth


def split_dataset(dataset, n_zero_batches, test_fraction, val_fraction=None):
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between (0, 1)")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between (0, 1)")

    test_size = np.ceil(test_fraction * n_zero_batches)
    train_size = n_zero_batches - test_size

    if test_size == 0 or train_size == 0:
        raise ValueError(
            "The train and test datasets must have at least 1 element. Reduce the split fraction or get more data.")

    train_set, test_set = split(dataset, test_size)

    val_set = None
    if val_fraction:
        val_size = np.ceil(train_size * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set
