import tensorflow as tf
import pandas as pd

import numpy as np


def load_dataset(dataset, split_file):
    seqs, num_skills, num_students = read_file(dataset)

    with open(split_file, 'r') as f:
        student_assignment = f.read().split(' ')

    training_seqs = [seqs[i] for i in range(0, len(seqs)) if student_assignment[i] == '1']
    testing_seqs = [seqs[i] for i in range(0, len(seqs)) if student_assignment[i] == '0']

    return training_seqs, testing_seqs, num_skills, num_students


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
    num_students = len(seqs_by_student)
    return [seqs_by_student[k] for k in sorted_keys], next_problem_id, num_students


def read_file_alt(dataset_path):
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

def transform_data_alt(df, num_students, num_exercises, batch_size=32, mask_value=-1.0, shuffle=True):

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
    # and transforms it into something like this:
    #    student
    #    3               ([6, 9], [4, 3], [1, 1])
    #
    # of particular note is the time shifting the sequence done here to move exercise
    # while keeping exercise / is_correct in place via slicing
    seq = df.groupby('student').apply(
        lambda r: (
            r['feat_exercise_with_answer'].values[:-1],
            r['exercise'].values[1:],
            r['is_correct'].values[1:],
        )
    )

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
    # now we have two tensors that represent the one-hot encoded exercise
    #
    # (<tf.Tensor: shape=(2, 13), dtype=float32, numpy=
    #    array([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]], dtype=float32)>,
    #  <tf.Tensor: shape=(2, 8), dtype=float32, numpy=
    #    array([[0., 0., 0., 0., 1., 0., 0., 1.],
    #    [0., 0., 0., 1., 0., 0., 0., 1.]], dtype=float32)>)
    features_depth = df['feat_exercise_with_answer'].max() + 1
    exercise_depth = df['exercise'].max() + 1

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

    length = num_students // batch_size
    return dataset, length, features_depth, exercise_depth



def transform_data(seqs, num_students, num_skills, batch_size=32, mask_value=-1.0, shuffle=True):
    # go back to lacase grande and review what format we need for the seq generator

    # experiment:   once we load correctly, can we feed in after

    dataset = tf.data.Dataset.from_tensors(
        tensors=seqs,
        output_types=(tf.int32, tf.int32, tf.float32)
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_students)


    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142
    features_depth = num_skills * 2 + 1
    skill_depth = num_skills + 1

    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skill_depth),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )
    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(mask_value, mask_value),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = num_students // batch_size

    return dataset, length, features_depth, skill_depth