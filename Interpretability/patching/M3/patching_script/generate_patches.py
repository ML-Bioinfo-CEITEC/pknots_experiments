#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For each sequence in the input CSV file, the predictions for it's patched versions are calculated and then saved to an output CSV file. 

Sample format of the CSV file:
 sequence,knot_start,knot_end
 HMC...,10,50
 ANN...,100,211
 ...

The patch is simply a moving block of PATCH_CHAR repeated patch_size-times from left to right of the sequence (overlap step can be 
changed). 

Output currently contains positions and predictions of overall minimum predictions for each patch size.

Basic usage:
$ python3 generate_patches.py --patch_sizes 10 50 80 200 --input_path <INPUT_PATH> --output_path <OUTPUT_PATH>

Usage of arguments if it is supposed to be appended to the output file instead of outputting predictions to a new file:
(for cases when the computations suddenly stops so that it is possibly to continue on from that exact sequence)
$ python3 generate_patches.py --patch_sizes 10 50 80 200 --start_index 42 --input_path <INPUT_PATH> --append --output_path <OUTPUT_PATH>

Usage if sequences with length < 500 are supposed to be padded to length 500:
$ python3 generate_patches.py --patch_sizes 10 50 80 200 --input_path <INPUT_PATH> --output_path <OUTPUT_PATH> --fix_length

The script displayes the computation progress by printing out the number of processed and remaining sequences.
"""

import argparse
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import csv
from os.path import exists
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, GlobalAveragePooling1D, Dense
import numpy as np
import pandas as pd


nucleo_dic = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
    'X': 20
}

MODEL = '/home/jovyan/models/knots_simple_CNN/cnn_10epochs.h5'
CSV_DELIMITER = ','
SEQ_DIM = 500
PATCH_CHAR = 'X'

SEQ = 0
KNOT_START = 1
KNOT_END = 2

def get_model():
    return tf.keras.models.load_model(MODEL)


def get_data(data_csv):
    with open(data_csv, newline='') as f:
        reader = csv.reader(f, delimiter=CSV_DELIMITER)
        data = list(reader)
    return data


def generate_header_text(patch_sizes):
    header_text = 'seq;'
    for patch_size in patch_sizes:
        header_text += f'patch_{patch_size}_preds;patch_{patch_size}_min_pos_start;patch_{patch_size}_min_pos_end;patch_{patch_size}_min_pred;'
    header_text += 'real_start;real_end\n'
    return header_text


def fix_sequence_size(seq):
    if len(seq) < SEQ_DIM:
        seq += 'X' * (SEQ_DIM-len(seq))
    return seq


def patch_sequence(sequence, patch_size, overlap_step, patch_char):
    # move patch of given size from left to right of the sequence
    interval_starts = [0]
    interval_ends = [0]
    patched_sequences = [sequence]
    patch = patch_char * patch_size
    last_patch_start_i = len(sequence) - patch_size + 1
    
    for i in range(0, last_patch_start_i, overlap_step):
        patched_seq = sequence[:i] + patch + sequence[i+patch_size:]
        interval_starts.append(i)
        interval_ends.append(i+patch_size)
        patched_sequences.append(patched_seq)

    return interval_starts, interval_ends, patched_sequences


def encode_sequence(seq):
    seq_onehot = tf.one_hot([nucleo_dic[c] for c in seq], depth=21)
    return np.expand_dims(seq_onehot, axis=0)


# https://stackoverflow.com/questions/68903548/tf-data-create-a-dataset-from-a-list-of-numpy-arrays-of-different-shape
def create_generator(list_of_arrays):
    for i in list_of_arrays:
        yield i


def get_patch_predictions(model, sequence, patch_size, overlap_step, patch_char=PATCH_CHAR):
    # generate patched versions of a given sequence and calculate all the predictions
    interval_starts, interval_ends, patched_sequences = patch_sequence(sequence, patch_size, overlap_step, patch_char)
    encoded_sequences = [encode_sequence(_) for _ in patched_sequences]
    dataset = tf.data.Dataset.from_generator(lambda: create_generator(encoded_sequences), output_types= tf.float32)
    predictions = model.predict(dataset, verbose=0)
    predictions = [_[0] for _ in predictions.tolist()]
    return pd.DataFrame({'interval_start': interval_starts, 
                         'interval_end': interval_ends, 
                         'prediction': predictions})


def calculate_one_seq_results(model, sequence, patches, overlap_step=1):
    patches_len = len(patches)
    text = f'{sequence};'
    
    for i in range(patches_len):
        patch_size = patches[i]
        df_scores = get_patch_predictions(model, sequence, patch_size, overlap_step)

        # get index of the sequence which resulted in the lowest prediction score:
        min_i = df_scores['prediction'].idxmin()
        text += f'{str(df_scores["prediction"].tolist())};{df_scores.iloc[min_i]["interval_start"]};{df_scores.iloc[min_i]["interval_end"]};{df_scores.iloc[min_i]["prediction"]};'
    return text


def define_arguments():
    """
        Add arguments to ArgumentParser (argparse) module instance.

        :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ps', '--patch_sizes', nargs='+', type=int, default=[10, 50, 80, 200])
    parser.add_argument('-fl', '--fix_length', help='Use if input sequences are supposed to be padded to length 500.', action='store_true')
    parser.add_argument('-i', '--input_path', help='Input CSV path', type=str, default='/home/jovyan/data/proteins/cores_spout_test.csv')
    parser.add_argument('-si', '--start_index', help='Start index of first processed sequence from CSV file', type=int, default=1)
    parser.add_argument('-a', '--append', help='Append results to the output file content (default is overwriting)', action='store_true')
    parser.add_argument('-o', '--output_path', help='Output CSV path', type=str, default='/home/jovyan/data/spout_cores_test_overlaps.csv')

    return parser.parse_args()


if __name__ == '__main__':
    start = datetime.now()
    print(f'===> Start at {start.strftime("%H:%M:%S")}')
    
    args = define_arguments()
    
    model = get_model()
    data = get_data(args.input_path)
    patch_sizes = args.patch_sizes
    output_path = args.output_path
    
    if args.append and exists(output_path):
        output_file = open(output_path, 'a')
        output_file.close()
    else:
        output_file = open(output_path, 'w')
        header = generate_header_text(patch_sizes)
        output_file.write(header)
        output_file.close()
        
    start_index = args.start_index
    print(f'The sequences from CSV file will be processed from index {start_index}')
    for i in range(start_index, len(data)):
        sequence = data[i][SEQ]
        print(f'[{i:3}/{len(data)}] Calculating results for sequence "{sequence[:10]}..."')

        # skip sequences that are too long
        if len(sequence) > SEQ_DIM:
            print(f'Sequence (i={i}) is too long (len={len(sequence)}), skipping.\n')
            continue

        if args.fix_length:
            sequence = fix_sequence_size(sequence)
        text = calculate_one_seq_results(model, sequence, patch_sizes)
        text += f'{data[i][KNOT_START]};{data[i][KNOT_END]}'
        
        # opening for each sequence so that the intermediate results don't get lost (but for the cost of slower computation)
        output_file = open(args.output_path, 'a')
        output_file.write(text + '\n')
        output_file.close()
        
    end = datetime.now()
    print(f'===> Done at {end.strftime("%H:%M:%S")} (took {end - start})')
