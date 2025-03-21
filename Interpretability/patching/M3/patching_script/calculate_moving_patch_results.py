#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is used for getting predictions for patched versions of sequences that are given in an input CSV file. 

Expected format of columns of the input CSV file:
seq_id,sequence,global_metric_value,domain_architecture,interpro,max_knot_topology,seq_length,label,family,knot_start,knot_end,knot_len,core_percentage

For each sequence it's patched versions are generated (patch sizes can be specified by providing the --patch_sizes argument). 
Patch is simply a moving block of PATCH_CHAR repeated patch_size-times from left to right of the sequence (overlap step can be 
changed). 

Output contains the position and prediction of overall minimum prediction for each patch size for each sequence.

Usage with all arguments:
$ python3 calculate_moving_patch_results.py --patch_sizes 20 50 80 100 200 --start_index 1 --input_path <INPUT_PATH> --append --output_path <OUTPUT_PATH>

The script outputs the computation progress by specifying the number of processed and remaining sequences.
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

MODEL = '/home/jovyan/models/2023_data_v1/cnn_10epochs_94.h5'   # M3 v2
#MODEL = '/home/jovyan/models/knots_simple_CNN/cnn_10epochs.h5'  # M3 v1
CSV_DELIMITER = ','
SEQ_DIM = 500
PATCH_CHAR = 'X'

SEQ_ID = 0
SEQ = 1
SEQ_LEN = 6
FAMILY = 8
KNOT_START = 9
KNOT_END = 10
KNOT_LEN = 11

def get_model():
    return tf.keras.models.load_model(MODEL)


def get_data(data_csv):
    with open(data_csv, newline='') as f:
        reader = csv.reader(f, delimiter=CSV_DELIMITER)
        data = list(reader)
    return data


def generate_header_text(patch_sizes):
    header_text = 'id;seq;seq_len;'
    for patch_size in patch_sizes:
        header_text += f'patch_{patch_size}_preds;patch_{patch_size}_starts;patch_{patch_size}_min_start;patch_{patch_size}_min_end;patch_{patch_size}_min_pred;patch_{patch_size}_min_overlap_pred;patch_{patch_size}_min_overlap_real;patch_{patch_size}_overlap;'
    header_text += 'real_start;real_end;knot_length;family\n'
    return header_text


def fix_sequence_size(seq):
    if len(seq) < SEQ_DIM:
        seq += 'X' * (SEQ_DIM-len(seq))
    return seq


################################################################################################
######################## helper functions for calculating interval overlaps: ##################
def calculate_score_overlap_wrt_predicted(x_start, x_end, y_start, y_end):
    predicted_len = (x_end - x_start) if (x_end - x_start) != 0 else 1
    return max(0, min(x_end, y_end) - max(x_start, y_start)) / predicted_len

# what percentage of the actual knot core was found based on the prediction:
def calculate_score_overlap_wrt_real(x_start, x_end, y_start, y_end):
    real_len = (y_end - y_start) if (y_end - y_start) != 0 else 1
    return max(0, min(x_end, y_end) - max(x_start, y_start)) / real_len


# https://www.reddit.com/r/datascience/comments/vqtac5/metric_or_measure_of_how_well_two_time_intervals/
# intersection over union:
def calculate_score_overlap(x_start, x_end, y_start, y_end):
    intersection = min(x_end, y_end)-max(x_start, y_start)
    union = max(x_end, y_end) - min(x_start, y_start)
    return intersection/union if union > 0 else 0

def calculate_overlap_of_min_patch_with_core(x_start, x_end, y_start, y_end):
    wrt_pred = calculate_score_overlap_wrt_predicted(x_start, x_end, y_start, y_end)
    wrt_real = calculate_score_overlap_wrt_real(x_start, x_end, y_start, y_end)
    joined_metric = 0.0
    if wrt_pred != 0.0 and wrt_real != 0.0:
        joined_metric = calculate_score_overlap(x_start, x_end, y_start, y_end)
    return (wrt_pred, wrt_real, joined_metric)
################################################################################################
################################################################################################


################################################################################################
######################## helper functions for getting model predictions: ######################
def patch_sequence(sequence, patch_size, overlap_step, patch_char):
    interval_starts = [-1]
    patched_sequences = [sequence]
    patch = patch_char * patch_size
    last_patch_start_i = len(sequence) - patch_size + 1
    
    for i in range(0, last_patch_start_i, overlap_step):
        patched_seq = sequence[:i] + patch + sequence[i+patch_size:]
        interval_starts.append(i)
        patched_sequences.append(patched_seq)

    return interval_starts, patched_sequences


def encode_sequence(seq):
    seq_onehot = tf.one_hot([nucleo_dic[c] for c in seq], depth=21)
    return np.expand_dims(seq_onehot, axis=0)


# https://stackoverflow.com/questions/68903548/tf-data-create-a-dataset-from-a-list-of-numpy-arrays-of-different-shape
def create_generator(list_of_arrays):
    for i in list_of_arrays:
        yield i


def get_patch_predictions(model, sequence, patch_size, overlap_step, patch_char=PATCH_CHAR):
    interval_starts, patched_sequences = patch_sequence(sequence, patch_size, overlap_step, patch_char)
    encoded_sequences = [encode_sequence(_) for _ in patched_sequences]
    del(patched_sequences)
    dataset = tf.data.Dataset.from_generator(lambda: create_generator(encoded_sequences), output_types= tf.float32)
    predictions = model.predict(dataset, verbose=0)
    del(encoded_sequences)
    predictions = [_[0] for _ in predictions.tolist()]
    return pd.DataFrame({'interval_start': interval_starts,
                         'predictions': predictions})


def calculate_one_seq_results(model, sequence, real_start, real_end, patches, overlap_step=1):
    patches_len = len(patches)
    text = ''
    
    # TODO: refactor this part (concat all patched versions and predict for all at once, not for each patch_size)
    for i in range(patches_len):
        patch_size = patches[i]
        df_scores = get_patch_predictions(model, sequence, patch_size, overlap_step)

        # get index of the sequence which resulted in the lowest prediction score:
        min_i = df_scores['predictions'].idxmin()

        # if min_i == 0, the minimum prediction is the original sequence (not any of its patched versions)
        if min_i != 0:
            min_start = df_scores.iloc[min_i]["interval_start"]
            min_end = df_scores.iloc[min_i]["interval_start"] + patch_size
            overlap_pred, overlap_real, overlap = calculate_overlap_of_min_patch_with_core(min_start, min_end, real_start, real_end)
        else:
            min_start = -1
            min_end = -1
            overlap_pred, overlap_real, overlap = 0, 0, 0
            print(f'Patching with patch_size={patch_size} did not results in any drop at all.')

        text += f'{str(df_scores["predictions"].tolist())};{str(df_scores["interval_start"].tolist())};{min_start};{min_end};{df_scores.iloc[min_i]["predictions"]};{overlap_pred};{overlap_real};{overlap};'
        del(df_scores)
    return text
################################################################################################
################################################################################################


def define_arguments():
    """
        Add arguments to ArgumentParser (argparse) module instance.

        :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ps', '--patch_sizes', nargs='+', type=int, default=[10, 50, 80, 200])
    parser.add_argument('-fl', '--fix_length', help='Trigger whether input sequences are supposed to be padded to length 500.', action='store_true')
    parser.add_argument('-i', '--input_path', help='Input CSV path', type=str, required=True)
    parser.add_argument('-si', '--start_index', help='Start index of first processed sequence from CSV file', type=int, default=1)
    parser.add_argument('-a', '--append', help='Append results to the output file content (default is overwriting)', action='store_true')
    parser.add_argument('-o', '--output_path', help='Output CSV path', type=str, required=True)

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
        knot_start = int(data[i][KNOT_START])
        knot_end = int(data[i][KNOT_END])
        print(f'[{i:3}/{len(data)}] Calculating results for sequence "{sequence[:10]}..."')

        # skip sequences that are too long
        if len(sequence) > SEQ_DIM:
            print(f'Sequence (i={i}) is too long (len={len(sequence)}), skipping.\n')
            continue

        if args.fix_length:
            sequence = fix_sequence_size(sequence)
        text = f'{data[i][SEQ_ID]};{sequence};{data[i][SEQ_LEN]};'
        text += calculate_one_seq_results(model, sequence, knot_start, knot_end, patch_sizes)
        text += f'{knot_start};{knot_end};{data[i][KNOT_LEN]};{data[i][FAMILY]}'
        
        # opening for each sequence so that the intermediate results don't get lost (but for the cost of slower computation)
        output_file = open(args.output_path, 'a')
        output_file.write(text + '\n')
        output_file.close()
        
    end = datetime.now()
    print(f'===> Done at {end.strftime("%H:%M:%S")} (took {end - start})')
