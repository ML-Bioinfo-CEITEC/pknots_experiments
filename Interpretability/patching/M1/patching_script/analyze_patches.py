#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For each sequence in the input HF Dataset (filtered by 'train'/'test' set and FamilyName), the predictions for it's patched versions are calculated and then saved to an output CSV file. Then the overall minimums are etxracted and overlaps are calculated. 

The patch is simply a moving block of PATCH_CHAR repeated patch_size-times from left to right of the sequence (overlap step can be 
changed).

Example for patch_size=5:
AAAAAAAAAAAAAAAAAAAAAAAAAA
XXXXXAAAAAAAAAAAAAAAAAAAAA
AXXXXXAAAAAAAAAAAAAAAAAAAA
AAXXXXXAAAAAAAAAAAAAAAAAAA
...
AAAAAAAAAAAAAAAAAAAAAXXXXX

Basic usage:
$ python3 analyze_patches.py -i <INPUT_HF_DATASET> -ff <FAMILY_FILTER> -ps <PATCH_SIZE> -s <'train'/'test'> -o <OUTPUT_DIR_PATH>

The script displayes computation progress by printing out partial results.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollator, Trainer, TrainingArguments
from datasets import load_metric, Features, Value
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score
from math import exp


#########################################
# TODO: set before running the script:  #
#########################################
HF_USER_NAME = 'roa7n'
#########################################

MODEL = 'EvaKlimentova/knots_AF_v4_50b' # previous: 'roa7n/knots_protbertBFD_alphafold'
OVERLAP_STEP = 1
PATCH_CHAR = 'X'


def get_data(hf_dataset_name, hf_dataset_set, family_name):
    dss = load_dataset(hf_dataset_name)
    dss = dss[hf_dataset_set]
    df = pd.DataFrame(dss)
    # Dataset filtering:
    df = df.loc[df['FamilyName'] == family_name]
    print(f'===| Loaded HF Dataset "{hf_dataset_name}" {hf_dataset_set} set with {len(df)} {family_name} sequences.')
    df = df.loc[df['label'] == 1]
    print(f'===| Positive sequences: {len(df)}')
    # df = df.sample(5) # for testing purposes
    return df


def patch_sequence(sequence_id, sequence, patch_size, overlap_step, patch_char):
    # the original (un-patched) seq is indicated by index '-1' (makes processing later easier)
    patched_sequences = [[f'{sequence_id}_{patch_size}_-1', sequence, 1]]
    patch = patch_char * patch_size
    last_patch_start_i = len(sequence) - patch_size + 1
    
    # move patch of given size from left to right of the sequence
    for i in range(0, last_patch_start_i, overlap_step):
        patched_seq = sequence[:i] + patch + sequence[i+patch_size:]
        patched_sequences.append([f'{sequence_id}_{patch_size}_{i}', patched_seq, 1])

    return patched_sequences


def patch_dataset(df, patch_size):
    new_sequences = []

    for i in range(1, len(df)):
        seq_id = df.iloc[i]['ID']
        seq_str = df.iloc[i]['uniprotSequence']
        patched_versions = patch_sequence(seq_id, seq_str, patch_size, OVERLAP_STEP, PATCH_CHAR)
        new_sequences += patched_versions

        if i % 1000 == 0:
            print(f'====| Sequence patching in progress: [{i}/{len(df)}]')
            
    print(f'===| Patched all sequences.')
    df = pd.DataFrame(new_sequences, columns=['id','sequence_str', 'label'])
    return df


def push_df_to_hf_hub(df, set_filter, family_filter, patch_size, other_specifiers=''):
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(f'{HF_USER_NAME}/patched_{set_filter}_f_{family_filter}_ps_{patch_size}_{other_specifiers}_v2023d')
    return hf_dataset


def tokenize_function(s, tokenizer):
    seq_split = ' '.join(s['sequence_str'])
    return tokenizer(seq_split)


def tokenize_hf_dataset(hf_dataset, tokenizer):
    tokenized_dataset = hf_dataset.map(lambda row: tokenize_function(row, tokenizer), remove_columns=['id', 'sequence_str'], num_proc=4)
    tokenized_dataset.set_format('pt')
    return tokenized_dataset


def get_predictions(hf_dataset, output_dir_path):
    tqdm.pandas()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tokenized_dataset = tokenize_hf_dataset(hf_dataset, tokenizer)
    
    # training_args = TrainingArguments(output_dir_path, fp16=True, per_device_eval_batch_size=50, report_to='none')  
    training_args = TrainingArguments(output_dir_path, per_device_eval_batch_size=50, report_to='none')  

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    print(f'===| Started calculating predictions for patched sequences at {start.strftime("%H:%M:%S")}.')
    predictions, _, _ = trainer.predict(tokenized_dataset)
    predictions = [np.exp(p[1]) / np.sum(np.exp(p), axis=0) for p in predictions]
    return predictions


def extract_seq_min_info(df):
    preds = list(df['m1_preds'])
    start_ids = list(df.index)
    starts = []
    for id_str in start_ids:
        starts.append(id_str.split('_')[2])  # take the start index of patch
    min_pred = min(preds)
    min_i = starts[np.argmin(preds)]
    return preds, starts, min_pred, min_i


def reduce_minimums(df_raw, df_patched_preds, patch_size):
    # Transform knot core info to more suitable format:
    # -1 because knot core info is indexed from 1 and not 0 (patch position is indexed from 0)
    df_raw['knot_start'] = df_raw['Knot Core'].apply(lambda _: int(_.replace('(', '').replace(')', '').replace(',', '').split()[0])-1 if _ else 0)
    df_raw['knot_end'] = df_raw['Knot Core'].apply(lambda _: int(_.replace('(', '').replace(')', '').replace(',', '').split()[1])-1 if _ else 0)
    
    # Calculate minimum for each sequence: 
    # (1. Filter all patched predictions of one sequence, 2. Check orig (-1) version had a knot (prediction > 0.5), 3. Take overall minimum)
    raw_ids = list(set([_.split('_')[0] for _ in df_patched_preds['id'].to_list()])) # list of IDs from original df
    df_raw = df_raw.set_index('ID')
    df_raw.index = df_raw.index.astype('str')
    df_patched_preds = df_patched_preds.set_index('id')
    df_patched_preds.index = df_patched_preds.index.astype('str')
    # Iterate over IDs in the list:
    reduced_data = []

    for i in range(len(raw_ids)):
        raw_id = raw_ids[i]
        raw_seq_info = df_raw.loc[raw_id]

        if f'{raw_id}_{patch_size}_-1' in df_patched_preds.index:
            seq_info = df_patched_preds.loc[f'{raw_id}_{patch_size}_-1']

            df_seq = df_patched_preds[df_patched_preds.index.str.contains(raw_id)].copy()
            patched_preds, patched_starts, min_pred, min_start = extract_seq_min_info(df_seq)
            del(df_seq)

            seq_dict = {'id': raw_id,
                        'sequence_str': raw_seq_info['uniprotSequence'],
                        'sequence_pred': seq_info['m1_preds'],
                        'patched_starts': patched_starts,
                        'patched_preds': patched_preds,
                        'min_start': min_start,
                        'min_pred': min_pred,
                        'knot_start': raw_seq_info['knot_start'],
                        'knot_end': raw_seq_info['knot_end'],
                        'family': raw_seq_info['FamilyName']}        
        else:
            seq_dict = {'id': raw_id,
                        'sequence_str': raw_seq_info['uniprotSequence'],
                        'sequence_pred': None,
                        'patched_starts': None,
                        'patched_preds': None,
                        'min_start': None,
                        'min_pred': None,
                        'knot_start': raw_seq_info['knot_start'],
                        'knot_end': raw_seq_info['knot_end'],
                        'family': raw_seq_info['FamilyName']} 

        if i % 500 == 0:
            print(f'====| Extracting sequence patch prediction minimums: [{i:4}/{len(raw_ids)}]') 

        reduced_data.append(seq_dict)

    print(f'===| Extracted all sequence patch prediction minimums.')
    return pd.DataFrame(reduced_data)


# https://stackoverflow.com/questions/2953967/built-in-function-for-computing-overlap-in-python
# what percentage of the predicted interval is actually in the knot core: 
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


def  calculate_overlaps(df, patch_size):
    # Take only the sequences with patched predictions:
    df = df.loc[df['patched_preds'].notnull()]
    
    # Calculate overlaps of minimum prediction with actual knot core interval:
    df['min_start'] = pd.to_numeric(df['min_start'])
    df['min_pred'] = pd.to_numeric(df['min_pred'])
    df['knot_start'] = pd.to_numeric(df['knot_start'])
    df['knot_end'] = pd.to_numeric(df['knot_end'])
    df['min_end'] = df.apply(lambda row: row['min_start'] + patch_size, axis=1)
    
    df['overlap_pred'] = df.apply(lambda row: calculate_score_overlap_wrt_predicted(row['min_start'], row['min_end'], row['knot_start'], row['knot_end']), axis=1)
    df['overlap_real'] = df.apply(lambda row: calculate_score_overlap_wrt_real(row['min_start'], row['min_end'], row['knot_start'], row['knot_end']), axis=1)
    df['overlap'] = df.apply(lambda row: calculate_score_overlap(row['min_start'], row['min_end'], row['knot_start'], row['knot_end']), axis=1)
    
    df_reduced['drop_difference'] = df_reduced.apply(lambda row: row['sequence_pred'] - row['min_pred'], axis = 1)
    
    return df


def print_overlap_stats(df, column, print_str):
    print(f'=====| {print_str}:')
    pred_mean = df[column].mean()
    print(f'====| Mean: {pred_mean}')
    pred_med = df[column].median()
    print(f'====| Med: {pred_med}')
    pred_max = df[column].max()
    print(f'====| Max: {pred_max}')
    pred_min = df[column].min()
    print(f'====| Min: {pred_min}')
    print()


def export_stats(df, output_dir_path):
    # Overall stats:
    print(f'===> Stats: ----------')
    print(f'===| # All results: {len(df)}')
    df = df.loc[df['sequence_pred'] > 0.5]
    print(f'===| # Seqs predicted as knotted: {len(df)}')
    df = df.loc[df['patched_preds'].notnull()]
    print(f'===| # Seqs with patched predictions: {len(df)}')
    print(f'===| # Seqs without drop in prediction score: {len(df_reduced.loc[df_reduced["sequence_pred"] < df_reduced["min_pred"]])}')
    print()
    
    # Overlap stats:
    print_overlap_stats(df, 'overlap_pred', 'w.r.t. predicted length (how much of the patch is inside knot core)')
    print_overlap_stats(df, 'overlap_real', 'w.r.t. real length (how much of the knot core was discovered)')
    print_overlap_stats(df, 'overlap', 'intersection over union (both metrics together)')
    
    # Drops: 
    print_overlap_stats(df, 'min_pred', 'average prediction score drop')
    df['drop_difference'] = df.apply(lambda row: row['sequence_pred'] - row['min_pred'], axis = 1)
    print_overlap_stats(df, 'drop_difference', 'average drop difference')


def define_arguments():
    """
        Add arguments to ArgumentParser (argparse) module instance.

        :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ps', '--patch_size', type=int, default=10)
    parser.add_argument('-i', '--input_hf_dataset', help='Input HF Dataset name', type=str, required=True)
    parser.add_argument('-s', '--set_filter', help='Input HF Dataset set specifier (can be "train" or "test")', choices=['test', 'train'], default='test')
    parser.add_argument('-ff', '--family_filter', help='Input HF Dataset family name', type=str, required=True)
    parser.add_argument('-o', '--output_dir_path', help='Output directory path', type=str, default='/home/jovyan/data/patch_outputs/')

    return parser.parse_args()


if __name__ == '__main__':
    start = datetime.now()
    print(f'===> Start at {start.strftime("%H:%M:%S")} ----------')
    
    args = define_arguments()
    patch_size = args.patch_size
    input_hf_dataset = args.input_hf_dataset
    set_filter = args.set_filter
    family_filter = args.family_filter
    output_dir_path = args.output_dir_path
    
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    
    # 1. Generate patched versions of all train/ test ('set_filter') sequence for given family ('family_filter'):
    df_orig = get_data(input_hf_dataset, set_filter, family_filter)
    df_patched = patch_dataset(df_orig, patch_size)
    hf_dss_patched = push_df_to_hf_hub(df_patched, set_filter, family_filter, patch_size)
    
    # 2. Get predictions for patched sequences:
    df_patched['m1_preds'] = get_predictions(hf_dss_patched, output_dir_path)
    df_patched.to_csv(output_dir_path + f'preds_{set_filter}_f_{family_filter}_ps_{patch_size}_v2023d.csv', encoding='utf-8', index=False)
    df_patched['m1_preds'] = df_patched['m1_preds'].astype(np.float32)
    print(f'===| Finished calculating predictions for patched sequences at {start.strftime("%H:%M:%S")}.')
    push_df_to_hf_hub(df_patched, set_filter, family_filter, patch_size, 'preds')
    
    # 3. Reduce results of format (patched_seq: pred) to (orig_seq: all_preds, min_pred):
    df_reduced = reduce_minimums(df_orig, df_patched, patch_size)
    df_reduced.to_csv(output_dir_path + f'reduced_preds_{set_filter}_f_{family_filter}_ps_{patch_size}_v2023d.csv', sep=';', encoding='utf-8', index=False)
    
    # 4. Calculate overlaps:
    df_overlaps = calculate_overlaps(df_reduced, patch_size)
    df_overlaps.to_csv(output_dir_path + f'reduced_preds_overlaps_{set_filter}_f_{family_filter}_ps_{patch_size}_v2023d.csv', sep=';', encoding='utf-8', index=False)
    
    # 5. Calculate stats:
    export_stats(df_overlaps, output_dir_path)
    
    end = datetime.now()
    print(f'===> Done at {end.strftime("%H:%M:%S")} (took {end - start}) ----------')
