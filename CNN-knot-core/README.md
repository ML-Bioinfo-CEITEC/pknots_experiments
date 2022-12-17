# CNN Patching

Patching of input sequences to the simple CNN model: https://huggingface.co/EvaKlimentova/knots_simple_CNN

Used dataset: https://huggingface.co/datasets/EvaKlimentova/knots_SPOUTxRossmann

### 1. Working with the data:

Notebooks used to transform input data to a more suitable format or to export them in wanted format:

[`get_knotprot_knot_data.ipynb`](./get_knotprot_knot_data.ipynb) - Uses [KnotProt](https://knotprot.cent.uw.edu.pl/) API to download knot core data.

[`format_spout_knot_data.ipynb`](./format_spout_knot_data.ipynb) - Joins SPOUT HF data with knot core intervals.

[`export_knot_core_data.ipynb`](./export_knot_core_data.ipynb) - Exports only patch subsequences and knot core subsequences in FASTA format (used as input to [MEME](https://meme-suite.org/meme/tools/meme)).

### 2. Moving patch:

Script [`generate_patches.py`](./generate_patches.py) takes care of generating patches for each input sequence and calculating predictions of such 
modified sequences. 

Expected input CSV format:

```
sequence,knot_start,knot_end
HMC...,10,50
ANN...,100,211
```

Output format: 

```
seq;patch_N_preds;patch_N_min_pos_start;patch_N_min_pos_end;patch_N_min_pred;real_start;real_end
```

Basic usage: 
```
$ python3 generate_patches.py --patch_sizes 10 50 80 200 --input_path <INPUT_PATH> --output_path <OUTPUT_PATH>
```


All arguments:

| Short | Long argument | Description | Default |
|-|-|-|-|
|`-ps`|`--patch_sizes`|List of patch sizes given as integers delimeted by a whitespace.|`10 50 80 200`|
|`-fl`|`--fix_length`|Use if input sequences are supposed to be padded to length 500.|-|
|`-i`|`--input_path`|Input CSV path|`/home/jovyan/data/proteins/cores_spout_test.csv`|
|`-si`|`--start_index`|Start index of first processed sequence from CSV file|`1`|
|`-a`|`--append`|Append results to the output file content (default is overwriting)|-|
|`-o`|`--output_path`|Output CSV path|`/home/jovyan/data/spout_cores_test_overlaps.csv`|

### 3. Overlaps:

Moving patches results when analysing only the overall prediction minimum:

[`knot_core_overlaps_spout_test.ipynb`](./knot_core_overlaps_spout_test.ipynb)

[`knot_core_overlaps_spout_train.ipynb`](./knot_core_overlaps_spout_train.ipynb)

[`knot_core_overlaps_knotprot.ipynb`](./knot_core_overlaps_knotprot.ipynb)

Extended version of moving patch - three ways of extending the minimum interval:

[`all_spout_knot_core_overlaps_extended.ipynb`](./all_spout_knot_core_overlaps_extended.ipynb)