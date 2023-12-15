# Patching script

Latest version `[Dec 2023]`: [analyze_patches.py](./analyze_patches.py)

Computes patches for only the  positive sequence for a given family in given HF Dataset. If other filtering is supposed be applied (other than by `FamilyName`), modify directly the code [analyze_patches.py](./analyze_patches.py) near comment 'Dataset filtering'.

Contains the same logic as previously used separate Jupyter notebooks [generate_hf_patched_dataset.ipynb](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/Interpretability/patching/generate_hf_patched_dataset.ipynb), [m1_p40_predict.ipynb](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/Interpretability/patching/M1/m1_p40_predict.ipynb), [m1_p40_minimums.ipynb](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/Interpretability/patching/M1/m1_p40_minimums.ipynb), and [m1_p40_analysis.ipynb](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/Interpretability/patching/M1/m1_p40_analysis.ipynb).

## Usage:

Prerequisites: 

Libraries listed in [patching_env.yml](./patching_env.yml) (Exported using `conda env export -n patching_env | grep -v "^prefix: " > patching_env.yml`). 

```
$ conda env create -f patching_env.yml
```

Or alternatively here is the list of used commands:
```
$ conda create -n patching_env --yes python=3.10 ipykernel nb_conda_kernels
$ pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
$ pip3 install transformers datasets
$ pip3 install transformers['torch']
```

```
$ source activate patching_env
```

**Log in to HF using CLI (to be able to [upload HF Datasets](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html)):**

```
$ huggingface-cli login
```

## Run: 

Minimal (for `test` set with `patch_size`=10):

```
python3 analyze_patches.py -i <INPUT_HF_DATASET> -ff <FAMILY_FILTER>
```

All:

```
python3 analyze_patches.py -i <INPUT_HF_DATASET> -ff <FAMILY_FILTER> -ps <PATCH_SIZE> -s <'train'/'test'> -o <OUTPUT_DIR_PATH>
```

### Arguments:

(required are in **bold**)

| Short | Long argument | Description | Default |
|-|-|-|-|
|`-ps`|`--patch_size`|Patch size|`10`|
|`-i`|`--input_hf_dataset`|**Input HF Dataset name**|-|
|`-s`|`--set_filter`|Input HF Dataset set specifier (can be 'train' or 'test')|`test`|
|`-ff`|`--family_filter`|**Input HF Dataset family name**|-|
|`-o`|`--output_dir_path`|Output directory path|`/home/jovyan/data/patch_outputs/`|

## Other Info: 

**Inputs**: 

- HF Dataset name 

- specified Protein Family

**Outputs**: 

Artefacts that serve just for saving partial results between the main steps:

(*Originally implemented because the script was crashing during longer computations, but no crashes have occured when using the A100 GPU => saving partial results could be omitted in future versions of the script.*)

1. HF Dataset with patched versions of each sequence (name format `<USER>/patched_<'train'/'test'>_f_<FAMILY_NAME>_ps_<PATCH_SIZE>_v2023d`, where the last part stands for 'version 2023 December' == version of the script)

2. Backup of results for each patched sequence in CSV file (local only)

3. Backup of reduced results for each sequence in CSV file (local only)

The main results:

1. CSV with results for each sequence (patch minimum) + overlap information (path `<OUTPUT_DIR_PATH>/reduced_preds_overlaps_<SET_FILTER>_f_<FAMILY_FILTER>_ps_<PATCH_SIZE>_v2023d.csv`)

2. Results stats (prediction drops, overlaps) (can be displayed later again in [display_patch_results.ipynb](./display_patch_results.ipynb))
