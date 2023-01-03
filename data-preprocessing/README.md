# Data preprocessing

## Raw data

Raw data have been obtained from Pawel through the link in the main doc. Based on David's Uniprot ids, it collects knotted proteins (i.e. proteins present in AlphaKnot) and unknotted proteins (not present in AlphaKnot and pLDDT > 0.7). The names of the files `spout_all_knotted.csv.gz` (MD5 da068dc4eb2ef7f5799464a32d3dca85) and `spout_all_unknotted.csv.gz` (MD5 2dc831c4f180418733bcaf940239107a) are a bit misleading since it covers not only SPOUT family but other families as well.

## 1) Clustering

**Input:** `spout_all_unknotted.csv.gz`, `spout_all_knotted.csv.gz` (raw data)

**Output:** `clustered_VERSION.csv.gz` (data after length filter and clustering)

We only consider proteins with the minimal length 70 (since the shortest knotted protein is 67aa) and the maximal length 1000 (to avoid problems with transformers).

For clustering we use CD-HIT 4.8.1 (Aug 2021) and the following parameters

```
cd-hit -i all.fasta -o all_clustered -c 0.9 -s 0.8 -G 0 -aS 0.9 -T 0
```

For total `759683` sequences, we have found `357701` clusters. See [CDHIT_dataprep_v0.ipynb](CDHIT_dataprep_v0.ipynb) for the code.

## 2) Balancing knotted and unknotted sequence lengths

**Input:** `clustered_VERSION.csv.gz` (data after clustering)

**Output:** `length_normalized_VERSION.csv.gz` (length-normalized)

We want to have similar length distribution for knotted and unknotted proteins. [normalize_length_distribution.ipynb](normalize_length_distribution.ipynb) will compare the distributions of knotted and unknotted and trash all the unsatisfactory proteins.

We are now doing the length comparison not strictly but in bins of size 5.

The script also adds a 'label' column for easier later recognition of knotted and unknotted proteins.

Applied to the previous clustered dataset, we have 205 994 sequences in total.



## 3) Adding IPRO and PFAM names

**Input** `length_normalized_VERSION.csv.gz` (length-normalized)
**Output** `families_added_VERSION.csv.gz` (with added IPRO and PFAM)

Part of the given dataset may not contain information about IPRO and PFAM family, so use [add_ipro_and_pfam_family.ipynb](add_ipro_and_pfam_family.ipynb) to add it. It uses spyprot and searches Uniprot with sequence IDs and extracts information about their PFAM and IPRO family. To be consistent, it recomputes the families for all proteins. 

The ID column is modified to contain the pure Uniprot ID ('AF-' prefix and '-F1' suffix is deleted)


## 4) Splitting to train and test dataset

**Input** `families_added_VERSION.csv.gz` (with families)
**Output** dataset uploaded to Hugging Face

Use [split_dataset_train_test.ipynb](split_dataset_train_test.ipynb) to split the processed dataset into train and test set ready for model training and testing. The split is done randomly so far. The prepared dataset is then uploaded to Hugging Face.

The current processed version of the dataset can be found at [Hugging Face](https://huggingface.co/datasets/EvaKlimentova/knots_AF)
