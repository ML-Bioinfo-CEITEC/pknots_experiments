# Data preprocessing

## Raw data

Raw data have been obtained from Pawel through the link in the main doc. Based on David's Uniprot ids, it collects knotted proteins (i.e. proteins present in AlphaKnot) and unknotted proteins (not present in AlphaKnot and pLDDT > 0.7). The names of the files `spout_all_knotted.csv.gz` (MD5 da068dc4eb2ef7f5799464a32d3dca85) and `spout_all_unknotted.csv.gz` (MD5 2dc831c4f180418733bcaf940239107a) are a bit misleading since it covers not only SPOUT family but other families as well.

## 1) Clutering

**Input:** `spout_all_unknotted.csv.gz`, `spout_all_knotted.csv.gz` (raw data)

**Output:** `clustered_VERSION.csv.gz` (data after length filter and clustering)

We only consider proteins with the minimal length 70 (since the shortest knotted protein is 67aa) and the maximal length 1000 (to avoid problems with transformers).

For clustering we use CD-HIT 4.8.1 (Aug 2021) and the following parameters

```
cd-hit -i all.fasta -o all_clustered -c 0.9 -s 0.8 -G 0 -aS 0.9 -T 0
```

For total `759683` sequences, we have found `357701` clusters. See [CDHIT_dataprep_v0.ipynb](CDHIT_dataprep_v0.ipynb) for the code.