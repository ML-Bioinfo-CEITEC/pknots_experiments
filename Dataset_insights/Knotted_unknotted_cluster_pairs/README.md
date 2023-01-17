# Pairs of knotted-unknotted proteins with similar sequence

To inspect quality of data and AlphaFold predictions, we obtained pairs of proteins that have similar sequences (i.e. *edit distance*) and different predictions.


## 1) Clustering

We have constructed really tight clusters (>97% similarity) and selected just 2136 clusters containing at least one knotted and unknotted proteins. Code is given in [CDHIT_tight_clusters.ipynb](CDHIT_tight_clusters.ipynb).

## 2) Analysis

1. For each cluster we selected reference sequence (`is_ref=True`) and the most similar sequence (max `identity` or min. edited distance) with opposite knotted result than the reference sequence. 

2. For 10 - 20 most similar pairs we predicted structure with OmegaFold and got knotted predictions using https://topoly.cent.uw.edu.pl/ [[demo]](https://colab.research.google.com/drive/1DeDpJX0-m923X2ucM7QAv4z7Oae0jOn2?usp=sharing).

