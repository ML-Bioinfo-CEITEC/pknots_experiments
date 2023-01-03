# Pairs of knotted-unknotted proteins with similar sequence

To inspect quality of data and AlphaFold predictions, it would be interesting to get pairs of proteins that have

  i. similar sequences (i.e. *edit distance*)

  ii. different predictions

Also, instead of using AlphaFold predictions, we might use experimentally validated [KnotProt](https://knotprot.cent.uw.edu.pl/) proteins and look for the same or the most similar protein in AlphaFold prediction. Is it knotted or not?

## 1) Clustering

We have constructed really tight clusters (>97% similarity) and selected just 2136 clusters containing at least one knotted and unknotted proteins. Code is given in [CDHIT_tight_clusters.ipynb](CDHIT_tight_clusters.ipynb) and the pickled list of 2136 `cdhit_reader` clusters as `both_knotted_and_unknotted.pkl`.

### Ideas:

1. For each cluster, select reference sequence (`is_ref=True`) and the most similar sequence (max `identity` or min. edited distance) with opposite knotted result than the reference sequence. Prepare dataset with four columns: `unknotted_seq`, `knotted_seq`, `identity`, `edited_distance`. What is the distribution (histogram) of edited distances? What is the minimal one (or top10)?

1. For 10 - 20 most similar pairs, try to predict structure with OmegaFold or ESM fold and get knotted predictions using https://topoly.cent.uw.edu.pl/ [[demo]](https://colab.research.google.com/drive/1DeDpJX0-m923X2ucM7QAv4z7Oae0jOn2?usp=sharing). Also, check the InterPro families those pairs belong to.

