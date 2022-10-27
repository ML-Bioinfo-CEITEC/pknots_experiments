# Knotted Proteins Experiments

1) [**DistilBERT** finetuning](fine-tuning/): the original model trained only 1/2 epoch on SPOUT (*SPOUT_knotted.csv*) vs. Rossmann (*Rossmann.csv*) families, uploaded to 🤗 Hub and tested on several other families 

2) [**ProtBertBFD** embeddings](tsne-visualization/): we have calculated protein-level embeddings for on SPOUT and Rossmann families, got t-SNE visualizations and trained a small CNN classifier (accuracy >0.997)

3) **Interpretability**: we are working on BERT-like transformers interpretability, i.e. which part of the sequence contributed to the decision, it is still [a work in progress](https://github.com/ML-Bioinfo-CEITEC/cDNA-pretraining/tree/main/experiments/interpretability)
