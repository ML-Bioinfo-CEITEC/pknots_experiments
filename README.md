# Knotted Proteins Experiments

1) [**DistilProtBert** finetuning](fine-tuning/): [DistilProtBert](https://huggingface.co/yarongef/DistilProtBert) model fine-tuned only 1/2 epoch on SPOUT (*SPOUT_knotted.csv*) vs. Rossmann (*Rossmann.csv*) families, uploaded to [🤗 Hub](https://huggingface.co/simecek/knotted_proteins_demo_model) and tested on several other families 

2) [**ProtBertBFD** embeddings](tsne-visualization/): we have calculated [ProtBertBFD](https://huggingface.co/Rostlab/prot_bert_bfd) protein-level embeddings for on SPOUT and Rossmann families, got t-SNE visualizations and trained a [small CNN classifier](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/tree/main/CNN-on-embeddings) (accuracy >0.997)

3) **Interpretability**: we are working on BERT-like transformers interpretability, i.e. which part of the sequence contributed to the decision, it is still [a work in progress](https://github.com/ML-Bioinfo-CEITEC/cDNA-pretraining/tree/main/experiments/interpretability)

4) [**Simple CNN and Integrated Gradients**](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/tree/main/CNN-integrated-gradients): we created a simple Convoluational Neural Network trained on SPOUT x Rossmann. The CNN performs similarly to the previous models but is much smaller, easier to train and should be easier to interpret. We also tried Integrated Gradients interpretation method, you can try it [here](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/CNN-integrated-gradients/Integrated_Gradients.ipynb)

**If you want to be able to push to this repo (to add your code)**, email Petr (*simecek -at- mail.muni.cz*).
