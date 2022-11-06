# Knotted Proteins Experiments

So far, we have three different models, two kinds of visualizations (getting insight into the data) and two ways of interpretation.
## Models

1) [**DistilProtBert** finetuning](fine-tuning/) (**M1**): [DistilProtBert](https://huggingface.co/yarongef/DistilProtBert) model fine-tuned only 1/2 epoch on SPOUT (*SPOUT_knotted.csv*) vs. Rossmann (*Rossmann.csv*) families during the trip to Warsaw, uploaded to [🤗 Hub](https://huggingface.co/simecek/knotted_proteins_demo_model) and tested on several other families. This approach is the most promising but needs large GPU and computational resources.

2) [CNN on **ProtBertBFD** embeddings](tsne-visualization/) (**M2**): we have calculated [ProtBertBFD](https://huggingface.co/Rostlab/prot_bert_bfd) protein-level embeddings for on SPOUT and Rossmann families, got t-SNE visualizations and trained a [small CNN classifier](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/tree/main/CNN-on-embeddings) (accuracy >0.997) This approach needs to calculate embedings (=time) but after that the model is simple and small.

3) [**Simple CNN**](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/tree/main/CNN-integrated-gradients) (**M3**): we created a simple Convoluational Neural Network trained on SPOUT x Rossmann. The CNN performs similarly to the previous models but is much smaller, easier to train and should be easier to interpret. This is approach is likely to work even on your personal computer but might 

## Visualizations

1) Dimensionality reduction technique: We have calculated **t-SNE** on ProtBertBFD protein-level embeddings for on SPOUT and Rossmann families. You can see that ProtBertBFD embeddings cluster knotted vs unknotted proteins on [t-SNE plot](tsne-visualization/tsne_knots_spout.png) much better that randomized model [embeddings t-SNE](tsne-visualization/tsne_randomized_weights.png).

2) Alternativaly, we have tried **PCA plot** on the same ProtBertBFD protein-level embeddings. Again, [PCA plot](pca-visualization/) seem to distinguish knotted vs unknotted proteins (PCA1).

## Interpretation

1) **Captum**: we are working on BERT-like transformers interpretability, i.e. which part of the sequence contributed to the decision using the [Captum](https://levelup.gitconnected.com/huggingface-transformers-interpretability-with-captum-28e4ff4df234) library. We began applying **Integrated Gradients** along with other visualizations on [DNABert](https://github.com/ML-Bioinfo-CEITEC/cDNA-pretraining/tree/main/experiments/interpretability) model. Then we tried to apply to same approach on DistilProtBert, it is still [a work in progress](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/tree/main/DistilProtBert-interpretability) due to memory issues.

2) We also tried **Integrated Gradients** on M3 CNN model, you can see it [here](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/CNN-integrated-gradients/Integrated_Gradients.ipynb)

**If you want to be able to push to this repo (to add your code)**, email Petr (*simecek -at- mail.muni.cz*).
