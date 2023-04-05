# Knotted Proteins Experiments


## Content description

```
Data_preprocessing	# pipeline for data processing of raw AF data to training datasets
Dataset_insights	# data analysis, visualization
Interpretability	# application of standard interpretation techniques + patching
Models			# source codes for model training
Technical
```


## Models

All models were trained on [AF dataset](https://huggingface.co/datasets/EvaKlimentova/knots_AF)

| Model name | Architecture                                                                                      | Availability                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| M1         | [ProtBert-BFD](https://huggingface.co/Rostlab/prot_bert_bfd)                                      | [Hugging Face](https://huggingface.co/roa7n/knots_protbertBFD_alphafold)             | 
| M1 older   | [DistilProtBert](https://huggingface.co/yarongef/DistilProtBert)                                  | [Hugging Face](https://huggingface.co/EvaKlimentova/knots_distillprotbert_alphafold) |
| M2         | simple CNN trained on embeddings from [ProtBertBFD](https://huggingface.co/Rostlab/prot_bert_bfd) | [Hugging Face](https://huggingface.co/EvaKlimentova/knots_M2_embeddings_alphafold)   |
| M3         | [PENGUINN](https://www.frontiersin.org/articles/10.3389/fgene.2020.568546/full) trained on ohe    | [Hugging Face](https://huggingface.co/roa7n/knots_simple_CNN)                        |



**If you want to be able to push to this repo (to add your code)**, email Petr (*simecek -at- mail.muni.cz*).
