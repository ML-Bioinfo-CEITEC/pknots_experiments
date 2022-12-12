# Fine-tuning

- We have used [DistilProtBert](https://huggingface.co/yarongef/DistilProtBert) model 
- While fine-tuning this model for knot detector, we run into problems with GPU (best run was ~ 1/2 epoch), see [DistilProtBert_finetuning.ipynb](DistilProtBert_finetuning.ipynb)
- The model was uploaded to 🤗 Hub as [simecek/knotted_proteins_demo_model](https://huggingface.co/simecek/knotted_proteins_demo_model)
- Accuracy on a test set seems to be very high (>99%)
- Further examination of GPU memory problems needed (the original repo contains fine-tuning of large datasets on small GPU)

### Update:

- The script was rerun and finished without interuption (cca 1-2 h on our server for 100.000 samples) [DistilProtBert_finetuning-M1_Eva.ipynb](DistilProtBert_finetuning-M1_Eva.ipynb)
- We trained on SPOUT x Rossmann (same high accuracy ~0.99)
- Trained also on Alphafold (version from Main doc 8.12. - clustered_dataset_09.csv, 240 255 samples) - accuracy ~0.9055, model is available on [Hugging Face](https://huggingface.co/EvaKlimentova/knots_distillprotbert_alphafold)
