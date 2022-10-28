# Fine-tuning

- We have used [DistilProtBert](https://huggingface.co/yarongef/DistilProtBert) model 
- While fine-tuning this model for knot detector, we run into problems with GPU (best run was ~ 1/2 epoch), see [DistilProtBert_finetuning.ipynb](DistilProtBert_finetuning.ipynb)
- The model was uploaded to 🤗 Hub as [simecek/knotted_proteins_demo_model](https://huggingface.co/simecek/knotted_proteins_demo_model)
- Accuracy on a test set seems to be very high (>99%)
- Further examination of GPU memory problems needed (the original repo contains fine-tuning of large datasets on small GPU)