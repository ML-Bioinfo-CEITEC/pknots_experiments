# Training simple CNN and interpreting with Integrated Gradients

## Model
- (the model was originally trained on SPOUT knotted x Rossmann unknotted dataset) 
- NEWER version is trained on [updated SPOUT x Rossmann dataset](https://huggingface.co/datasets/EvaKlimentova/knots_SPOUTxRossmann) with same knotted x unknotted sequence length distribution - you can fix length distribution in your custom dataset by undersampling using [prepare_dataset_with_same_length_distribution.py](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/CNN-integrated-gradients/prepare_dataset_with_same_length_distribution.py) script
- model architecture is [PENGUINN](https://www.frontiersin.org/articles/10.3389/fgene.2020.568546/full) with modified input
- inputs are one hot encoded sequences of length 500; longer sequences are not used, shorter sequences are padded
- model can be found [here](https://huggingface.co/EvaKlimentova/knots_simple_CNN)

## Integrated gradients
- method for CNN interpretation: https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
- you can play with [this](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/CNN-integrated-gradients/Integrated_Gradients.ipynb) script and see on what the model focuses

