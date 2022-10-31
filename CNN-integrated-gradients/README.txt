# Training simple CNN and interpreting with Integrated Gradients

## Model
- the model is trained on SPOUT knotted x Rossmann unknotted dataset
- model architecture is [PENGUINN](https://www.frontiersin.org/articles/10.3389/fgene.2020.568546/full) with modified input
- inputs are one hot encoded sequences of length 500; longer sequences are not used, shorter sequences are padded
- model can be found [here](https://huggingface.co/EvaKlimentova/knots_simple_CNN)

## Integrated gradients
- method for CNN interpretation: https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
- you can play with [this](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments/blob/main/CNN-integrated-gradients/Integrated_Gradients.ipynb) script and see on what the model focuses

