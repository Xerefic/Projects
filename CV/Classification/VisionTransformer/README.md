# An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale

## Contents

* [Paper](Paper.pdf)


## Summary 

In Computer Vision, attention is applied in conjunction with convolutional networks or used to replace components of convolutional networks keeping their overal structure in place.

### Approach

The image is split into patches and the sequence of linear embeddings of these patches are provided as input to a Transformer. Image patches are treated the same was as tokens in NLP. 

When trained pre-trained at a large scale and transferred to smaller tasks, ViT approaches SoTA Image Recognition benchmarks.

## Architecture

The model design follows the original Transformer.

![Layout](assets/Architecture.jpg)

### Vision Transformer

The standard Transformer receives as input a 1D sequence of token embeddings. The image ![image](https://latex.codecogs.com/pdf.latex?%5Cbg_white%20x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BH%20%5Ctimes%20W%20%5Ctimes%20C%7D) 




## Implementation

* [Original Implementation](https://github.com/google-research/vision_transformer)

* [HuggingFace Implementation](https://github.com/huggingface/transformers)

* [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
