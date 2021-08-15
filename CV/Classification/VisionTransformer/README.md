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

The standard Transformer receives as input a 1D sequence of token embeddings. The image ![image](https://latex.codecogs.com/gif.latex?%5Cbg_white%20x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BH%20%5Ctimes%20W%20%5Ctimes%20C%7D) is reshaped into  sequence of flattened 2D patches ![patch](https://latex.codecogs.com/gif.latex?%5Cbg_white%20x_p%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%20%5Ctimes%20%28P%5E2%20%5Ccdot%20C%29%7D) where (H,W)is the resolution of the original image, C is the number of channels, (P,P) is the resolution of each image patch and ![N](https://latex.codecogs.com/gif.latex?%5Cbg_white%20N%20%3D%20HW/P%5E2) is the resulting number of patches, the input sequence length for the transformer.

The Transformer uses a constant latent vector of size D through all of its layers. The flattened patches are mapped to D dimensions with a trainable linear projection.

A learnable embedding to the sequence of embedded patches ![embed](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctextbf%7Bz%7D%5E0_0%20%3D%20%5Ctextup%7Bx%7D_%7B%5Ctextup%7Bclass%7D%7D) is prepended whose state at the output of the Transformer encoder ![encoded](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctextbf%7Bz%7D%5E0_L) serves as the image representation y. During both pre-training and fine-tuning, a classification head is attached to ![encoded](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctextbf%7Bz%7D%5E0_L). The classification head is implemented by a MLP with one hidden layer at pre-training and by a single linear layer at fine-tuning.

Standard learnable 1D Position embeddings are added to the patch embeddings to retain positional information. The resulting sequence of embedding vectors serves as input to the encoder.

The MLP contains two layers with a GELU non-linearity.

![1](https://latex.codecogs.com/gif.latex?%5Cbg_white%20z_0%20%3D%20%5Cleft%20%5B%20x_%7B%5Ctextup%7Bclass%7D%7D%3B%20x_p%5E1%20%5Ctextbf%7BE%7D%3B%20%5Cdots%20%3B%20x_p%5EN%20%5Ctextbf%7BE%7D%20%5Cright%20%5D%20&plus;%20%5Ctextbf%7BE%7D_%7B%5Ctextup%7Bpos%7D%7D%2C%20%5Ctextbf%7B%20E%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%28P%5E2%5Ccdot%20C%29%5Ctimes%20D%7D%2C%20%5Ctextbf%7B%20E%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%28N&plus;1%29%5Ctimes%20D%7D)

![2](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bz%7D%5E%7B%5Cprime%7D_l%20%3D%20%5Ctextup%7BMSA%7D%5Cleft%20%28%20%5Ctextup%7BLN%7D%20%28%5Cmathbf%7Bz%7D_%7Bl-1%7D%29%20%5Cright%20%29%20&plus;%20%5Cmathbf%7Bz%7D_%7Bl-1%7D)

![3](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cmathbf%7Bz%7D_l%20%3D%20%5Ctextup%7BMLP%7D%5Cleft%20%28%20%5Ctextup%7BLN%7D%20%28%5Cmathbf%7Bz%7D%5E%7B%5Cprime%7D%29%5Cright%20%29&plus;%20%5Cmathbf%7Bz%7D%5E%7B%5Cprime%7D)


![4](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Ctextup%7BLN%7D%20%28%5Cmathbf%7Bz%7D_%7BL%7D%5E0%29)

##### Inductive Bias

The Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, two-dimensional neighbourhood structure and translation equivariance are built in each layer of the whole model, whereas, in ViT, only the MLP layers are local and translationally equivariant while the self-attention layers are global.

#### Fine-Tuning 

The ViT is pre-trained on large datasets and then fine-tuned to downstream tasks. The pre-trained prediction head is remobed and a zero-initialized feed-forward layer is attached. 2D interpolation are preformed on the pre-trained position embeddings. When feeding images of higher resolution, the patch size is same which results in a larger effective sequence length.


## Implementation

* [Original Implementation](https://github.com/google-research/vision_transformer)

* [HuggingFace Implementation](https://github.com/huggingface/transformers)

* [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
