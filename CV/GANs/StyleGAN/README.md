# A Style-Based Generator Architecture for Generative Adversarial Networks

## Contents

* [StyleGAN](Paper.pdf)
* [StyleGAN2](Paper++.pdf)

## Summary 

An alternative generator architecture for generative adversarial networks from style transfer literature. This architecture leads to an automatically learned, unsupervised separation of high-level attributes and stocastic variation in generated images.

Motivated by style transfer literature, the generator architecture is re-designed in a way that exposes nobel ways to control the image synthesis process. The generator starts from a learned constant input and adjusts the style of the image at each convolutional layer based on the latent code. Combined with the noise directly injected into the network, this architectural change leads to automatic unsupervised seperation of high-level attributes.

### Approach

The generator embeds the input latent code into an intermediate latent space, which has a profound effect on how the factors of variation are represented in the network. The input latent space myst follow the probability density of the training data which leads to some degree of entanglement. The intermediate latent space is free from that restriction. 

As previoused methods for estimating the degree of latent space disentanglement are not directly applicable in our case, therefore two new automated metrics
* Perceptual Path Length
* Linear Separability
for quantifying these aspects of the generator.

## Architecture

### Style-based Generator

The latent code is provided to the generator through an input layer, i.e., the first layer of a Feed Forward Network. The input layer is ommited altogether, instead a learned constant is used.

Given a latent code **z** in the input latent space, a non-linear mapping network first produces **w**, the intermediate latent space. The dimensionality of both the spaces are 512 and the mapping f is implemented using an 8-layer MLP. The learned affine transformations then specialize **w** to styles **y** that control adaptive instancw normalization operations after every convolution layer of the synthesis network g.

The generator is also provided with a direct means to generate stochastic detail by introducing explicit noise inputs. The noise image is broadcasted to all feature maps using learned perfeature scaling factors and then added to the output of the corresponding convolution.


## Implementation

* [StyleGAN Implementation](https://github.com/NVlabs/stylegan)

* [StyleGAN2 Implementation](https://github.com/NVlabs/stylegan2)