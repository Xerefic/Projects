# Deep Learning on Point Sets for 3D Classification and Segmentation

## Contents

* [PointNet](Paper.pdf)
* [PointNet++](Paper++.pdf)


## Summary 

In this paper, a novel type of neural network is designed that directly consumes point clouds, providing a unified architecture for applications ranging from Object classification, Part segmentation to scene semantic parsing.

Typical convolutional architectures require highly regular input data formats like image grids or 3D voxels in order to perform weight sharing and other kernel optimizations. \\
Since point clouds are not in a regular format, they are typically transformed to regular 3D voxel grids or collection of images before feeding to the deep net architecture. This data representation transformation renders the resulting data unnecessarily voluminous.

PointNet is a unified architecture that directly takes point clouds as inputs and outputs either class labels for the entire input or per point segment labels for each point of the input. 


## Problem Statement

The basic architecture of the network is simple as in the initial stages, each point is processed identically and independently. The points are represented by just three coordinates (x, y, z). Additional dimensions may be added by computing normals and other local or global features.

A deep learning framework that directly consumes unordered point sets as inputs. A point cloud is represented as a set of 3D points ![points](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20P_i%20%7C%20i%3D1%2C%20%5Cdots%2C%20n%20%5Cright%20%5C%7D) where each point P is a vector of its (x, y, z) coordinate plus extra feature channels such as color, normal, etc. 

* For the **Object Classification** task, the input cloud is directly sampled from a shape or pre-segmented from a scene point cloud.

* For **Semantic Segmentation**, the input can be a single object for part region segmentation or a sub-volume from a 3D scene for object region segmentation.

## Architecture

![Layout](assets/Architecture.jpg)

The network has three key modules:
* The max pooling layer as a symmetric function to aggregate information from all the points
* A local and global information combintion structure
* A two joint alignment network that aligns both input points and features.

### Symmetry Function for Unordered Input

In order to make a model invarient to input permutation, three strategies exist
- Sort input into a canonical order
- Treat the input as a sequence to train an RNN
- Use a simple symmetric function to aggregate the information from each point.

The symmetric function takes n vectors as input and outputs a new vector that is invariant to the input order. The idea is to approximate a general function defined on a point set by applying a symmetric function on transformed elements in the set: ![function](https://latex.codecogs.com/gif.latex?f%5Cleft%20%28%20%5Cleft%20%5C%7B%20x_1%2C%20%5Cdots%2C%20x_n%20%5Cright%20%5C%7D%20%5Cright%20%29%3Dg%5Cleft%20%28%20h%28x_1%29%2C%20%5Cdots%2C%20h%28x_n%29%20%5Cright%20%29) where ![f](https://latex.codecogs.com/gif.latex?f%3A%202%5E%7B%5Cmathbb%7BR%7D%5EN%7D%20%5Crightarrow%20R), ![h](https://latex.codecogs.com/gif.latex?h%3A%20%5Cmathbb%7BR%7D%5EN%20%5Crightarrow%20%5Cmathbb%7BR%7D%5EK) and ![g](https://latex.codecogs.com/gif.latex?g%3A%20%5Cunderbrace%7B%5Cmathbb%7BR%7D%5EK%20%5Ctimes%20%5Cdots%20%5Ctimes%20%5Cmathbb%7BR%7D%5EK%7D_%5Ctext%7Bn%7D%20%5Crightarrow%20%5Cmathbb%7BR%7D).

### Local and Global Information Aggregation

The output from the above section forms a vector which is a global signature of the input set. A SVM or a Multi-Layer Perceptron classifier can be trained on the shape global features for classification. 

### Joint Alignment Network

The semantic labeling of a point cloud has to be invariant if the point cloud undergoes a certain geometrical transformation, such as rigit transformation.

A natural solution is to aligh all input set to a canonical space before feature extraction.

Input form of point clouds allows us to achieve this goal in a much simpler way. There is no need to invent any new layers and no alias is introduced as in the image case. An affine transformation matrix is predicted using a mini-network and directly apply this transformation to the coordinates of the input points. The mini-network itself resembles the big network and is composed by basic modules of point independent feature extraction.

We constrain the feature transformation matrix to be close to an orthogonal matrix: ![Lreg](https://latex.codecogs.com/gif.latex?L_%7Breg%7D%20%3D%20%5Cleft%20%5C%7C%20I-AA%5ET%20%5Cright%20%5C%7C%5E2_F) where A is the feature alignment matrix predicted by the mini-network.





## Implementation

* [PointNet Implementation](https://github.com/fxia22/pointnet.pytorch)
* [PointNet++ Implementation](https://github.com/facebookresearch/votenet)
