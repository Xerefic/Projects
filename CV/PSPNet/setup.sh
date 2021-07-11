#!/bin/bash

kaggle datasets download -d balraj98/stanford-background-dataset
mkdir input
cd input
unzip ../stanford-background-dataset.zip
rm ../stanford-background-dataset.zip

