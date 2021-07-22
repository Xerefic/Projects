#!/bin/bash

mkdir datasets
cd datasets
wget https://image-net.org/data/decathlon-1.0-data-imagenet.tar
tar -xvf ./decathlon-1.0-data-imagenet.tar
rm decathlon-1.0-data-imagenet.tar