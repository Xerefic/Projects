#!/bin/bash

mkdir datasets
cd datasets
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ./ADEChallengeData2016.zip
rm ./ADEChallengeData2016.zip
