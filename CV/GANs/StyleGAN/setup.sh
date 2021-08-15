!#/bin/bash

pip3 install pyprind

mkdir datasets
cd datasets

gdown --id 1lXxrl45Ba0G8bAvvhn1bOfUavIHlkOHT
unzip CelebAMask-HQ.zip
rm CelebAMask-HQ.zip