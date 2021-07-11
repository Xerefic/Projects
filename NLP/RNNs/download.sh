#!/bin/bash

kaggle competitions download -c quora-insincere-questions-classification

unzip quora-insincere-questions-classification.zip

unzip embeddings.zip -d embeddings/

rm -r embeddings/GoogleNews-vectors-negative300/
rm -r embeddings/paragram_300_sl999/
rm -r embeddings/wiki-news-300d-1M/
rm embeddings.zip
rm quora-insincere-questions-classification.zip
