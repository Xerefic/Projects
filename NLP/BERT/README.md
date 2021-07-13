# Pre-training of Deep Bidirectional Transformers forLanguage Understanding

## Contents

* [Paper](Paper.pdf)

## Summary 

A new language representation model BERT (Bidirectional Encoder Representations from Transformers). BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

A left to right architecture, where every token can only tend to previous tokens, could be very harmful when applying fine-tuning tasks, where it is crucial to incorporate context from both directions.

#### Framework

* Pre-Training - During Pre-Training, the model is trained on unlabelled data over different pre-training tasks.

* Fine-Tuning - BERT Model is first initialized with pre-trained parameters and all of the parameters are fine-tuned using labeled data from the downstream tasks. 

### Implementation

BERT's model architecture is a multi-layered bidirectional Transfoemer encoder.

#### Input/Output Representations

The input representation is able to unambiguously represent both a single sentence or a pair os sentences in one token sequence using the WordPiece embeddings.

The fitst token of every sentence is always a special classification token [CLS]. The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.

The sentences are sperated with a special token [SEP] and a learned emvedding to every token indicating the sentence it belongs to.

For a given token, its input representation is constructed by summing the corresponding token, segment and the position embeddings.

#### Pre-Training

##### Task 1 - Masked Language Models (MLM)

Standard conditional language models can only be trained either from left-to-right or right-to-left, since bidirectional conditioning would allow the model to trivially predict the target word.
Therefore, In order to train a deep bidirectional representation, some percentage of the tokens at random are masked. In the experiments 15% of the WordPiece tokens are masked at random.

Masking is a downside as it creates a mismatch between pre-trained and fine-tuning as the MASK token is not present during fine-tuning. Hence, the masked words are replaced with the actual MASK token.

##### Task 2 - Next Sequence Prediction (NSP)

A binarized next sentence prediction task that can be trivially generated from any corpus is pre-trained. When choosing the sentences A and B for each pre-training example, 50% of the time B (labeled as IsNext) is actual and 50% time it is a random sentence from the corpus (labeled NotNext).

##### Pre-Training Data

The procedure largely follows the existing literature on language model pre-training.

It is critcal that a document-level corpus is used rather than a shuffled sentence-level corpus.

#### Fine-Tuning 