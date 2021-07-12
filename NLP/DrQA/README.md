# Reading Wikipedia to Answer Open-Domain Questions

## Contents

* [Paper](Paper.pdf)
* [Scripts](scripts/)


## Summary 

This paper considers the problem of answering factoid questions in an open-domain setting using Wikipedia as the unique knowledge source. Having a single knowledge source forces the model to be very precise while searching for an answer.

In order to answer any question, one must retrieve the relevant articles and then scan them to indentify the answer.

## Architecture

### Document Retriever

Uses an efficient (non-machine learning) document retrieval system to first narrow the search space and docus on relevant articles. A simple inverted index lookup followed by term vector model scoring is used.

Articles and questions are compared as TF-IDF (Term Frequency — Inverse Document Frequency) weighted bag-of-word vectors. It is further improved by taking local word order into account with n-gram features (BEST: bigram).

### Document Reader

Given a question q consisting og l tokens and a document of n paragraphs where a single paragraph p consists of m tokens, an RNN model is developed which is applied to each paragraph and then finally aggregated to predict the answers.

#### Paragraph Encoding

The tokens p_i in a paragraph is represented as a sequence of feature vectors P_i which is then passed as the input to the RNN: RNN(P_1, P_2, ..., P_m). A multi-layer bidirectional Long Short-term memory network and take **p_i** as the concatenation of each layer's hidden units in the end. 

The feature vector P_i is comprised of

* Word Embeddings: f_emb(p_i) = E(p_i). Using the 300-dimensional GloVe embeddings. The 1000 most frequent question words are fine tuned as some key words could be crucial to QA systems.

* Exact Match: f_exact_match(p_i) = I(p_i in q). Uses three simple binary features indicating whether p_i can be exactly matched to one of the question word in q.

* Token Features: f_token(p_i) = [POS(p_i), NER(p_i), TF(p_i)]. Manual features which reflect some properties of the token are added which include Part-of-speech (POS), Named-entity-recognition (NER) and (Normalized) Term-frequency (TF).

* Aligned Question Embedding: f_align(p_i) = sum a_ij E(q_j) where the attention score a_ij captures similarity between p_i and each question word q_j.


#### Question Encoding

Another RNN is applied on the word embeddings of q_i and the resulting hidden units are combined into one single vector (q_1, q_2, ..., q_l) -> **q**, where **q** = sum b_j q_j and b_j encodes the importance of each question word.

#### Prediction

At the paragraph level, the goal is to predict the span of tokens that is most likely the correct answer.  
Two classifiers are trained independently over the paragraph vectors (**p_1**, **p_2**, ..., **p_m**) and the question vector **q** to predict the two ends of the span. 


## Data

* Wikipedia (Knowledge Source) - Uses the 2016-12-21 dump of English Wikipedia as the knowledge source.

* SQuAD (The Stanford Question Answering Dataset) - Uses SQuAD for training and evaluating the Document Reader.





