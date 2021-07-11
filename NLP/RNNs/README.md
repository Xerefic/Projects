# Quora Insincere Questions Classification

The Dataset along with the embeddings can be downloaded from the [Kaggle Competition Page](https://www.kaggle.com/c/quora-insincere-questions-classification/data) </br>
* The [Dataloader](dataloader.py) is a wrapper class capable of converting the text input into embeddings using pre-trained Glove embeddings.
* The [Model](model.py) has LSTM as well as RNN modules defined using in-built definitions of pytorch.
* The [Notebook](LSTM.ipynb) calls the wrapper classes to train the model.

To Download and set-up the environment automatically,</br>
* Clone the repository.
* Run the following commands.
```
  cd NLP/Quora/
  chmod 755 download.sh
  ./download.sh
```
NOTE: Must have Kaggle API setup already.
