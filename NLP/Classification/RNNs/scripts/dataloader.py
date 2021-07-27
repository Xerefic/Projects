import torch
import os
import spacy
import nltk
import torchtext

class CreateDataset(torch.utils.data.Dataset):

    def __init__(self, PATH, batch_size=32):
        self.PATH = PATH
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.spacy = spacy.load("en_core_web_sm")

        self.TEXT = torchtext.legacy.data.Field(sequential=True, tokenize="spacy")
        self.LABEL = torchtext.legacy.data.LabelField(dtype=torch.long, sequential=False)

        self.initData()
        self.initEmbed()

        self.makeData()

    def initData(self):
        DATA = os.path.join(self.PATH, 'inputs/')

        self.train_data, self.valid_data, self.test_data = torchtext.legacy.data.TabularDataset.splits(
                        path=DATA, 
                        train="train.csv", validation="valid.csv", test="test.csv", 
                        format="csv", 
                        skip_header=True, 
                        fields=[('Text', self.TEXT), ('Label', self.LABEL)])

    def initEmbed(self):
        EMBED = os.path.join(self.PATH, "embeddings/glove.840B.300d/glove.840B.300d.txt")

        self.TEXT.build_vocab(self.train_data,
                         vectors=torchtext.vocab.Vectors(EMBED), 
                         max_size=20000, 
                         min_freq=10)
        self.LABEL.build_vocab(self.train_data)

    def makeData(self):
        self.train_iterator, self.valid_iterator, self.test_iterator = torchtext.legacy.data.BucketIterator.splits(
                        (self.train_data, self.valid_data, self.test_data), 
                        sort_key=lambda x: len(x.Text), 
                        batch_size=self.batch_size,
                        device=self.device)

    def lengthData(self):
        return len(self.train_data), len(self.valid_data), len(self.test_data)
    
    def lengthVocab(self):
        return len(self.TEXT.vocab), len(self.LABEL.vocab)

    def freqLABEL(self):
        return self.LABEL.vocab.freqs

    def getData(self):
        return self.train_iterator, self.valid_iterator, self.test_iterator

    def getEmbeddings(self):
        return self.TEXT.vocab.vectors