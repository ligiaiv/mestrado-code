import torch, torchtext
from torch import nn, optim
import numpy
import os
import torch.utils.data as tud
from readFile import fileReader
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset

class Classifier(nn.Module):
    def __init__(self,options, pretrained_embeddings=None):
        super(Classifier, self).__init__()
        self.options = options
        self.embedding = nn.Embedding(num_embeddings=options["vocab_size"],embedding_dim = options["emb_dim"],padding_idx=0)
        self.lstm = nn.LSTM(input_size=options['emb_dim'],
                        hidden_size=options['hidden_lstm_dim'],
                        num_layers=options['num_layers'],
                        batch_first=True,
                        bidirectional=options['bidirectional'])


        lstm_out_size = options['hidden_lstm_dim']
        if options['bidirectional']:
            lstm_out_size *= 2
            
        self.linear = nn.Linear(in_features = lstm_out_size,
                                out_features=options["num_labels"])
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,x,length):

        batch_size = x.size()[1]
        embeddings = self.embedding(x)
        # print("embeddings",embeddings.shape)
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, length, batch_first=False)

        outputs, (ht, ct) = self.lstm(embeddings)

        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        # print("outputs",outputs.shape)

        if self.options['bidirectional']:
            ht = ht.view(self.options['num_layers'], 2, batch_size, 
                        self.options['hidden_lstm_dim'])
            ht = ht[-1]  # get last (forward and backward) hidden states 
                        # from last layer
            # concatenate last hidden states from forward and backward passes
            lstm_out = torch.cat([ht[0], ht[1]], dim=1)
            # print("lstm_out",lstm_out.shape)
        else:
            lstm_out = ht[-1,:,:] # get the last hidden state of the outmost layer

        linear_out = self.linear(lstm_out)
        scores = self.softmax(linear_out)



        
        # linear_out = self.linear(LSTM)
        # scores = self.softmax(linear_out)

        return scores


class datasetBuilder():
        def __init__(self,path, filename_data, filename_labels = None):
            self.en = spacy.load('en_core_web_sm')


            self.TEXT = Field(sequential=True, 
                    tokenize=self.tokenizer, 
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower=True,
                    include_lengths = True)
            self.LABEL =Field(sequential=False, use_vocab=True)
            



            self.data = None
            self.train_set = None
            self.test_set = None
            self.validation_set = None
            self.path = path
            self.filename_data = filename_data
            self.filename_labels = filename_labels

            self.loadFile()

        def loadFile(self):

            if self.path == None:
                self.path = os.getcwd().split('/')
                self.path.pop()
                self.path = '/'.join(self.path)+'/Datasets/'
            self.data = TabularDataset.splits(path=self.path,skip_header=True,train = self.filename_data, format='csv',fields=[('n', None),('id', None),('text', self.TEXT), ('label', self.LABEL)])[0]
            
            self.preprocessData()
        def __len__(self):
                return len(self.data)

        # def __getitem__(self, idx):
        #         return {'x': self.data[idx].text, 
        #                 'length': len(self.data[idx].text), 
        #                 'y': self.data[idx].label}

        def tokenizer(self,text): # create a tokenizer function
            return [tok.text for tok in self.en.tokenizer(text)]  

        def preprocessData(self):

            # print("data0",self.data[0].text)
            self.TEXT.build_vocab(self.data, vectors="glove.6B.100d")
            self.LABEL.build_vocab(self.data)

            # print("vocab",self.LABEL.vocab.itos)

        def splitDataset(self,train_split,val_split):

            if(train_split+val_split)>=1:
                print("Cada parte deve ser menor do que 1. As somas das partes devem ser menores que 1")
                return
            train_size = round(train_split*len(self.data))
            test_size = round(val_split*len(self.data))
            validation_size = (len(self.data)-train_size)-test_size

            self.train_set,self.test_set,self.validation_set = self.data.split([train_size,test_size,validation_size])
            
            return self.train_set,self.test_set,self.validation_set
                


# def splitDataset(train_split,val_split):

#             if(train_split+val_split)>=1:
#                 print("Cada parte deve ser menor do que 1. As somas das partes devem ser menores que 1")
#                 return
#             train_size = round(train_split*len(self.data))
#             test_size = round(val_split*len(self.data))
#             validation_size = (len(self.data)-train_size)-test_size

#             train_set,test_set,validation_set = tud.random_split(self,[train_size,test_size,validation_size])
            
#             return train_set,test_set,validation_set