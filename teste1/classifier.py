import torch
from torch import nn, optim
import numpy
import torch.utils.data as tud
from readFile import fileReader


class Classifier(nn.Module):
    def __init__(self,options, pretrained_embeddings=None):
        self.embedding = nn.Embedding(num_embeddings=options["vocab_size"],
                                        embedding_dim = options["emb_dim"],
                                        padding_idx=0)
        self.lstm = nn.LSTM(input_size=options['emb_dim'],
                        hidden_size=options['hidden_lstm_dim'],
                        num_layers=options['num_layers'],
                        batch_first=True,
                        bidirectional=options['bidirectional'])


        lstm_out_size = options['hidden_lstm_dim']
        if options['bidirectional']:
            lstm_out_size *= 2
            
        self.linear = nn.Linear(in_features = options["vocab_size"],
                                out_features=options["num_labels"])
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,x,length):

        batch_size = x.size()[0]
        embeddings = self.embedding(x)
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, length, batch_first=True)
        outputs, (ht, ct) = self.lstm(embeddings)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if self.options['bidirectional']:
            ht = ht.view(self.options['num_layers'], 2, batch_size, 
                        self.options['hidden_lstm_dim'])
            ht = ht[-1]  # get last (forward and backward) hidden states 
                        # from last layer
            # concatenate last hidden states from forward and backward passes
            lstm_out = torch.cat([ht[0], ht[1]], dim=1)
        else:
            lstm_out = ht[-1,:,:] # get the last hidden state of the outmost layer

        linear_out = self.linear(lstm_out)
        scores = self.softmax(linear_out)



        
        # linear_out = self.linear(LSTM)
        # scores = self.softmax(linear_out)

        return scores


class thisDataset(tud.Dataset):
        def __init__(self,path, filename_data, filename_labels):

            self.data = None
            self.target = None
            self.path = path
            self.filename_data = filename_data
            self.filename_labels = filename_labels

            self.loadFile()
                # self.data_np = numpy.load(filename_data)
                # self.length_np = numpy.load(filename_length)
                # self.labels_np = numpy.load(filename_labels)
                # self.data = torch.tensor(self.data_np).long()
                # self.length = torch.tensor(self.length_np).long()
                # self.labels = torch.tensor(self.labels_np).long()
        def loadFile(self):

            if self.path == None:
                self.path = os.getcwd().split('/')
                self.path.pop()
                self.path = '/'.join(self.path)+'/Datasets/'
            reader  =  fileReader(self.path+self.filename_data,self.path+self.filename_labels)
            self.data,self.target = reader.readData()
            print(self.path)
        def __len__(self):
                return len(self.data)

        def __getitem__(self, idx):
                return {'x': self.data[idx], 
                        'length': len(self.data[idx]), 
                        'y': self.labels[idx]}