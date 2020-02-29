import torch
import torchtext
from torch import nn, optim
import numpy as np
import os
import random
import torch.utils.data as tud
from readFile import fileReader
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset, Dataset
import helper
import tqdm
from transformers import BertTokenizer, BertModel
import copy



class Classifier(nn.Module):
    def __init__(self, options, pretrained_embedding=None):
        super(Classifier, self).__init__()
        self.options = options

        self.embedding = nn.Embedding(num_embeddings=options["vocab_size"],
                                      embedding_dim=options["emb_dim"],
                                      padding_idx=0)

        if pretrained_embedding is not None:
            print("\n\t-Using pre-trained embedding")
            self.embedding.from_pretrained(
                pretrained_embedding.vectors, freeze=options["freeze_emb"])
        # self.embedding.load_state_dict()
        self.lstm = nn.LSTM(input_size=options['emb_dim'],
                            hidden_size=options['hidden_lstm_dim'],
                            num_layers=options['num_layers'],
                            batch_first=True,
                            bidirectional=options['bidirectional'])

        lstm_out_size = options['hidden_lstm_dim']
        if options['bidirectional']:
            lstm_out_size *= 2

        self.linear = nn.Linear(in_features=lstm_out_size,
                                out_features=options["num_labels"])
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, length):

        batch_size = x.size()[1]
        # print("x size in forward:",x.size())
        # print("x in",x)
        
        embeddings = self.embedding(x)
        embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, length, batch_first=False)

        outputs, (ht, ct) = self.lstm(embeddings)

        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=False)

        if self.options['bidirectional']:
            ht = ht.view(self.options['num_layers'], 2, batch_size,
                         self.options['hidden_lstm_dim'])
            ht = ht[-1]  # get last (forward and backward) hidden states
            # from last layer
            # concatenate last hidden states from forward and backward passes
            lstm_out = torch.cat([ht[0], ht[1]], dim=1)
        else:
            # get the last hidden state of the outmost layer
            lstm_out = ht[-1, :, :]

        linear_out = self.linear(lstm_out)
        scores = self.softmax(linear_out)

        # linear_out = self.linear(LSTM)
        # scores = self.softmax(linear_out)

        return scores


class datasetBuilder():
    def __init__(self, path, filename_data,options, filename_labels=None):

        # load tokenizer modules
        self.options =options
        self.en = spacy.load('en_core_web_sm')
        self.bertTokenizer=BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
        

        if options["architecture"] == "bert":
            # tokenizer = self.bertTokenizer.tokenize
            #tokenization, mapping (already done), and numericalization at once
            tokenizer = self.bertTokenizer.encode
            pad_index = self.bertTokenizer.convert_tokens_to_ids(self.bertTokenizer.pad_token)
            print("pad_index",pad_index)
            eos_index = self.bertTokenizer.convert_tokens_to_ids(self.bertTokenizer.eos_token)
            bos_index = self.bertTokenizer.convert_tokens_to_ids(self.bertTokenizer.bos_token)

            print("Bert Tokenizer")
        else:
            tokenizer = self.spacy_tokenizer
            pad_index = "<pad>"
            bos_index = "<sos>"
            eos_index = "<eos>"
        
        self.TEXT = Field(sequential=True,
                          tokenize=tokenizer,
                          init_token=bos_index,
                          eos_token=eos_index,
                          lower=not(options["architecture"]=='bert'),
                          include_lengths=True,
                          fix_length=options["seq_len"],
                          use_vocab=not(options["architecture"]=='bert'),
                          pad_token=pad_index)
        self.LABEL = Field(sequential=False, use_vocab=True)

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
        self.data = TabularDataset.splits(path=self.path, skip_header=True, train=self.filename_data, format='csv', fields=[
                                          ('n', None), ('id', None), ('text', self.TEXT), ('label', self.LABEL)])[0]
        # print(self.data.examples[0].text)
        self.preprocessData()

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #         return {'x': self.data[idx].text,
    #                 'length': len(self.data[idx].text),
    #                 'y': self.data[idx].label}

    def spacy_tokenizer(self, text):  # create a tokenizer function
        # x = [tok.text for tok in self.en.tokenizer(text)]
        # print("X\t",x)
        return [tok.text for tok in self.en.tokenizer(text)]
        

    def preprocessData(self):

        if not(self.options["architecture"] == 'bert'):
            self.TEXT.build_vocab(self.data, vectors="glove.6B.100d")
        # tokenized_text = self.bertTokenizer.tokenize(some_text)

        # self.bertTokenizer.convert_tokens_to_ids(tokenized_text)

        self.LABEL.build_vocab(self.data)

        print(self.data.examples[0].text)

    def randomShuffle(self):
        self.data = np.random.shuffle(self.data)


class myDataset(Dataset):
    def __init__(self, data, fields):
        self.examples = data
        self.fields = fields

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getitem__(self, idx):
        return self.examples[idx]

# training loop


def train_model(options, train_iter, val_iter, optimizer, model, loss_function):
    accs_val = []
    accs_train = []
    for epoch in range(options['num_epochs']):
        print("training epoch", epoch + 1)

        # train model on training data
        for batch in tqdm.tqdm(train_iter):
            # for batch in train_iter:
            x, l = batch.text

            # l = numpy.ma.size(batch.text,0)
            y = batch.label
            optimizer.zero_grad()
            x, l, y = helper.sort_by_length(x, l, y)
            if (options["architecture"]=="bert"):
                scores = model(x)
            else:
                scores = model(x, l)
            labels = y
            loss = loss_function(scores, labels)
            loss.backward()
            optimizer.step()

        acc_train,_,_,_ = helper.evaluate_model(train_iter, model, "train", options["num_labels"], sort=True)
        accs_train.append(acc_train)
        acc_val,_,_,_ = helper.evaluate_model(val_iter, model,"val", options["num_labels"], sort=True)
        accs_val.append(acc_val)

    return (accs_train, accs_val)
    # monitor performance on dev data


class myConcatDataset(Dataset):

    def __init__(self, datasets, fields=None):
        self.examples = None
        self.fields=None
    
        self.indices = []
        if hasattr(datasets[0], 'fields'):
            self.fields = datasets[0].fields
        else:
            self.fields = fields
        self.datasets = datasets
        self.examples = self.concat_datasets(self.datasets)
        self.i = 0

    def concat_datasets(self, datasets):
        total_dataset = []
        for subset in datasets:
            total_dataset += subset.examples
        return total_dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        # return {'text': self.dataset[idx].text,
        # 		'length': len(self.dataset[idx].text),
        # 		'label': self.dataset[idx].label
        # 		}
        return self.examples[idx]


def mySplitDataset(dataset, ratios, rand=False, dstype=None):
    if np.sum(ratios) > 1:
        print("ERROR: ratios must sum 1 or less")
    elif np.sum(ratios) < 1:
        ratios.append(0.1)
    total = len(dataset)
    sizes = []
    sizes = np.round(np.array(ratios)*total).astype(np.int)
    sizes[-1] = total - np.sum(sizes[0:-1])

    start_point = 0
    pieces = []
    if rand:
        if dstype == "Tabular":
            data = random.sample(dataset.examples, len(dataset))
        else:
            data = random.sample(dataset.examples, len(dataset))
        # dataset = ttd.Dataset(data,dataset.fields)
        # for size in pieces:

    for size in sizes:
        pieces.append(
            myDataset(data[start_point:start_point+size], dataset.fields))
        start_point += size

    return pieces


class BertForSequenceClassification(nn.Module):
  
    def __init__(self, options):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = options["num_labels"]
        self.bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = False)
        self.dropout = nn.Dropout(options["dropout"])
        self.classifier = nn.Linear(options["hidden_size"], options["num_labels"])
        nn.init.xavier_normal_(self.classifier.weight)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        #changed from 'output_all_encoded_layers' to 'output_hidden_states' and put in the configuration
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

def train_bert_model(model, criterion, optimizer, scheduler, num_epochs=25):
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            sentiment_corrects = 0
            
            
            # Iterate over data.
            for inputs, sentiment in dataloaders_dict[phase]:
                #inputs = inputs
                #print(len(inputs),type(inputs),inputs)
                #inputs = torch.from_numpy(np.array(inputs)).to(device) 
                inputs = inputs.to(device) 

                sentiment = sentiment.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model(inputs)

                    outputs = F.softmax(outputs,dim=1)
                    
                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])

                
            epoch_loss = running_loss / dataset_sizes[phase]

            
            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} sentiment_acc: {:.4f}'.format(
                phase, sentiment_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test.pth')


        print()

    print('Best val Acc: {:4f}'.format(float(best_loss)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model