from classifier import Classifier, datasetBuilder
from readFile import fileReader
import os, pandas, numpy,torch
from torch import nn,optim
import torch.utils.data as tud
from torchtext.data import Iterator, BucketIterator

import helper

#
#Read File
#
# path = os.getcwd().split('/')
# path.pop()
# path = '/'.join(path)+'/'
# reader  =  fileReader(path+"Datasets/tweets_hate_speech.csv",path+"Datasets/NAACL_SRW_2016.csv")
# data,target = reader.readData()
# print(path)
#
# Baseado no código da aula de pytorch de Heike Adel dada no congresso RANLP2019
#


# loading data
path = os.getcwd().split('/')
path.pop()
path = '/'.join(path)+'/'
pretrained_embeddings = torch.tensor(numpy.load(path + "Embeddings/"+"embeddings.npy"))

# setting hyperparameters
options = {
                'vocab_size': 23078,
                'emb_dim': 300,
                'num_labels': 3,
                'hidden_lstm_dim': 200,
                'bidirectional': True,
                'num_layers': 2,
                'num_epochs': 20,
                'batch_size': 100
          }

#creating dataset
dataset = datasetBuilder(path+'Datasets/',"labeled_data.csv")
# train_size = round(0.7*len(dataset))
# test_size = round(0.2*len(dataset))
# validation_size = (len(dataset)-train_size)-test_size

train_set,test_set,validation_set = dataset.splitDataset(0.7,0.2)

print("TRAIN",len(train_set))
print("TEST",len(test_set))
print("VALIDATION",len(validation_set))
# print(type(tud.random_split(dataset,[train_size,test_size,validation_size])))

print(type(validation_set))

# initializing model and defining loss function and optimizer
model = Classifier(options, pretrained_embeddings)
# model.cuda()  # do this before instantiating the optimizer
loss_function = nn.NLLLoss()
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)



# training loop
train_iter,val_iter = BucketIterator.splits(datasets=(train_set,validation_set),batch_sizes=(64,64),device = -1, 
                        sort_key =  lambda x: len(x.text),
                        sort_within_batch = False,repeat = False)
test_iter = Iterator(test_set,batch_size=64,device=-1,sort=False,sort_within_batch=False,repeat=False,sort_key=lambda x: len(x.text))

# train_loader = tud.DataLoader(train_set, batch_size=options['batch_size'], 
#                               shuffle=True)
# dev_loader = tud.DataLoader(validation_set, batch_size=options['batch_size'])
# test_loader = tud.DataLoader(test_set, batch_size=options['batch_size'])





for epoch in range(options['num_epochs']):
        print("training epoch", epoch + 1)
        # train model on training data
        for data in train_iter:
                optimizer.zero_grad()
                x, l, y = helper.sort_by_length(data['x'], data['length'], data['y'])
                scores = model(x, l)
                labels = y
                loss = loss_function(scores, labels)
                loss.backward()
                optimizer.step()

        helper.evaluate_model(train_iter, model, "train", sort=True)
        # monitor performance on dev data
        helper.evaluate_model(val_iter, model, "dev", sort=True)

helper.evaluate_model(test_set, model, "test", sort=True)