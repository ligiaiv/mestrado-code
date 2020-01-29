from classifier import Classifier, datasetBuilder
from readFile import fileReader
import os, pandas,torch, numpy
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

# print(type(validation_set))

#NN options

options = {
                'vocab_size': len(dataset.TEXT.vocab),
                'emb_dim': 100,
                'num_labels': len(dataset.LABEL.vocab),
                'hidden_lstm_dim': 200,
                'bidirectional': True,
                'num_layers': 2,
                'num_epochs': 20,
                'batch_size': 64
          }


#Create Iterators

train_iter,val_iter = BucketIterator.splits(datasets=(train_set,validation_set),batch_sizes=(options["batch_size"],options["batch_size"]),device = -1, 
                        sort_key =  lambda x: len(x.text),
                        sort_within_batch = False,repeat = False)
test_iter = Iterator(test_set,batch_size=options["batch_size"],device=-1,sort=False,sort_within_batch=False,repeat=False,sort_key=lambda x: len(x.text))



# initializing model and defining loss function and optimizer
model = Classifier(options, pretrained_embeddings)
# model.cuda()  # do this before instantiating the optimizer


loss_function = nn.NLLLoss()
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)


# training loop


for epoch in range(options['num_epochs']):
        print("training epoch", epoch + 1)
        # train model on training data
        for batch in train_iter:
                # print(batch.__dict__.keys())
                # print(type(batch.text))
                # print("zero:",dataset.TEXT.vocab.itos[1])
                # print(len(batch))
                # print("DIR",numpy.ma.size(batch.text,0))
                # quit()
                x,l=batch.text

                # l = numpy.ma.size(batch.text,0)
                y = batch.label 
                optimizer.zero_grad()
                x, l, y = helper.sort_by_length(x, l, y)
                scores = model(x,l)
                labels = y
                loss = loss_function(scores, labels)
                loss.backward()
                optimizer.step()

        helper.evaluate_model(train_iter, model, "train", sort=True)
        # monitor performance on dev data
        helper.evaluate_model(val_iter, model, "dev", sort=True)

helper.evaluate_model(test_set, model, "test", sort=True)