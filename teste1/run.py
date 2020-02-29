# -*- coding: utf-8 -*-

from classifier import Classifier, datasetBuilder, train_model, myConcatDataset, myDataset, mySplitDataset,BertForSequenceClassification
from readFile import fileReader
import os
import pandas
import torch
import json
import numpy as np
from torch import nn, optim
import torch.utils.data as tud
from torchtext.data import Iterator, BucketIterator
import torchtext.data as ttd
import random
import sys
from datetime import datetime, timezone

import helper

#
# Baseado no código da aula de pytorch de Heike Adel dada no congresso RANLP2019
#

# Step 1: Read data from disk

# Step 2: Tokenize text           "I like cake" -> ["<sos>","I","like","cake","<eos>"]

# Step 3: Create mapping word->idx        obs: no need to do it with BERT, already done

# Step 4: Convert text to list of ints using the mapping  ["I","like","cake"]->[2,145,2255,256,3]

# Step 5: Load data using whatever the framework requires

# Step 6: Pad the data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA is available", torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)

#
# Step 1:   loading data
#
print("Reading data file ...")
path = os.getcwd().split('/')
path.pop()
path = '/'.join(path)+'/'
# pretrained_embeddings = torch.tensor(numpy.load(path + "Embeddings/"+"embeddings.npy"))

    # setting hyperparameters
print("Reading variables file...")
options = json.load(open("variables.json", "r"))

    # creating dataset
dataset = datasetBuilder(path+'Datasets/', "labeled_data.csv",options = options)
print("DATASET LEN:", len(dataset))


# print("TEXT",dataset.data.examples[0].text)

# more NN options
if options["architecture"]=="bert":
    options["vocab_size"] = None
else:
    options["vocab_size"] = len(dataset.TEXT.vocab)

options["num_labels"] = len(dataset.LABEL.vocab)
print("\n\tOPTIONS:",options)


# initializing model and defining loss function and optimizer
# model.cuda()  # do this before instantiating the optimizer

# short sample of DATASET for developing

dataset = mySplitDataset(dataset.data,np.tile(0.05,20),rand=True)[0]
print("Short DATASET LEN:", len(dataset))

KFOLD = options["kfold"]

# splitting dataset
split_lengths = (int(len(dataset)/KFOLD))
split_lengths = np.append(np.tile(split_lengths, KFOLD-1),
                          len(dataset)-(KFOLD-1)*split_lengths).tolist()

subsets = mySplitDataset(dataset, np.tile(
    0.1, 10), rand=True, dstype="Tabular")

# inicializing arrays to save metrics
test_metrics = np.ndarray((4,0))
train_val_metrics = np.ndarray((2,options["num_epochs"],0))

#kfold loop

for index in range(KFOLD):
    print("KFOLD:", index)
    #
    #	Create objects to run
    #	obs: recreate model every time to restart weights
    #
    if options["architecture"] == "bert":
        model = BertForSequenceClassification(options)
    else:
        model = Classifier(options, subsets[0].fields['text'].vocab)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    loss_function = nn.NLLLoss()

    #
    #	Create sets
    #
    test_set = subsets[index]

    train_set = subsets[:index]+subsets[index+1:]
    train_set = myConcatDataset(train_set)
    train_set_size = int(0.9*len(train_set))
    # print("TTRAIN_CONCAT_SET:\t",train_set.examples[0].text)
    train_set, validation_set = mySplitDataset(
        train_set, [0.9, 0.1], rand=True)

    print("TRAIN_SET:", len(train_set))
    print("TEST_SET:", len(test_set))
    print("VALIDATION_SET:", len(validation_set))
    # print("TRAIN SET Sample: ",train_set.examples[0].text)

    #
    #	Create Loaders
    #
    train_loader, val_loader = BucketIterator.splits(datasets=(train_set, validation_set), batch_sizes=(options["batch_size"], options["batch_size"]), device=device,
                                                     sort_key=lambda x: len(x.text), sort=False,
                                                     sort_within_batch=False, repeat=False)
    test_loader = Iterator(test_set, batch_size=options["batch_size"], device=device,
                           sort=False, sort_within_batch=False, repeat=False, sort_key=lambda x: len(x.text))

    #
    #	Train model
    #
    acc_train, acc_val = train_model(options=options, train_iter=train_loader,
                                       val_iter=val_loader, optimizer=optimizer, model=model, loss_function=loss_function)
    train_val_metrics = np.dstack((train_val_metrics,np.array([acc_train,acc_val])))
    print("train_val_metrics",train_val_metrics.shape)
    
    #
    #	Test and save
    #
    aprf_test = helper.evaluate_model(test_loader, model, "test", options["num_labels"], architecture = options["architecture"], sort=True)
    test_metrics = np.hstack((test_metrics,np.expand_dims(np.array(aprf_test),axis = 1)))

    # end loop    

#
#	Print results to file
#

results_dict = {
    "train_val":train_val_metrics.tolist(),
    "test":test_metrics.tolist(),
    "options":options
}
now = datetime.now(timezone.utc)
current_time = now.strftime("%m-%d-%Y__%H:%M:%S")
print("Writing results to file...")
with open("results/out_"+current_time+".json", "w") as outfile:
    json.dump(results_dict, outfile)
print("Done!")
