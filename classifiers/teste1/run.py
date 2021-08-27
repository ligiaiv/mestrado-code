# -*- coding: utf-8 -*-

from classifier import  LSTM_from_CNN, CNN_LSTMClassifier,LSTMClassifier,CNN1DClassifier,LSTM_CNNClassifier, datasetBuilder, train_model, myConcatDataset, myDataset, mySplitDataset,BertForSequenceClassification
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
# import matplotlib.pyplot as plt


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
path = os.getcwd()+"/"
# path = os.getcwd().split('/')
# path.pop()
# path = '/'.join(path)+'/'
# pretrained_embeddings = torch.tensor(numpy.load(path + "Embeddings/"+"embeddings.npy"))

    # setting hyperparameters
print("Reading variables file...")
options = json.load(open("teste1/variables.json", "r"))

    # creating dataset
dataset = datasetBuilder(path+'Datasets/', "labeled_data.csv",options = options)
print("DATASET LEN:", len(dataset))

# more NN options
if options["architecture"]=="bert":
    options["vocab_size"] = None
else:
    options["vocab_size"] = len(dataset.TEXT.vocab)

options["num_labels"] = len(dataset.LABEL.vocab)
print(options["num_labels"])
print("\n\tOPTIONS:",options)


# initializing model and defining loss function and optimizer
# model.cuda()  # do this before instantiating the optimizer

# short sample of DATASET for developing

# dataset = mySplitDataset(dataset.data,np.tile(0.05,20),rand=True)[0]
# print("Short DATASET LEN:", len(dataset))
if options["use_kfold"]:
    KFOLD = options["kfold"]
    # KFOLD = 2
    # splitting dataset

    subsets = mySplitDataset(dataset.data, np.tile(
        1/KFOLD, KFOLD), rand=True, dstype="Tabular")

    # subsets = subsets[:10]
    # inicializing arrays to save metrics
    test_metrics = np.ndarray((4,0))
    train_val_metrics = np.ndarray((2,options["num_epochs"],0))



#kfold loop
def prepare_train(options,dataset,index = None,split_ratio = [0.7,0.15,0.15]):
    
    
    if isinstance(dataset,list):
        vocab = dataset[0].fields['text'].vocab
    else:
        vocab = dataset.fields['text'].vocab

    if options["architecture"] == "bert":
        model = BertForSequenceClassification(options)
    elif options["architecture"]=="lstm":
        model = LSTMClassifier(options, vocab)
    elif options["architecture"]=="lstm-cl":
        model = LSTMClassifier(options)
    elif options["architecture"] == "cnn":
        model = CNN1DClassifier(options, vocab)
    elif options["architecture"] == "cnn-cl":
        model = CNN1DClassifier(options)
    elif options["architecture"] == "lstm-cnn":
        model = LSTM_CNNClassifier(options)
    elif options["architecture"] == "cnn-lstm":
        model = CNN_LSTMClassifier(options)
    # elif options["architecture"] == "cnn-lstm":
    #     model = LSTM_from_CNN(options)
    else:
        print("ERROR: architecture provided"+options["architecture"]+"does not match any option")
        quit()

    if options["use_kfold"]:
        test_set = subsets[index]

        train_set = subsets[:index]+subsets[index+1:]
        train_set = myConcatDataset(train_set)
        train_set_size = int(0.9*len(train_set))
        # print("TTRAIN_CONCAT_SET:\t",train_set.examples[0].text)
        train_set, validation_set = mySplitDataset(
            train_set, [0.9, 0.1], rand=True)
    else:
        train_set,test_set, validation_set = mySplitDataset(
            dataset, split_ratio, rand=True)

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
    return (model,train_loader,val_loader,test_loader)

def lstmfcnn(options,dataset):
    pesos = torch.load("weights.pth")
    print(pesos)
    #
    #	Test and save
    #
    # acc,prec,rec,f1,confmat = helper.evaluate_model(test_loader, model, "test", options["num_labels"], architecture = options["architecture"], sort=True)
    # test_metrics = np.array([acc,prec,rec,f1])
    # cnn_output = model.cnn_output

    
    options["architecture"] = "cnn-lstm"
    model, train_loader, val_loader, test_loader = prepare_train(options,dataset.data)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    loss_function = nn.NLLLoss()

    acc_train, acc_val = train_model(options=options, 
                                    train_iter=train_loader,
                                    val_iter=val_loader, 
                                    optimizer=optimizer, 
                                    model=model, 
                                    loss_function=loss_function,
                                    weights=pesos)
    train_val_metrics = np.array([acc_train,acc_val])
    
weights=None

if options["read_weights"]:
    weights = torch.load("weights.pth")

if options["use_kfold"]:
    for index in range(KFOLD):
        print("KFOLD:", index)
        model, train_loader, val_loader, test_loader = prepare_train(options,subsets,index)
        optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
        # loss_function = nn.NLLLoss()
        loss_function = nn.CrossEntropyLoss()
        acc_train, acc_val = train_model(options=options, train_iter=train_loader,
                                        val_iter=val_loader, optimizer=optimizer, model=model,
                                        loss_function=loss_function,
                                        weights=weights)
        train_val_metrics = np.dstack((train_val_metrics,np.array([acc_train,acc_val])))
        print("train_val_metrics",train_val_metrics.shape)
        
        #
        #	Test and save
        #
        acc,prec,rec,f1,confmat = helper.evaluate_model(test_loader, model, "test", options["num_labels"], architecture = options["architecture"], sort=True)
        test_metrics = np.hstack((test_metrics,np.expand_dims(np.array([acc,prec,rec,f1]),axis = 1)))

        # end loop    
else:
    # print('deciding if lstmfcnn or not')
    # if options["architecture"]=="cnn-lstm":
    #     print("decided yes")
    #     lstmfcnn(options,dataset)
    


    model, train_loader, val_loader, test_loader = prepare_train(options,dataset.data)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    loss_function = nn.NLLLoss()

    acc_train, acc_val = train_model(options=options, train_iter=train_loader,
                                        val_iter=val_loader, optimizer=optimizer, model=model, 
                                        loss_function=loss_function,
                                        weights=weights)
    train_val_metrics = np.array([acc_train,acc_val])
    print("train_val_metrics",train_val_metrics.shape)
    
    #
    #	Test and save
    #
    acc,prec,rec,f1,confmat = helper.evaluate_model(test_loader, model, "test", options["num_labels"], architecture = options["architecture"], sort=True)
    test_metrics = np.array([acc,prec,rec,f1])

    # torch.save(model.state_dict(), "weights.pth")



#
#	Print results to file
#

results_dict = {
    "train_val":train_val_metrics.tolist(),
    "test":test_metrics.tolist(),
    "confmat":confmat.tolist(),
    "options":options
}
now = datetime.now(timezone.utc)
current_time = now.strftime("%m-%d-%Y__%H:%M:%S")
print("Writing results to file...")
with open("teste1/results/out_"+current_time+".json", "w") as outfile:
    json.dump(results_dict, outfile)
print("Done!")
