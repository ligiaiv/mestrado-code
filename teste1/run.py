from classifier import Classifier, datasetBuilder,train_model,myConcatDataset,myDataset
from readFile import fileReader
import os, pandas,torch
import numpy as np
from torch import nn,optim,randperm
import torch.utils.data as tud
from torchtext.data import Iterator, BucketIterator
import torchtext.data as ttd
from sklearn.model_selection import KFold
import random

import helper

#
#Read File

#
# Baseado no código da aula de pytorch de Heike Adel dada no congresso RANLP2019
#

#using cuda or not
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# loading data
path = os.getcwd().split('/')
path.pop()
path = '/'.join(path)+'/'
# pretrained_embeddings = torch.tensor(numpy.load(path + "Embeddings/"+"embeddings.npy"))

# setting hyperparameters


#creating dataset
dataset = datasetBuilder(path+'Datasets/',"labeled_data.csv")
print("DATASET LEN:",len(dataset))
def mySplitDataset(dataset,ratios,rand = False, dstype = None):
		if np.sum(ratios) >1:
				print("ERROR: ratios must sum 1 or less")
		elif np.sum(ratios) <1:
				ratios.append(0.1)
		total = len(dataset)
		sizes = []
		sizes = np.round(np.array(ratios)*total).astype(np.int)
		sizes[-1] = total - np.sum(sizes[0:-1])

		start_point = 0
		pieces = []
		if random:
			if dstype == "Tabular":
				data = random.sample(dataset.examples,len(dataset))	
			else:
				data = random.sample(dataset.dataset,len(dataset))
			# dataset = ttd.Dataset(data,dataset.fields)
			# for size in pieces:
		
		for size in sizes:
				pieces.append(myDataset(data[start_point:start_point+size],dataset.fields))
				start_point+= size
		return pieces



#NN options

options = {
				'vocab_size': len(dataset.TEXT.vocab),
				'emb_dim': 100,
				'num_labels': len(dataset.LABEL.vocab),
				'hidden_lstm_dim': 200,
				'bidirectional': True,
				'num_layers': 2,
				'num_epochs': 5,
				'batch_size': 64
		  }




# initializing model and defining loss function and optimizer
model = Classifier(options)
# model.cuda()  # do this before instantiating the optimizer


loss_function = nn.NLLLoss()
# for name, param in model.named_parameters():
# 	if param.requires_grad:
# 		print(name)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)


kf = KFold(n_splits=10)

# train_set,test_set,validation_set = mySplitDataset(dataset.data,[0.7,0.2])

# print("TRAIN",len(train_set))
# print("TEST",len(test_set))
# print("VALIDATION",len(validation_set))

split_lengths = (int(len(dataset.data)/10))
split_lengths = np.append(np.tile(split_lengths,9),len(dataset.data)-9*split_lengths).tolist()
# subsets = tud.random_split(dataset.data,split_lengths)
print(dataset.data.__dict__.keys())
subsets = mySplitDataset(dataset.data,np.tile(0.1,10),rand=True,dstype = "Tabular")
for index in range(10):

	test_set = subsets[index]

	print(subsets[:index]+subsets[index+1:])
	train_set = subsets[:index]+subsets[index+1:]
	# train_set = tud.Subset(subsets[0],[x for x in sub.indices for sub in subsets[:index]+subsets[index+1:]])
	train_set =myConcatDataset(train_set)

	train_set_size = int(0.9*len(train_set))
	print([train_set_size,len(train_set)-train_set_size])
	# print(tud.random_split(train_set,[train_set_size,len(train_set)-train_set_size]))

	train_set,validation_set = mySplitDataset(train_set,[0.9,0.1],rand=True)
	# train_set,validation_set = tud.random_split(train_set,[train_set_size,len(train_set)-train_set_size])
	print(len(train_set.dataset))
	# train_set = train_set.dataset
	# validation_set = validation_set.dataset
	print("TRAIN_SET:",len(train_set))
	print("TEST_SET:",len(test_set))
	print("VALIDATION_SET:",len(validation_set))

	# train_loader = tud.DataLoader(train_set,batch_size=options["batch_size"],shuffle=True)
	# val_loader = tud.DataLoader(validation_set,batch_size=options["batch_size"],shuffle=True)
	# test_loader = tud.DataLoader(test_set,batch_size=options["batch_size"],shuffle=False)

	train_loader,val_loader = BucketIterator.splits(datasets=(train_set,validation_set),batch_sizes=(options["batch_size"],options["batch_size"]),device = device, 
												sort_key =  lambda x: len(x.text),
												sort_within_batch = False,repeat = False)
	test_loader = Iterator(test_set,batch_size=options["batch_size"],device = device,sort=False,sort_within_batch=False,repeat=False,sort_key=lambda x: len(x.text))

	train_model(options,train_loader,val_loader,optimizer,model,loss_function)	

helper.evaluate_model(test_set, model, "test", sort=True)