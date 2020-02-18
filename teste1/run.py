from classifier import Classifier, datasetBuilder,train_model,myConcatDataset,myDataset,mySplitDataset
from readFile import fileReader
import os, pandas,torch,json
import numpy as np
from torch import nn,optim
import torch.utils.data as tud
from torchtext.data import Iterator, BucketIterator
import torchtext.data as ttd
import random,sys
from datetime import datetime

import helper

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-kf", help="number of folds, default 10")
# parser.add_argument("-e", help="number of epochs, 5")

# args = parser.parse_args()
# KFOLD = args.kf
#Read File

#
# Baseado no código da aula de pytorch de Heike Adel dada no congresso RANLP2019
#

#using cuda or not
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA is available",torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)

# loading data
path = os.getcwd().split('/')
path.pop()
path = '/'.join(path)+'/'
# pretrained_embeddings = torch.tensor(numpy.load(path + "Embeddings/"+"embeddings.npy"))

# setting hyperparameters


#creating dataset
dataset = datasetBuilder(path+'Datasets/',"labeled_data.csv")
print("DATASET LEN:",len(dataset))



#NN options
options = json.load(open("variables.json","r")) 
options["vocab_size"] = len(dataset.TEXT.vocab)
options["num_labels"] = len(dataset.LABEL.vocab)

print(options)
# options = {
# 				'vocab_size': len(dataset.TEXT.vocab),
# 				'emb_dim': 100,
# 				'num_labels': len(dataset.LABEL.vocab),
# 				'hidden_lstm_dim': 200,
# 				'bidirectional': True,
# 				'num_layers': 2,
# 				'num_epochs': args.e,
# 				'batch_size': 64
# 		  }




# initializing model and defining loss function and optimizer
model = Classifier(options)
# model.cuda()  # do this before instantiating the optimizer


loss_function = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
KFOLD = options["kfold"]

split_lengths = (int(len(dataset.data)/KFOLD))
split_lengths = np.append(np.tile(split_lengths,KFOLD-1),len(dataset.data)-(KFOLD-1)*split_lengths).tolist()

subsets = mySplitDataset(dataset.data,np.tile(0.1,10),rand=True,dstype = "Tabular")
accs = []
for index in range(KFOLD):
	print("KFOLD:",index)
	test_set = subsets[index]
	print(test_set)

	train_set = subsets[:index]+subsets[index+1:]
	train_set =myConcatDataset(train_set)
	train_set_size = int(0.9*len(train_set))

	train_set,validation_set = mySplitDataset(train_set,[0.9,0.1],rand=True)

	print("TRAIN_SET:",len(train_set))
	print("TEST_SET:",len(test_set))
	print("VALIDATION_SET:",len(validation_set))
	print(train_set)
	# train_loader = tud.DataLoader(train_set,batch_size=options["batch_size"],shuffle=True)
	# val_loader = tud.DataLoader(validation_set,batch_size=options["batch_size"],shuffle=True)
	# test_loader = tud.DataLoader(test_set,batch_size=options["batch_size"],shuffle=False)
	
	train_loader,val_loader = BucketIterator.splits(datasets=(train_set,validation_set),batch_sizes=(options["batch_size"],options["batch_size"]),device = device, 
												sort_key =  lambda x: len(x.text),sort = True,
												sort_within_batch = False,repeat = False)
	test_loader = Iterator(test_set,batch_size=options["batch_size"],device = device,sort=False,sort_within_batch=False,repeat=False,sort_key=lambda x: len(x.text))

	train_model(options= options,train_iter= train_loader,val_iter= val_loader,optimizer= optimizer,model= model,loss_function= loss_function)	

	acc = helper.evaluate_model(test_loader, model, "test", sort=True)
	print("ACCS",accs)
	print("ACC",acc)
	accs.append(acc)

now = datetime.now()
current_time = now.strftime("%m-%d-%Y__%H:%M:%S")
with open("results/"+current_time, "w") as outfile:
	outfile.write(','.join([str(x) for x in accs]))


