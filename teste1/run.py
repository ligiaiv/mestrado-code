from classifier import Classifier, datasetBuilder,train_model
from readFile import fileReader
import os, pandas,torch
import numpy as np
from torch import nn,optim
import torch.utils.data as tud
from torchtext.data import Iterator, BucketIterator
from sklearn.model_selection import KFold

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
# pretrained_embeddings = torch.tensor(numpy.load(path + "Embeddings/"+"embeddings.npy"))

# setting hyperparameters


#creating dataset
dataset = datasetBuilder(path+'Datasets/',"labeled_data.csv")
# train_size = round(0.7*len(dataset))
# test_size = round(0.2*len(dataset))
# validation_size = (len(dataset)-train_size)-test_size
print(dataset.data[0].text)

def mySplitDataset(dataset,ratios):
		if np.sum(ratios) >1:
				print("ERROR: ratios must sum 1 or less")
		elif np.sum(ratios) <1:
				ratios.append(0.1)
		total = len(dataset)
		# train_size,test_size,val_size = sizes
		sizes = []
		sizes = np.round(np.array(ratios)*total).astype(np.int)
		# sizes = [round(x*total) for x in ratios]
		# for ratio in ratios:
				
		#         sizes.append(round(ratio*total))
				# train_size = round(train_split*len(self.data))
				# test_size = round(val_split*len(self.data))
				# validation_size = (len(self.data)-train_size)-test_size
		sizes[-1] = total - np.sum(sizes[0:-1])

		start_point = 0
		pieces = []
		for size in sizes:
				pieces.append(dataset[start_point:start_point+size])
				start_point+= size
		return pieces


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
				'num_epochs': 5,
				'batch_size': 64
		  }




# initializing model and defining loss function and optimizer
model = Classifier(options)
# model.cuda()  # do this before instantiating the optimizer


loss_function = nn.NLLLoss()
for name, param in model.named_parameters():
	if param.requires_grad:
		print(name)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)


kf = KFold(n_splits=10)

# train_set,test_set,validation_set = dataset.splitDataset(0.7,0.2)
train_set,test_set,validation_set = mySplitDataset(dataset.data,[0.7,0.2])

print(train_set[0].text)
print("TRAIN",len(train_set))
print("TEST",len(test_set))
print("VALIDATION",len(validation_set))

# dataset.data.split(np.tile(0.1,10),random_state=0)
split_lengths = (int(len(dataset.data)/10))
split_lengths = np.append(np.tile(split_lengths,9),len(dataset.data)-9*split_lengths).tolist()

subsets = tud.random_split(dataset.data,split_lengths)
print(subsets)
# dataset.randomShuffle()
# steps = np.delete(np.round(np.linspace(0,len(dataset),11)),0).astype(int)
# last_step = 0
for index in range(10):

	# test_set = dataset.data[last_step:steps[index]]
	# train_set = dataset.data[:last_step]+dataset.data[steps[index]:]
	# last_step = steps[index]
	
	test_set = subsets[index]
	train_set = subsets[:index]+subsets[index+1:]
	train_set = tud.ConcatDataset(train_set)
	train_set_size = int(0.9*len(train_set))
	train_set,validation_set = tud.random_split(train_set,[train_set_size,len(train_set)-train_set_size])
	print("TRAIN_SET:",len(train_set))
	print("TEST_SET:",len(test_set))
	print("VALIDATION_SET:",len(validation_set))
	quit()
quit()
for train_index, test_index in kf.split(dataset):
	print("TRAIN:", train_index, "TEST:", test_index)
	quit()
	#Create Iterators
	train_set,test_set = dataset.data[train_index],dataset.data[test_index]

	# train_set,validation_set = train_set
	train_iter,val_iter = BucketIterator.splits(datasets=(train_set,validation_set),batch_sizes=(options["batch_size"],options["batch_size"]),device = torch.device('cuda'), 
												sort_key =  lambda x: len(x.text),
												sort_within_batch = False,repeat = False)
	test_iter = Iterator(test_set,batch_size=options["batch_size"],device = torch.device('cuda'),sort=False,sort_within_batch=False,repeat=False,sort_key=lambda x: len(x.text))



	# X_train, X_test = data[train_index], data[test_index]
	# y_train, y_test = targets[train_index], targets[test_index]

	# train_model()
	# confusion_matrix,acc_n,acc_cr = test_model(X_test,y_test)
	# aucs_n.append(acc_n)
	# aucs_cruz.append(acc_cr)

	


helper.evaluate_model(test_set, model, "test", sort=True)