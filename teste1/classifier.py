import torch, torchtext
from torch import nn, optim
import numpy as np
import os,random
import torch.utils.data as tud
from readFile import fileReader
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset,Dataset
import helper
import tqdm
class Classifier(nn.Module):
	def __init__(self,options):
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
		embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, length, batch_first=False)

		outputs, (ht, ct) = self.lstm(embeddings)

		outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)

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

			self.TEXT.build_vocab(self.data, vectors="glove.6B.100d")
			self.LABEL.build_vocab(self.data)

		def randomShuffle(self):
			self.data = np.random.shuffle(self.data)

class myDataset(Dataset):
	def __init__(self,data,fields):
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
def train_model(options,train_iter,val_iter,optimizer,model,loss_function):
	for epoch in range(options['num_epochs']):
		print("training epoch", epoch + 1)


		# train model on training data
		for batch in tqdm.tqdm(train_iter):		
		# for batch in train_iter:
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
		helper.evaluate_model(val_iter, model, "val", sort=True)

		# monitor performance on dev data




class myConcatDataset(Dataset):

	def __init__(self,datasets,fields = None):
		self.dataset = None
		self.indices = []
		if hasattr(datasets[0], 'fields'):
			self.fields = datasets[0].fields
		else:
			self.fields = fields
		self.datasets = datasets
		self.dataset=self.concat_datasets(self.datasets)
		self.i = 0

	def concat_datasets(self,datasets):
		total_dataset = []
		for subset in datasets:
			total_dataset+=subset.examples
		return total_dataset
			

	def __len__(self):
		return len(self.dataset)

	

	def __getitem__(self, idx):

		# return {'text': self.dataset[idx].text, 
		# 		'length': len(self.dataset[idx].text), 
		# 		'label': self.dataset[idx].label
		# 		}
		return self.dataset[idx]

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
	if rand:
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
