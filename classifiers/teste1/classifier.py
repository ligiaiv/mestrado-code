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
import copy,re
import pandas as pd



class datasetBuilder():
	ALPHABETH_STRING = r' abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}'

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
			eos_index = self.bertTokenizer.convert_tokens_to_ids(self.bertTokenizer.eos_token)
			bos_index = self.bertTokenizer.convert_tokens_to_ids(self.bertTokenizer.bos_token)

		elif '-cl' in options["architecture"]:
			tokenizer = self.char_tokenizer
			pad_index = "<pad>"
			bos_index = "<sos>"
			eos_index = "<eos>"
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

	def char_tokenizer(self, text):  # create a tokenizer function
		# text = text.lower()
		return list(text.lower())
			

	def preprocessData(self):
		vector_str = self.options["emb_type"]+"."+str(self.options["emb_dim"])+"d"


		if 'cl' in self.options["architecture"]: 
			self.TEXT.build_vocab(self.data)

		elif not(self.options["architecture"] == "bert"):
			if 'my' in self.options["emb_type"]:
				vectors = torchtext.vocab.Vectors(vector_str+".txt",cache="Embeddings")
				self.TEXT.build_vocab(self.data, vectors=vectors)
			else:		
				self.TEXT.build_vocab(self.data, vectors=vector_str, vectors_cache = "Embeddings")
			
		# elif self.options["architecture"] == "cnn-cl":
		# 	# stoi = dict(zip(list(self.ALPHABETH_STRING), list(range(1,self.ALPHABETH_STRING+1))))
		# 	# itos = dict(zip(stoi.values(),stoi.keys()))
		# 	self.TEXT.build_vocab(self.data)
		# tokenized_text = self.bertTokenizer.tokenize(some_text)

		# self.bertTokenizer.convert_tokens_to_ids(tokenized_text)

		self.LABEL.build_vocab(self.data)

		# print(self.data.examples[0].text)

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
		print(np.sum(ratios))
		np.append(ratios,0.1)
		# ratios.append(0.1)
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

def train_model(options, train_iter, val_iter, optimizer, model, loss_function, weights = None):
	# print("point 1")
	accs_val = []
	accs_train = []
	for epoch in range(options['num_epochs']):
		print("training epoch", epoch + 1)

		# train model on training data
		for batch in tqdm.tqdm(train_iter):
			# print("point 2")
			# print("point 3")
			# for batch in train_iter:
			x, l = batch.text

			# l = numpy.ma.size(batch.text,0)
			y = batch.label
			optimizer.zero_grad()
			x, l, y = helper.sort_by_length(x, l, y)
			# print("point 4")
			if (options["architecture"]=="bert"):
				scores = model(x.T)
			else:
				scores = model(x, l)
			# print("point 5")
			labels = y
			loss = loss_function(scores, labels)
			# print("loss",loss)
			loss.backward()
			optimizer.step()
			# print("point 6")

		acc_train,_,_,_,_ = helper.evaluate_model(train_iter, model, "train", options["num_labels"],architecture = options["architecture"], sort=True)
		accs_train.append(acc_train)
		acc_val,_,_,_,_ = helper.evaluate_model(val_iter, model,"val", options["num_labels"], architecture = options["architecture"], sort=True)
		accs_val.append(acc_val)
		print("Acc train: ",acc_train,"\tacc val: ",acc_val)
	return (accs_train, accs_val)
	# monitor performance on dev data


class LSTMClassifier(nn.Module):
	def __init__(self, options, pretrained_embedding=None):
		super(LSTMClassifier, self).__init__()
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
							batch_first=False,
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
		print(embeddings.shape)
		embeddings = nn.utils.rnn.pack_padded_sequence(
			embeddings, length, batch_first=False)

		outputs, (ht, ct) = self.lstm(embeddings)

		outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(
			outputs, batch_first=False)
		print("output (seq_len, batch, num_directions * hidden_size)",outputs.shape)

		print("ht (num_layers * num_directions, batch, hidden_size)",ht.shape)

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
		# print("lstm scores",scores.shape)
		return scores



class BertForSequenceClassification(nn.Module):
  
	def __init__(self, options):
		super(BertForSequenceClassification, self).__init__()
		self.num_labels = options["num_labels"] #4
		self.bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = False) #embedding
		self.dropout = nn.Dropout(options["dropout"])
		self.classifier = nn.Linear(options["hidden_size_bert"], options["num_labels"])
		nn.init.xavier_normal_(self.classifier.weight)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		#changed from 'output_all_encoded_layers' to 'output_hidden_states' and put in the configuration
		# print("input_ids",input_ids.shape)
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# print("pooled_output",pooled_output.shape)
		pooled_output = self.dropout(pooled_output)
		# print("pooled_output_dropout",pooled_output.shape)
		logits = self.classifier(pooled_output)
		scores = self.softmax(logits)
		return scores


class CNN1DClassifier(nn.Module):
  
	def __init__(self, options,pretrained_embedding = None):
		super(CNN1DClassifier, self).__init__()
		self.options = options

		self.embedding = nn.Embedding(num_embeddings=options["vocab_size"],
									  embedding_dim=options["emb_dim"],
									  padding_idx=0)

		if pretrained_embedding is not None:
			print("\n\t-Using pre-trained embedding")
			self.embedding.from_pretrained(
				pretrained_embedding.vectors, freeze=options["freeze_emb"])
		# self.embedding.load_state_dict()
		
		# self.linear = nn.Linear(in_features=lstm_out_size,
		# 						out_features=options["num_labels"])
		# self.softmax = nn.LogSoftmax(dim=1)

		self.dropout_input = nn.Dropout2d(0.25)
		# options["emb_dim"]
		# dimension order of cnn1d input (batch,channels_in, seq_len)
		self.conv1 = nn.Sequential(
			nn.Conv1d(options["emb_dim"],100,3),
			nn.ReLU(),
			nn.MaxPool1d(options["seq_len"]-2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv1d(options["emb_dim"],100,4),
			nn.ReLU(),
			nn.MaxPool1d(options["seq_len"]-3)
		)
		self.conv3 = nn.Sequential(
			nn.Conv1d(options["emb_dim"],100,5),
			nn.ReLU(),
			nn.MaxPool1d(options["seq_len"]-4)
		)
		# self.conv = nn.Sequential(conv1,conv2,conv3)
		self.dropout_output = nn.Dropout2d(0.5)
		self.activation1 = nn.ReLU()
		self.dense = nn.Linear(3*100,options["num_labels"])
		# self.activation2 = nn.Softmax(dim = 1)
		self.activation2 = nn.LogSoftmax(dim=1)
		self.cnn_output = None

	def forward(self, x, length):
		
		
		# batch_size = x.size()[1]
		# print("x size in forward:",x.size())
		# print("x in",x)
		
		embeddings = self.embedding(x)
		# embeddings = nn.utils.rnn.pack_padded_sequence(
		# 	embeddings, length, batch_first=False)

		# outputs, (ht, ct) = self.lstm(embeddings)

		# outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(
		# 	outputs, batch_first=False)
		# print("after embedding layer ",embeddings.shape)
		embeddings = embeddings.transpose(0,1)
		embeddings = embeddings.transpose(2,1)
		# print("before conv1 ",embeddings.shape)

		x = self.dropout_input(embeddings)
		# print("before conv 1",x.shape)

		c1 = self.conv1(x)
		# print("after conv 1",c1.shape)
		c2 = self.conv2(x)
		# print("after conv 2",c2.shape)

		c3 = self.conv3(x)
		# print("after conv 3",c3.shape)
		# print("c1",c1.shape)
		x = torch.cat([c1,c2,c3],dim=1)
		self.cnn_output = x 

		# print("x after cat",x.shape)
		x = x.squeeze()
		# x = self.conv(x)
		x = self.dropout_output(x)
		# print("after dropout",x.shape)

		x = self.activation1(x)
		# print("after activation",x.shape)

		x = self.dense(x)
		scores = self.activation2(x)#softmax
		# linear_out = self.linear(lstm_out)
		# scores = self.softmax(linear_out)
		# print("snn scores",scores.shape)
		return scores

class LSTM_CNNClassifier(nn.Module):
  
	def __init__(self, options,pretrained_embedding = None):
		super(LSTM_CNNClassifier, self).__init__()

		self.options = options

		self.embedding = nn.Embedding(num_embeddings=options["vocab_size"],
									  embedding_dim=options["emb_dim"],
									  padding_idx=0)

		if pretrained_embedding is not None:
			print("\n\t-Using pre-trained embedding")
			self.embedding.from_pretrained(
				pretrained_embedding.vectors, freeze=options["freeze_emb"])

		self.lstm = nn.LSTM(input_size=options['emb_dim'],
							hidden_size=options['hidden_lstm_dim'],
							num_layers=options['num_layers'],
							batch_first=True,
							bidirectional=options['bidirectional'])

		lstm_out_size = options['hidden_lstm_dim']
		if options['bidirectional']:
			lstm_out_size *= 2


		self.dropout_input = nn.Dropout2d(0.25)
		# options["emb_dim"]
		# dimension order of cnn1d input (batch,channels_in, seq_len)
		self.conv1 = nn.Sequential(
			nn.Conv1d(lstm_out_size,100,3),
			nn.ReLU(),
			nn.MaxPool1d(options["seq_len"]-2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv1d(lstm_out_size,100,4),
			nn.ReLU(),
			nn.MaxPool1d(options["seq_len"]-3)
		)
		self.conv3 = nn.Sequential(
			nn.Conv1d(lstm_out_size,100,5),
			nn.ReLU(),
			nn.MaxPool1d(options["seq_len"]-4)
		)
		# self.conv = nn.Sequential(conv1,conv2,conv3)
		self.dropout_output = nn.Dropout2d(0.5)
		self.activation1 = nn.ReLU()
		self.dense = nn.Linear(in_features=3*100,
								out_features=options["num_labels"])
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x, length):
		
		# batch_size = x.size()[1]
		# print("x size in forward:",x.size())
		# print("x in",x)
		
		embeddings = self.embedding(x) 
		# print("lstm-cnn embeddings.shape",embeddings.shape)
		# embeddings = nn.utils.rnn.pack_padded_sequence(
		# 	embeddings, length, batch_first=False)

		outputs, (ht, ct) = self.lstm(embeddings)
		# print("lstm-cnn outputs.shape",outputs.shape)

		# outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(
		# 	outputs, batch_first=False)

		lstm_out = outputs

		# print("lstm-cnn outputs.shape",lstm_out.shape)
		# if self.options['bidirectional']:
		# 	ht = ht.view(self.options['num_layers'], 2, batch_size,
		# 				 self.options['hidden_lstm_dim'])
		# 	ht = ht[-1]  # get last (forward and backward) hidden states
		# 	# from last layer
		# 	# concatenate last hidden states from forward and backward passes
		# 	lstm_out = torch.cat([ht[0], ht[1]], dim=1)
		# else:
		# 	# get the last hidden state of the outmost layer
		# 	lstm_out = ht[-1, :, :]

		# linear_out = self.linear(lstm_out)cnn_lstm
		# scores = self.softmax(linear_out)

		#
		#		COLOCAR CNN AQUI!!!!!!!!!!!!!!!!!!!!!!!!!!!!111
		#		output da lstm shape = (seq_len,batch_size,hidden_LSTM_size(=100))
		#
		#		input do cnn = (batch,channels_in, seq_len)

		cnn_in = lstm_out.transpose(0,1)
		cnn_in = cnn_in.transpose(2,1)
		x = self.dropout_input(cnn_in)
		c1 = self.conv1(x)
		print("after conv 1",x.shape)
		c2 = self.conv2(x)
		print("after conv 2",x.shape)

		c3 = self.conv3(x)

		x = torch.cat([c1,c2,c3],dim=1)
		x = x.squeeze()
		x = self.dropout_output(x)
		x = self.activation1(x)
		
		x = self.dense(x)
		scores = self.softmax(x)#softmax
		
		return scores


class CNN_LSTMClassifier(nn.Module):
  
	def __init__(self, options,pretrained_embedding = None, weights=None):
		super(CNN_LSTMClassifier, self).__init__()
		print("Using CNN-LSTM model")
		self.options = options

		self.embedding = nn.Embedding(num_embeddings=options["vocab_size"],
									  embedding_dim=options["emb_dim"],
									  padding_idx=0)

		if pretrained_embedding is not None:
			print("\n\t-Using pre-trained embedding")
			self.embedding.from_pretrained(
				pretrained_embedding.vectors, freeze=options["freeze_emb"])

		self.lstm = nn.LSTM(input_size=100,
							hidden_size=options['hidden_lstm_dim'],
							num_layers=options['num_layers'],
							# batch_first=True,
							bidirectional=options['bidirectional'])

		self.dropout_input = nn.Dropout2d(0.25)

		self.conv = nn.Sequential(
			nn.Conv1d(options["emb_dim"],100,3),
			nn.ReLU(),
			# nn.MaxPool1d(options["seq_len"]-2)
			nn.MaxPool1d(2)
		)

		# self.conv1 = nn.Sequential(
		# 	nn.Conv1d(options["emb_dim"],100,3),
		# 	nn.ReLU(),
		# 	# nn.MaxPool1d(options["seq_len"]-2)
		# )
		# self.conv2 = nn.Sequential(
		# 	nn.Conv1d(options["emb_dim"],100,4),
		# 	nn.ReLU(),
		# 	# nn.MaxPool1d(options["seq_len"]-3)
		# )
		# self.conv3 = nn.Sequential(
		# 	nn.Conv1d(options["emb_dim"],100,5),
		# 	nn.ReLU(),
		# 	# nn.MaxPool1d(options["seq_len"]-4)
		# )
		# self.conv = nn.Sequential(conv1,conv2,conv3)
		self.dropout_output = nn.Dropout2d(0.5)
		self.activation1 = nn.ReLU()


		lstm_out_size = options['hidden_lstm_dim']


		self.linear = nn.Linear(in_features=options['hidden_lstm_dim'],
								out_features=options["num_labels"])
		self.softmax = nn.LogSoftmax(dim=1)


		if weights is not None:
			print("using weights")
			self.conv1.load_state_dict(weights, strict=False)
			self.conv1[0].req= False
			self.conv2.load_state_dict(weights, strict=False)
			self.conv2[0].req= False
			self.conv3.load_state_dict(weights, strict=False)
			self.conv3[0].req= False
			
	def forward(self, x, length):
		
		embeddings = self.embedding(x)
		# print("after emb ",embeddings.shape)
		embeddings = embeddings.transpose(0,1)
		embeddings = embeddings.transpose(2,1)
		x = self.dropout_input(embeddings)
		
		x = self.conv(x)
		print('out of conv',x.shape)
		# c1 = self.conv1(x)
		# # print("after conv 1",c1.shape)
		# c2 = self.conv2(x)
		# # print("after conv 2",c2.shape)

		# c3 = self.conv3(x)

		# x = torch.cat([c1,c2,c3],dim=2)

		x = x.transpose(2,1)
		x = x.transpose(0,1)
		# print("before LSTM ",x.shape)
		print('before lstm',x.shape)
		outputs, (ht, ct) = self.lstm(x)
		# print("after lstm ",ht.shape)

		# lstm_out = outputs
		lstm_out = ht[-1, :, :]
		# print("LSTM_out ",lstm_out.shape)

		linear_out = self.linear(lstm_out)

		scores = self.softmax(linear_out)

		return scores

class LSTM_from_CNN(nn.Module):
	def __init__(self, options, pretrained_embedding=None):
		super(LSTM_from_CNN, self).__init__()
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
		# print("lstm scores",scores.shape)
		return scores
