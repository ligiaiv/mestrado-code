import os, json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import keras.preprocessing.sequence as sequenceModule
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from random import sample

class DataHandler():

	def processData(self,filename, maxlen, MAX_VOCAB_SIZE,extra_filename = None,split = False):
		self.tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

		self.readDataFile(filename)
		# if split:
		# 	self.splitSentences()
		# 	[self.split_sequences] = self.tokenization([self.sentences,self.extra_sentences],MAX_VOCAB_SIZE)

		if extra_filename:
			self.readDataFile(extra_filename,extra=True)
			[self.sequences,self.extra_sequences] = self.tokenization([self.sentences,self.extra_sentences],MAX_VOCAB_SIZE)
		else:
			[self.sequences] = self.tokenization([self.sentences],MAX_VOCAB_SIZE)
		# print(self.tokenizer.index_word)
		# with open("vocab",'w') as vocab_file:
		# 	for line in self.tokenizer.word_index:
		# 		vocab_file.write(line+"\n")

		with open("teste2-attention/idx2word.json",'w') as wordidx:
			json.dump({
				"idx2word":self.tokenizer.index_word,
				"word2idx":self.tokenizer.word_index
			},wordidx)


		print('Found %s unique tokens.' % len(self.word2idx))
		self.data = sequenceModule.pad_sequences(self.sequences,maxlen = maxlen)
		if extra_filename:
			self.extra_data = sequenceModule.pad_sequences(self.extra_sequences,maxlen=maxlen)
		print('Shape of data tensor: ',self.data.shape)
	# def splitSentences(self):
	# 	self.extra_split = {}
	# 	for i,s in enumerate(self.sentences):
	# 		s1 = [x.split(',') for x in s.split('.')]
	# 		s2 = [x for x in s1 if (x is not '' and x is not ' ')]
	# 		if len(s2)>1:
	# 			extra_split[i] = s2
	# 		else:
	# 			extra_split[i] = None 
	def readDataFile(self,filename,extra = False):
		path = os.getcwd()
		df = pd.read_csv(path+'/Datasets/'+filename)
		labels_words = df["class"].values
		if not extra:
			self.classes = np.unique(labels_words).tolist()
			self.n_classes = len(self.classes)
		print(self.classes)

		full_arraya=np.ndarray((len(labels_words),0))
		for cl in self.classes:
			cl_array = (labels_words==cl).astype(int)
			print(full_arraya.shape,cl_array.shape)
			full_arraya = np.column_stack((full_arraya,cl_array))
		if full_arraya.sum()!= len(full_arraya):
			print("Classes not macthing")
			quit()
		if extra:
			self.extra_labels_array = full_arraya
			
			# print("self.extra_labels_array",self.extra_labels_array)
			self.extra_sentences = df["text"].values.tolist()
			# quit()
		else:
			self.labels_array = full_arraya
			self.sentences = df["text"].values.tolist()

		print(self.labels_array)
		print("{}:{}".format(filename,self.labels_array.sum(axis = 0)))

		# self.int_encoding = np.argmax(self.labels_array,axis = 0)
		print("df",df.shape)
		print("arr",self.labels_array.shape)
		

	def histogram(self,sequences,base_name):
		seq_lens = [len(x) for x in sequences]
		plt.hist(seq_lens,bins = list(range(0,max(seq_lens)+1)),width = 0.7)
		plt.xlabel('Quantidade de Palavras',fontsize=10)
		plt.ylabel('Nº de Tweets',fontsize=10)
		plt.xticks(fontsize=8)
		plt.yticks(fontsize=8)
		# plt.ylabel('Frequency',fontsize=15)
		plt.title('Histograma - Nº de Palavras por Tweet - '+base_name,fontsize=12)
		plt.show()
		# quit()

	

	def tokenization(self,sentences_list,MAX_VOCAB_SIZE,extra = False):
		total_sentences = [item for sublist in sentences_list for item in sublist]
		if not extra:
			self.tokenizer.fit_on_texts(total_sentences) #gives each word a number
			self.word2idx = self.tokenizer.word_index

		sequences = [self.tokenizer.texts_to_sequences(sentences) for sentences in sentences_list]
		# print(sequences)
		return sequences

	def readEmbedding(self,embedding_file,path,dim):
		self.embeddings_index = {}
		EMBEDDING_DIR = path
		f = open(os.path.join(EMBEDDING_DIR, embedding_file))
		for line in f:
			values = line.split()
			if len(values) != dim+1:
				continue
			try:
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				self.embeddings_index[word] = coefs
			except:
				pass
			
		f.close()

		print('Found %s word vectors.' % len(self.embeddings_index))

		return self.embeddings_index

	def createMatrix(self,EMBEDDING_DIM):
		self.embedding_matrix = np.zeros((len(self.word2idx) + 1, EMBEDDING_DIM))
		for word, i in self.word2idx.items():
			embedding_vector = self.embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				self.embedding_matrix[i] = embedding_vector
	
	# def splitData(data,rate):
def get_accuracy(hypos, refs):
	print("hypos",hypos)
	hypos = (hypos==np.max(hypos,axis=1)[:,None])
	print("hypos {}\nrefs{}".format(hypos,refs))
	conf_matrix = np.matmul(hypos.transpose(), refs)
	print("conf_matrix",conf_matrix)
	assert(len(hypos) == len(refs))
	metrics = np.ndarray((4, 0))
	for categ in range(hypos.shape[1]):
		TP = conf_matrix[categ, categ]
		FP = conf_matrix[categ, :].sum() - TP
		FN = conf_matrix[:, categ].sum() - TP
		TN = conf_matrix.sum() - FP - FN - TP

		acc = (TP+TN)/(TP+TN+FP+FN)  # acc
		prec = TP/(TP+FP)  # prec
		recall = TP/(TP+FN)  # recall
		f1 = 2*recall*prec/(recall+prec)

		cat_metrics = np.expand_dims(np.array([acc, prec, recall, f1]), axis=1)
		metrics = np.hstack((metrics, cat_metrics))

	metrics = np.nan_to_num(metrics)

	[acc, prec, recall, f1] = (
		(metrics*conf_matrix.sum(axis=0)).sum(axis=1))/(conf_matrix.sum()).tolist()
	return (acc, prec, recall, f1,conf_matrix)
class LstmOptions:

	def __init__(self,options):
		self.out = options["hidden_lstm_dim"]
		self.bidirectional = options["bidirectional"]
		self.emb_dim = options["emb_dim"]
		self.freeze_emb = options["freeze_emb"]
		self.attention = options["attention_size"]
		self.dropout = options["dropout"]
		self.layers = options["num_layers"]
		self.seq_len = options["seq_len"]
	
def create_sentence(sentence,alfa,prob_table = None):
	sentence = np.squeeze(sentence)
	indexes = np.where(sentence > 0)[0]
	if alfa >= 1:
		times = alfa
	else:
		times = int(alfa*len(indexes))
	#times = alfa
	if prob_table is not None:	
		# print(indexes)
		# print(sentence[indexes])
		# print(prob_table.shape)
		probs = prob_table[0,sentence[indexes]]
		# print("probs",probs)
		# print(probs.sum())
		weights = np.exp(1/probs)/np.sum(np.exp(1/probs))
		print("Probability Weights: ", weights)
		# print(weights)
		idx = np.random.choice(indexes,min(times,len(indexes)),p = weights,replace=False)
		# quit()
	else:# print(indexes)
		idx = np.random.choice(indexes,min(times,len(indexes)),replace=False)
	
	new_sentences=np.ndarray((0,len(sentence)))
    # words = sentence.split()
	for i in idx:
		new_ = np.insert(np.delete(sentence,i),0,0)
		# print("new_sentences {} new_ {}".format(new_sentences.shape,new_.shape))
		new_sentences = np.vstack((new_sentences,np.expand_dims(new_,0)))
	return new_sentences, times

def data_augmentation(X,Y,TIMES = 5, prob_table = None):
	
	new_X = np.ndarray((0,X.shape[1]))
	new_Y = np.ndarray((0,Y.shape[1]))
	# TIMES = 5
	for i in range(len(Y)):
		if prob_table is not None:
			# print("np.where(Y[i]==1)",np.where(Y[i]==1))
			# print(Y[i])
			new,yTIMES = create_sentence(X[i],TIMES,prob_table=prob_table[np.where(Y[i]==1)]) #send probs of that class
		else:
			new,yTIMES = create_sentence(X[i],TIMES)
		new = np.vstack((new,X[i]))
		new_X = np.vstack((new_X,new))
		# print(Y[i])
		# print("new_Y {} tiled Y{}".format(new_Y.shape,np.repeat(np.expand_dims(Y[i],axis=0),TIMES, axis=0).shape))
		# new_ = np.vstack(np.repeat(np.expand_dims(Y[i],axis=0),TIMES, axis=0),)
		new_Y = np.vstack((new_Y,np.repeat(np.expand_dims(Y[i],axis=0),yTIMES+1, axis=0)))
		# print(new_X.shape)
	
	random_order=np.arange(len(new_X))
	np.random.shuffle(random_order)	
	new_X = new_X[random_order]
	new_Y = new_Y[random_order]

	return new_X,new_Y

def get_word_probability(X,Y):
	n_words = np.max(X)+1
	ohX = np.zeros((len(X),n_words))
	aux = np.arange(len(X))[:,None]
	ohX[aux,X] =1
	probability_by_class = np.ndarray((0,n_words))
	for i in range(Y.shape[1]):
		probs = np.sum(ohX[Y[:,i] == 1],axis = 0)
		probability_by_class = np.vstack((probability_by_class,probs))
	# probability_by_class = probability_by_class/np.expand_dims(probability_by_class.sum(axis = 1),axis=1)
	return probability_by_class

def add_and_shuffle(Xlist,Ylist):
	Xout = np.concatenate(Xlist,axis=0)
	Yout = np.concatenate(Ylist,axis = 0)
	print(Yout)
	# Xout = [item for sublist in Xlist for item in sublist]
	# Yout = [item for sublist in Ylist for item in sublist]

	print(len(Xout))
	random_order=np.arange(len(Xout))
	np.random.shuffle(random_order)
	print(random_order)
	Xout = Xout[random_order]
	Yout = Yout[random_order]

	return Xout,Yout