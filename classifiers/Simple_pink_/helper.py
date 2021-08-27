import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt


class DataHandler():

	def readDataFile(self,filename, maxlen, MAX_VOCAB_SIZE):
		path = os.getcwd()
		df = pd.read_csv(path+'/Datasets/'+filename)
		labels_words = df["class"].values
		classes = np.unique(labels_words).tolist()
		print(classes)
		full_arraya=np.ndarray((len(labels_words),0))
		for cl in classes:
			cl_array = (labels_words==cl).astype(int)
			print(full_arraya.shape,cl_array.shape)
			full_arraya = np.column_stack((full_arraya,cl_array))
		# sexism_array = (labels_words=="sexism").astype(int)
		# racism_array= (labels_words=="racism").astype(int)
		# none_array = (labels_words=="none").astype(int)
		# self.labels_array = np.stack([sexism_array,racism_array,none_array]).T
		# self.labels_array = np.stack(full_arraya).T
		self.labels_array = full_arraya
		print(self.labels_array)

		# print("labels_array",labels_array)
		self.int_encoding = np.argmax(self.labels_array,axis = 0)
		# print("int_encoding",int_encoding)
		print("df",df.shape)
		print("arr",self.labels_array.shape)
		# new_df = pd.concat([df,pd.DataFrame(data = self.labels_array.T, columns=classes_names)],axis=1)

		# print(new_df.columns)
		# print(new_df)



		sentences = df["text"].values.tolist()
	
		tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
		tokenizer.fit_on_texts(sentences) #gives each word a number
		sequences = tokenizer.texts_to_sequences(sentences)
		self.word2idx = tokenizer.word_index
		# seq_lens = [len(x) for x in sequences]
		# plt.hist(seq_lens)
		# plt.show()
		print('Found %s unique tokens.' % len(self.word2idx))
		self.data = sequence.pad_sequences(sequences,maxlen = maxlen)
		print('Shape of data tensor: ',self.data.shape)
	
	def readEmbedding(self,embedding_file,path):
		self.embeddings_index = {}
		EMBEDDING_DIR = path
		f = open(os.path.join(EMBEDDING_DIR, embedding_file))
		for line in f:
			values = line.split(" ")
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			self.embeddings_index[word] = coefs
		f.close()

		print('Found {} word vectors.'.format(len(self.embeddings_index)))

		return self.embeddings_index

	def createMatrix(self,EMBEDDING_DIM):
		#leaves the embedding_matrix[0] as a zero vector
		self.embedding_matrix = np.zeros((len(self.word2idx) +1, EMBEDDING_DIM))
		#start counting from 1
		for word, i in self.word2idx.items():
			embedding_vector = self.embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				self.embedding_matrix[i] = embedding_vector
		print(self.embedding_matrix[0])
		print("Embedding matrix of shape {}".format(self.embedding_matrix.shape))
	# def splitData(data,rate):
def get_accuracy(hypos, refs):
	print("AA",(hypos>0.5).sum())
	if hypos.shape[1]==1:
		hypos = np.hstack((hypos,1-hypos))
		refs = np.hstack((refs,1-refs))
	print("hypos",hypos.shape)
	# print(hypos == np.max(hypos,axis=1)[:,None])
	hypos = (hypos==np.max(hypos,axis=1)[:,None])
	# hypos = (hypos==np.max(hypos,axis=1)[:,None])

	# print(hypos)
	# quit()
	print("hypos {}\nrefs{}".format(hypos.shape,refs.shape))
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

