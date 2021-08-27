from __future__ import print_function
import tensorflow as tf
# print(tf.__version__)
# print(tf.keras.__version__)
# import ktrain
# from ktrain import text
# import os,json
# import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, Softmax
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D
from keras.layers import Add, Multiply, Flatten, Layer, Lambda
from keras.layers.merge import concatenate
from keras import Model
# from tf.keras.layers import Attention

# from keras.datasets import imdb
# from keras.utils import to_categorical


class AttentionLayer(Model):
	def __init__(self,units):
		super(AttentionLayer, self).__init__()
		self.W1 = Dense(units)
		self.W2 = Dense(units)
		self.V = Dense(1)
	
	def __call__(self,sequence_outputs,state_h):
		# score = tf.nn.tanh(
		# 	self.W1(sequence_outputs) + self.W2(state_h))
		w1 = self.W1(sequence_outputs)
		w2 = self.W2(state_h)
		wsum = Add()([w1,w2])
		score = Activation("tanh")(wsum)
		# attention_weights shape == (batch_size, max_length, 1)
		# attention_weights = tf.nn.softmax(self.V(score), axis=1)
		attention_weights = Softmax(axis = 1,name = "attention_softmax")(self.V(score))
		context_vector = Multiply()([attention_weights,sequence_outputs])
		# context_vector = attention_weights * sequence_outputs
		print("attention_weights {}\nsequence_outputs {}\ncontext_vector{}".format(attention_weights,sequence_outputs,context_vector))
		# context_vector = tf.reduce_sum(context_vector, axis=1)
		# context_vector = ComputeSum()(context_vector,axis = 1)
		context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
		return context_vector, attention_weights		  

def LSTM_Attention_Model(nclasses,maxlen,max_features,embedding_size,attention_size,lstm_output_size,train_embedding=True, bidirectional = False):
	sequence_input = Input(shape = (maxlen,))
	embedding = Embedding(max_features, embedding_size, input_length=maxlen)(sequence_input)
	embedding.trainable = train_embedding    
	if bidirectional:
		print("BIDIRECTIONAL")

		(lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(LSTM(lstm_output_size, return_sequences=True, return_state=True), name="bi_lstm_0")(embedding)
		# lstm = Bidirectional(LSTM(lstm_output_size,return_sequences = True))(embedding)
		state_h = Concatenate()([forward_h, backward_h])
		state_c = Concatenate()([forward_c, backward_c])
	else:
		lstm, state_h, state_c = LSTM(lstm_output_size, return_sequences=True, return_state=True, name="bi_lstm_0")(embedding)
		print(lstm,state_h)
	context_vector, attention_weights = AttentionLayer(attention_size)(lstm, state_h)

	# dense1 = Dense(20, activation="relu")(context_vector)
	# dropout = Dropout(0.05)(dense1)
	output = Dense(nclasses, activation="sigmoid")(context_vector)


	model = Model(inputs = [sequence_input], outputs=[output])

	return model


def FF_Model(nclasses,maxlen,max_features,embedding_size,hidden_size,depth,train_embedding=True):
	
	# print("nclasses{}, maxlen{},maxfeatures{}, embedding_size{}, hidden_size{},depth{}".format(nclasses,maxlen,max_features,embedding_size,hidden_size,de) )
	input  = Input(shape = (maxlen,))
	embedding = Embedding(max_features, embedding_size, input_length=maxlen)(input)
	embedding.trainable = train_embedding    
	# dense =Dense(hidden_size, activation="relu")(embedding)

	flat = Flatten()(embedding)
	deep = Sequential()
	for i in range(depth-1):
		print(i)
		deep.add(Dense(hidden_size,activation="relu"))
	
	deep = deep(flat)
	out = Dense(nclasses,activation="sigmoid")(deep)

	model = Model(inputs = [input], outputs=[out])
	return model
 

def LSTM_Model(nclasses,max_features,options):
	input = Input(shape = (options.seq_len,))
	embedding = Embedding(max_features, options.emb_dim, input_length=options.seq_len)(input)
	embedding.trainable = not options.freeze_emb 
	dropout1 = Dropout(options.dropout)(embedding)   
	if options.bidirectional:
		print("BIDIRECTIONAL")
		lstm = Bidirectional(LSTM(options.out))(dropout1)
	else:
		lstm = LSTM(options.out)(dropout1)
	dropout2 = Dropout(options.dropout)(lstm)
	dense = Dense(nclasses)(dropout2)
	activation = Activation('sigmoid')(dense)

	model = Model(inputs = [input], outputs=[activation])

	return model

def CNN_LSTM_Model(nclasses,maxlen,max_features,embedding_size,lstm_output_size,filters,pool_size,dropout_rate,train_embedding):
	input = Input(shape = (maxlen,))
	embedding = Embedding(max_features, embedding_size, input_length=maxlen)(input)
	embedding.trainable = train_embedding
	conv1= Conv1D(filters,
					3,
					padding='valid',
					activation='relu',
					strides=1)(embedding)
	conv1_out = MaxPooling1D(pool_size=pool_size)(conv1)

	conv2= Conv1D(filters,
					4,
					padding='valid',
					activation='relu',
					strides=1)(embedding)
	conv2_out = MaxPooling1D(pool_size=pool_size)(conv2)

	# conv3= Conv1D(filters,
	# 				5,
	# 				padding='valid',
	# 				activation='relu',
	# 				strides=1)(embedding)
	# conv3_out = MaxPooling1D(pool_size=pool_size)(conv3)

	merged = concatenate([conv1_out,conv2_out],axis=1)
	dropout = Dropout(dropout_rate)(merged)

	# print("merged.shape",merged.shape)
	lstm = LSTM(lstm_output_size)(dropout)
	dense = Dense(nclasses)(lstm)
	activation = Activation('sigmoid')(dense)

	model = Model(inputs = [input], outputs=[activation])

	return model


def LSTM_CNN_Model(nclasses,maxlen,max_features,embedding_size,lstm_output_size,filters,pool_size,dropout_rate,train_embedding = True,bidirectional = False):
	input = Input(shape = (maxlen,))
	embedding = Embedding(max_features, embedding_size, input_length=maxlen)(input)
	embedding.trainable = train_embedding
	
	if bidirectional:
		print("BIDIRECTIONAL")
		(lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(LSTM(lstm_output_size, return_sequences=True, return_state=True), name="bi_lstm_0")(embedding)
		# lstm = Bidirectional(LSTM(lstm_output_size,return_sequences = True))(embedding)
		state_h = Concatenate()([forward_h, backward_h])
		state_c = Concatenate()([forward_c, backward_c])
	else:
		lstm, state_h, state_c = LSTM(lstm_output_size, return_sequences=True, return_state=True, name="bi_lstm_0")(embedding)
	
	
	conv1= Conv1D(filters,
					3,
					padding='valid',
					activation='relu',
					strides=1)(lstm)
	conv1_out = GlobalMaxPool1D()(conv1)

	conv2= Conv1D(filters,
					4,
					padding='valid',
					activation='relu',
					strides=1)(lstm)
	conv2_out = GlobalMaxPool1D()(conv2)

	# conv3= Conv1D(filters,
	# 				5,
	# 				padding='valid',
	# 				activation='relu',
	# 				strides=1)(lstm)
	# conv3_out = GlobalMaxPool1D()(conv3)

	merged = concatenate([conv1_out,conv2_out],axis=1)
	
	dropout = Dropout(dropout_rate)(merged)
	output = Dense(nclasses, activation="sigmoid")(dropout)

	model = Model(inputs = [input], outputs=[output])

	return model


def CNN_Model(nclasses,maxlen,max_features,embedding_size,filters,pool_size,dropout_rate,train_embedding = True):
	input = Input(shape = (maxlen,))
	embedding = Embedding(max_features, embedding_size, input_length=maxlen)(input)
	embedding.trainable = train_embedding
	
	
	conv1= Conv1D(filters,
					3,
					padding='valid',
					activation='relu',
					strides=1)(embedding)
	# conv1_out = MaxPooling1D(pool_size=pool_size)(conv1)
	conv1_out = GlobalMaxPool1D()(conv1)

	conv2= Conv1D(filters,
					4,
					padding='valid',
					activation='relu',
					strides=1)(embedding)
	# # conv2_out = MaxPooling1D(pool_size=pool_size)(conv2)
	conv2_out = GlobalMaxPool1D()(conv2)

	# conv3= Conv1D(filters,
	# 				5,
	# 				padding='valid',
	# 				activation='relu',
	# 				strides=1)(embedding)
	# conv3_out = MaxPooling1D(pool_size=pool_size)(conv3)
	# conv3_out = GlobalMaxPool1D()(conv3)
	merged = concatenate([conv1_out,conv2_out],axis=1)
	
	dropout = Dropout(dropout_rate)(merged)
	output = Dense(nclasses, activation="sigmoid")(dropout)

	model = Model(inputs = [input], outputs=[output])

	return model

