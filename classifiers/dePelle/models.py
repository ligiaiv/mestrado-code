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
from keras.layers import Input, Dense, Dropout, Activation, Concatenate
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers.merge import concatenate

# from keras.datasets import imdb
# from keras.utils import to_categorical

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
 

def LSTM_Model(nclasses,maxlen,max_features,embedding_size,lstm_output_size,train_embedding=True, bidirectional = False):
    input = Input(shape = (maxlen,))
    embedding = Embedding(max_features, embedding_size, input_length=maxlen)(input)
    embedding.trainable = train_embedding    
    if bidirectional:
        lstm = Bidirectional(LSTM(lstm_output_size))(embedding)
    else:
        lstm = LSTM(lstm_output_size)(embedding)
    dense = Dense(nclasses)(lstm)
    activation = Activation('sigmoid')(dense)

    model = Model(inputs = [input], outputs=[activation])

    return model

def CNN_LSTM_Model(maxlen,max_features,embedding_size,lstm_output_size,filters,train_embedding):
    input = Input(shape = (maxlen,))
    embedding = Embedding(max_features, embedding_size, input_length=maxlen)(input)
    embedding.trainable = train_embedding
    conv1= Conv1D(filters,
                    3,
                    padding='valid',
                    activation='relu',
                    strides=1)(embedding)
    # conv1_out = MaxPooling1D(pool_size=pool_size)(conv1)

    conv2= Conv1D(filters,
                    4,
                    padding='valid',
                    activation='relu',
                    strides=1)(embedding)
    # conv2_out = MaxPooling1D(pool_size=pool_size)(conv2)

    conv3= Conv1D(filters,
                    5,
                    padding='valid',
                    activation='relu',
                    strides=1)(embedding)
    # conv3_out = MaxPooling1D(pool_size=pool_size)(conv3)

    merged = concatenate([conv1,conv2,conv3],axis=1)
    lstm = LSTM(lstm_output_size)(merged)
    dense = Dense(3)(lstm)
    activation = Activation('sigmoid')(dense)

    model = Model(inputs = [input], outputs=[activation])

    return model

