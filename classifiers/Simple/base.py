from tensorflow import keras

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os
from helper import DataHandler, get_accuracy
import models


import pandas as pd
import numpy as np
embedding_type = "glove"
EMBEDDING_DIM = 50
maxlen = 40
MAX_VOCAB_SIZE=20000
# emb_dim = 1024 
batch_size = 64
epochs = 10
filters = 10
lstm_output_size = 20
print("Reading data file ...")
path = os.getcwd()+"/"

DATAFILE = "fortuna_labeled_data_preprocessed.csv"
path = os.getcwd()
print(path)
dataHandler = DataHandler()
dataHandler.readDataFile(DATAFILE,maxlen,MAX_VOCAB_SIZE)

dataHandler.readEmbedding('{}_s{}.txt'.format(embedding_type,EMBEDDING_DIM),os.path.join(path, "Embeddings"))
dataHandler.createMatrix(EMBEDDING_DIM)
embedding_matrix = dataHandler.embedding_matrix

X = dataHandler.data
Y = dataHandler.labels_array


Y = Y[:,0].reshape(Y.shape[0],1)
print(Y)

# Make random
DS_size = len(Y)
rand_index = np.arange(DS_size)
np.random.shuffle(rand_index)
X = X[rand_index]
Y = Y[rand_index]

split1 = round(0.7*DS_size)
x_train,y_train = X[:split1],Y[:split1]
x_test,y_test = X[split1:],Y[split1:]

print('Build model...')
model = models.LSTM_model({})

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
print("Y_train",y_train.shape)
print("X_train",x_train.shape)

model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy']
				)

print(model.summary())

print('Train...')
best_model = load_model('best_model.h5')
train_info = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1)
acc_train = train_info.history["accuracy"]
acc_val = train_info.history["val_accuracy"]

y_predict = model.predict(x_test,batch_size=batch_size)
acc,prec,rec,f1,confmat=get_accuracy(y_predict,y_test)

print('Test acc,prec,rec,f1:', acc,prec,rec,f1)