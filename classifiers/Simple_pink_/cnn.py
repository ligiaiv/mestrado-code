from tensorflow import keras

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
import os
from helper import DataHandler, get_accuracy


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
# word_index = dataHandler.word_index
X = dataHandler.data
Y = dataHandler.labels_array


Y = Y[:,0].reshape(Y.shape[0],1)
print(Y)
# quit() 

DS_size = len(Y)
rand_index = np.arange(DS_size)
np.random.shuffle(rand_index)
X = X[rand_index]
Y = Y[rand_index]

split1 = round(0.7*DS_size)
x_train,y_train = X[:split1],Y[:split1]
x_test,y_test = X[split1:],Y[split1:]

print('Build model...')

input_ = Input(shape = (maxlen,))
embedding_layer = Embedding(embedding_matrix.shape[0],
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=True)(input_)
# embedding_layer = Embedding(max_features, embedding_size, input_length=maxlen)(input)
# embedding_layer.trainable = train_embedding
conv1= Conv1D(filters,
				 3,
				 padding='valid',
				 activation='relu',
				 strides=1)(embedding_layer)
dropout = Dropout(0.5)(conv1)
conv1_out = MaxPooling1D(pool_size=3)(dropout)

# dense1 = Dense(5, activation='relu')(embedding)
# dropout = Dropout(0.5)(dense1)
flat = Flatten()(conv1_out)

dense2 = Dense(1)(flat)
activation = Activation('sigmoid')(dense2)

model = Model(inputs = [input_], outputs=[activation])
print("Y_train",y_train.shape)
print("X_train",x_train.shape)

model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy']
				)

print(model.summary())
# quit()
print('Train...')

train_info = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1)
# print(history.__dict__)
acc_train = train_info.history["accuracy"]
acc_val = train_info.history["val_accuracy"]
# quit()
# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
y_predict = model.predict(x_test,batch_size=batch_size)
acc,prec,rec,f1,confmat=get_accuracy(y_predict,y_test)
# test_metrics = np.hstack((test_metrics,np.expand_dims(np.array([acc,prec,rec,f1]),axis = 1)))
# train_val_metrics = np.dstack((train_val_metrics,np.array([acc_train,acc_val])))

# print('Test score:', score)
print('Test acc,prec,rec,f1:', acc,prec,rec,f1)
# print("history",history.__dict__)