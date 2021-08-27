from __future__ import print_function
import tensorflow as tf
# print(tf.__version__)
# print(tf.keras.__version__)
import ktrain
from ktrain import text
import os,json
import pandas as pd
import numpy as np

# from keras.models import Sequential, Model
# from keras.layers import Input, Dense, Dropout, Activation, Concatenate
# from keras.layers import Embedding
# from keras.layers import LSTM
# from keras.layers import Conv1D, MaxPooling1D
# from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping
# from keras.datasets import imdb
from keras.utils import to_categorical

from sklearn.model_selection import KFold

from helper import DataHandler, get_accuracy
from datetime import datetime, timezone
import models as myModels

print("Reading variables file...")
options = json.load(open("dePelle/variables.json", "r"))

DATAFILE = "OffComBR-master/OffComBR2.arff"

# Embedding
max_features = 20000 #vocab size
maxlen = options["seq_len"] # fiz histogtama do tamanho dos tweets
embedding_size = options["emb_dim"]
embedding_type = options["emb_type"]
train_embedding = options["train_emb"]
bidirectional = options["bidirectional"]
n_epochs = options["num_epochs"]
# max_vocab_size = 10000

# Convolution
kernel_size = 5
filters = 100
pool_size = 2

# LSTM
lstm_output_size = options["hidden_lstm_dim"]

# Training
batch_size = options["batch_size"]
epochs = options["num_epochs"]
print("Epochs",epochs)


print("Reading data file ...")
# path = os.getcwd().split('/')
# path.pop()
# path = '/'.join(path)+'/'

path = os.getcwd()
print(path)
dataHandler = DataHandler()
dataHandler.readDataFile(DATAFILE,maxlen,max_features,DELI="'")

dataHandler.readEmbedding('{}.{}d.txt'.format(embedding_type,embedding_size),path+"/Embeddings")
dataHandler.createMatrix(embedding_size)


X = dataHandler.data
Y = dataHandler.labels_array
print("Y",Y)
# print("Y.shape",Y.shape)
# quit()
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)

print('Build model...')

model = myModels.LSTM_Model(len(dataHandler.classes),maxlen,max_features,embedding_size,lstm_output_size,train_embedding,bidirectional)
# model = myModels.FF_Model(len(dataHandler.classes),maxlen,max_features,embedding_size,50,5,train_embedding)

kf = KFold(n_splits=options["kfold"])

model.save_weights('model.h5')


results = pd.DataFrame(columns=["fold","test_acc","train_data"])
test_metrics = np.ndarray((4,0))
train_val_metrics = np.ndarray((2,n_epochs,0))

print("X,Y",X.shape,Y.shape)
k = 0
random_order=np.arange(len(X))
np.random.shuffle(random_order)
print(random_order)
X = X[random_order]
Y = Y[random_order]
if len(dataHandler.classes) == 1:
	loss = 'binary_crossentropy'
else:
	loss = 'categorical_crossentropy'
print(loss)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
for train_index, test_index in kf.split(X):
	k+=1
	print("\n\t-----------{} fold-----------\n".format(k))
	x_train, x_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]
	print(x_train.shape,)
	model.load_weights('model.h5')

	model.compile(loss=loss,
				optimizer='adam',
				metrics=['accuracy']
				)

	print('Train...')
	train_info = model.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=epochs,
			validation_split=0.1,
			callbacks=[es])
	acc_train = train_info.history["accuracy"]
	acc_val = train_info.history["val_accuracy"]

	epoch_diff = n_epochs- len(acc_train)

	acc_train_val = np.hstack((np.array([acc_train,acc_val]),np.zeros((2,epoch_diff))))
	# quit()
	# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
	y_predict = model.predict(x_test,batch_size=batch_size)
	acc,prec,rec,f1,confmat=get_accuracy(y_predict,y_test)
	test_metrics = np.hstack((test_metrics,np.expand_dims(np.array([acc,prec,rec,f1]),axis = 1)))
	train_val_metrics = np.dstack((train_val_metrics,acc_train_val))

	# print('Test score:', score)
	print('Test accuracy:', acc)
	# print("history",history.__dict__)

results_dict = {
    "train_val":train_val_metrics.tolist(),
    "test":test_metrics.tolist(),
    "confmat":confmat.tolist(),
    "options":options
}
now = datetime.now(timezone.utc)
current_time = now.strftime("%m-%d-%Y__%H:%M:%S")
print("Writing results to file...")
with open("dePelle/results/out_"+current_time+".json", "w") as outfile:
    json.dump(results_dict, outfile)
print("Done!")
