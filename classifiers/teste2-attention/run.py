from __future__ import print_function
import tensorflow as tf
# print(tf.__version__)
# print(tf.keras.__version__)
import ktrain
from ktrain import text
import os,json
import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Concatenate
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
import keras.backend as K
from keras.callbacks import EarlyStopping

from keras.datasets import imdb
from keras.utils import to_categorical, plot_model

from sklearn.model_selection import KFold

from helper import DataHandler, get_accuracy,LstmOptions, data_augmentation,get_word_probability,add_and_shuffle
from datetime import datetime, timezone

import models as myModels
import tensorflow as tf
print("Reading variables file...")
options = json.load(open("teste2-attention/variables.json", "r"))



# Embedding
MAX_FEATURES = 20000 #vocab size
MAX_LEN = options["seq_len"] # fiz histogtama do tamanho dos tweets
EMBEDDING_SIZE = options["emb_dim"]
EMBEDDING_TYPE = options["emb_type"]
TRAIN_EMBEDDING = not options["freeze_emb"]
EARLY_STOPPING = options["early_stopping"]
SPLIT = options["split"]
# max_vocab_size = 10000

# Convolution
kernel_size = 5
filters = options["filters"]
pool_size = options["pool_size"]
dropout_rate = options["dropout"]

architecture = options["architecture"]
# LSTM
BIDIRECTIONAL = options["bidirectional"]
ATTENTION_SIZE = options["attention_size"]
lstm_output_size = options["hidden_lstm_dim"]
extra_training_data = options["extra_training_data"]

# Training
batch_size = options["batch_size"]
epochs = options["num_epochs"]

DATAFILE = options["datafile"]
#"labeled_data.csv"
print("Reading data file ...")
# path = os.getcwd().split('/')
# path.pop()
# path = '/'.join(path)+'/'

path = os.getcwd()
print(path)
dataHandler = DataHandler()
if options["extra_training_data"]:
	dataHandler.processData(DATAFILE,MAX_LEN,MAX_FEATURES,options["extra_datafile"])
else:
	dataHandler.processData(DATAFILE,MAX_LEN,MAX_FEATURES)	
dataHandler.processData(DATAFILE,MAX_LEN,MAX_FEATURES)
n_classes = dataHandler.n_classes
dataHandler.readEmbedding('{}.{}d.txt'.format(EMBEDDING_TYPE,EMBEDDING_SIZE),path+"/Embeddings",EMBEDDING_SIZE)
dataHandler.createMatrix(EMBEDDING_SIZE)
	



# quit()
print('Build model...')
if architecture =="lstm":
	model = myModels.LSTM_Model(n_classes,MAX_FEATURES,LstmOptions(options))
elif architecture == "lstm-attention":
	model = myModels.LSTM_Attention_Model(n_classes,MAX_LEN,MAX_FEATURES,EMBEDDING_SIZE,ATTENTION_SIZE,lstm_output_size,TRAIN_EMBEDDING,BIDIRECTIONAL)
elif architecture == "cnn-lstm":
	model = myModels.CNN_LSTM_Model(n_classes,MAX_LEN,MAX_FEATURES,EMBEDDING_SIZE,lstm_output_size,filters,pool_size,dropout_rate,TRAIN_EMBEDDING)
elif architecture == "lstm-cnn":
	model = myModels.CNN_LSTM_Model(n_classes,MAX_LEN,MAX_FEATURES,EMBEDDING_SIZE,lstm_output_size,filters,pool_size,dropout_rate,TRAIN_EMBEDDING)
elif architecture == "cnn":
	model = myModels.CNN_Model(n_classes,MAX_LEN,MAX_FEATURES,EMBEDDING_SIZE,filters,pool_size,dropout_rate,TRAIN_EMBEDDING)
else:
	print("ERROR, arquitecture not found: {}".format(architecture))
model_sumary = model.summary()
# plot_model(model, show_shapes=True, dpi=90)

# kf = KFold(n_splits=options["kfold"])

model.save_weights('model.h5')


results = pd.DataFrame(columns=["fold","test_acc","train_data"])
test_metrics = np.ndarray((4,0))
train_val_metrics = np.ndarray((2,options["num_epochs"],0))


X = dataHandler.data
Y = dataHandler.labels_array

print("X,Y",X.shape,Y.shape)
k = 0
X,Y = add_and_shuffle([X],[Y])
# random_order=np.arange(len(X))
# np.random.shuffle(random_order)
# print(random_order)
# X = X[random_order]
# Y = Y[random_order]

if extra_training_data:
	extraX = dataHandler.extra_data
	extraY = dataHandler.extra_labels_array

	extraX,extraY = add_and_shuffle([extraX],[extraY])

attention_parameters = {}
train_test_numbers = None
CALLBACKS = None
if EARLY_STOPPING is not None:
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOPPING)
	CALLBACKS = [es]

if options["use_kfold"]:
	kf = KFold(n_splits=options["kfold"])
	# KFOLD = 2
	# splitting dataset

	for train_index, test_index in kf.split(X):
		print("\n\t-----------{} fold-----------\n".format(k))
		x_train, x_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		if extra_training_data:
			x_train,y_train = add_and_shuffle([x_train,extraX],[y_train,extraY])
		train_test_numbers = [len(x_train),len(x_test)]
		print(x_train.shape,y_train.shape)
		if options["data_augmentation"]:
			prob_table = get_word_probability(X,Y)
			if options["prob_deletion"]:
				x_train,y_train = data_augmentation(x_train,y_train,options["data_augmentation"], get_word_probability(x_train,y_train))
			else:
				x_train,y_train = data_augmentation(x_train,y_train,options["data_augmentation"])

		# print(x_train.shape,y_train.shape)
		# quit()
		model.load_weights('model.h5')

		model.compile(loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy']
					)

		print('Train...')
		train_info = model.fit(x_train, y_train,
				batch_size=batch_size,
				epochs=epochs,
				validation_split=0.1,
				callbacks=CALLBACKS)
		# print(history.__dict__)
		acc_train = train_info.history["accuracy"]
		acc_val = train_info.history["val_accuracy"]

		epoch_diff = epochs- len(acc_train)

		# quit()
		# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
		acc_train_val = np.hstack((np.array([acc_train,acc_val]),np.zeros((2,epoch_diff))))
		# quit()
		# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
		y_predict = model.predict(x_test,batch_size=batch_size)
		acc,prec,rec,f1,confmat=get_accuracy(y_predict,y_test)
		test_metrics = np.hstack((test_metrics,np.expand_dims(np.array([acc,prec,rec,f1]),axis = 1)))
		train_val_metrics = np.dstack((train_val_metrics,acc_train_val))

		# y_predict = model.predict(x_test,batch_size=batch_size)
		# acc,prec,rec,f1,confmat=get_accuracy(y_predict,y_test)
		# test_metrics = np.hstack((test_metrics,np.expand_dims(np.array([acc,prec,rec,f1]),axis = 1)))
		# train_val_metrics = np.dstack((train_val_metrics,np.array([acc_train,acc_val])))

		# print('Test score:', score)
		print('Test accuracy:', acc)
		# print("history",history.__dict__)

		if "attention" in architecture:
			intermediate_layer_model = K.function([model.inputs],[ model.get_layer("attention_softmax").output])
			attention_weights = intermediate_layer_model(x_test)
			print("attention_weights\n{}\ntest_data{}".format(np.array(attention_weights).shape,np.array(x_test).shape))
		

			attention_parameters[k] = {
					"sentences":x_test.tolist(),
					"attention_weights":np.squeeze(np.array(attention_weights)).tolist(),
					"scores":y_predict.tolist()
					
				}

		# for layer in model.layers:
		# 	if layer.name is "attention_softmax":

		# 		print(layer.__dict__.keys())
		# 		print(layer.name)
		# 	print('\n-----------------------------------------------------\n')
else:
	data_length = len(X)
	middle = int(SPLIT*data_length)
	print("\n\t-----------Split {}:{}-----------\n".format(SPLIT, 1-SPLIT))
	x_train, x_test = X[:middle], X[middle:]
	y_train, y_test = Y[:middle], Y[middle:]

	if extra_training_data:
		x_train,y_train = add_and_shuffle([x_train,extraX],[y_train,extraY])

	if options["data_augmentation"]:
		x_train,y_train = data_augmentation(x_train,y_train,options["data_augmentation"])
	train_test_numbers = [len(x_train),len(x_test)]


	print(x_train.shape,)
	model.load_weights('model.h5')

	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy']
				)

	print('Train...')
	train_info = model.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=epochs,
			validation_split=0.1,
			callbacks=CALLBACKS)
	# print(history.__dict__)
	acc_train = train_info.history["accuracy"]
	acc_val = train_info.history["val_accuracy"]

	epoch_diff = epochs- len(acc_train)

	# quit()
	# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
	acc_train_val = np.hstack((np.array([acc_train,acc_val]),np.zeros((2,epoch_diff))))
	# quit()
	# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
	y_predict = model.predict(x_test,batch_size=batch_size)
	acc,prec,rec,f1,confmat=get_accuracy(y_predict,y_test)
	test_metrics = np.hstack((test_metrics,np.expand_dims(np.array([acc,prec,rec,f1]),axis = 1)))
	train_val_metrics = np.dstack((train_val_metrics,acc_train_val))

	# y_predict = model.predict(x_test,batch_size=batch_size)
	# acc,prec,rec,f1,confmat=get_accuracy(y_predict,y_test)
	# test_metrics = np.hstack((test_metrics,np.expand_dims(np.array([acc,prec,rec,f1]),axis = 1)))
	# train_val_metrics = np.dstack((train_val_metrics,np.array([acc_train,acc_val])))

	# print('Test score:', score)
	print('Test accuracy:', acc)
	# print("history",history.__dict__)

	if "attention" in architecture:
		intermediate_layer_model = K.function([model.inputs],[ model.get_layer("attention_softmax").output])
		attention_weights = intermediate_layer_model(x_test)
		print("attention_weights\n{}\ntest_data{}".format(np.array(attention_weights).shape,np.array(x_test).shape))
	

		attention_parameters[k] = {
				"sentences":x_test.tolist(),
				"attention_weights":np.squeeze(np.array(attention_weights)).tolist(),
				"scores":y_predict.tolist()
				
			}

# print("test_metrics",test_metrics.shape)
print("Test Average: ",np.average(test_metrics,axis=1))

results_dict = {
    "train_val":train_val_metrics.tolist(),
    "test":test_metrics.tolist(),
    "confmat":confmat.tolist(),
    "options":options,
	"model_sumary": model_sumary,
	"data_augmentation": options["data_augmentation"],
	"train_test": train_test_numbers
}
now = datetime.now(timezone.utc)
current_time = now.strftime("%m-%d-%Y__%H:%M:%S")
print("Writing results to file...")
with open("teste2-attention/results/out_"+current_time+".json", "w") as outfile:
    json.dump(results_dict, outfile)
if "attention" in architecture:
	with open("teste2-attention/word_result.json",'w') as outfile:
		json.dump(attention_parameters,outfile)
print("Done!")
