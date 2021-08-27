from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Concatenate
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate

import pandas as pd
import numpy as np

emb_dim = 1024 
batch_size = 64
epochs = 10
filters = 10
lstm_output_size = 10
print("Reading data file ...")
path = os.getcwd()+"/"

def load_file(file_name):
    df_in = pd.read_csv(file_name)
    print(df_in.columns)
    return embedding,labels

NAME = "fortuna_labeled_data.csv_vectorized.csv"
data,label = load_file(NAME)

DS_size = len(label)
rand_index = np.arange(DS_size)
np.random.shuffle(rand_index)
data = data[rand_index]
label = label[rand_index]

split1 = round(0.7*DS_size)
x_train,y_train = data[:split1],label[:split1]
x_test,y_test = data[split1:],label[split1:]

print('Build model...')

input_ = Input(shape = (emb_dim,))
embedding = Embedding(max_features, embedding_size, input_length=maxlen)(input)
embedding.trainable = train_embedding
lstm= LSTM(lstm_output_size)
dropout = Dropout(0.5)(conv1)
conv1_out = MaxPooling1D(pool_size=pool_size)(dropout)

# dense1 = Dense(5, activation='relu')(embedding)
# dropout = Dropout(0.5)(dense1)
dense2 = Dense(1)(conv1_out)
activation = Activation('sigmoid')(dense2)

model = Model(inputs = [input_], outputs=[activation])
print("Y_train",y_train.shape)
print("X_train",x_train.shape)
model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy']
				)

print('Train...')
print(y_train.shape)
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