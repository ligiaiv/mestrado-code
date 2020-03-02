import tensorflow as tf
# print(tf.__version__)
# print(tf.keras.__version__)
import ktrain
from ktrain import text
import os
import pandas as pd
import numpy as np

print("Reading data file ...")
path = os.getcwd().split('/')
path.pop()
path = '/'.join(path)+'/'
df = pd.read_csv(path+'Datasets/'+"labeled_data.csv")
labels_words = df["class"].values
sexism_array = (labels_words=="sexism").astype(int)
racism_array= (labels_words=="racism").astype(int)
none_array = (labels_words=="none").astype(int)
labels_array = np.stack([sexism_array,racism_array,none_array])
print(labels_array)
print("df",df.shape)
print("arr",labels_array.shape)
new_df = pd.concat([df,pd.DataFrame(data = labels_array.T, columns=["sexism","racism","none"])],axis=1)

print(new_df.columns)
print(new_df)

(x_train, y_train), (x_test, y_test), preproc  = text.texts_from_df(new_df,"text",["sexism","racism","none"],preprocess_mode="bert")

model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model,train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=6)

learner.fit_onecycle(2e-5, 1)