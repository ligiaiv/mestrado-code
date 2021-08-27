from keras.layers import Input, Dense, Dropout, Activation, Concatenate, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
class LSTM_model():
    def __init__(self,parameters):
        self.parameters = parameters
        self.maxlen = parameters['maxlen']
        self.embedding_matrix = parameters['embedding_matrix']
        self.EMBEDDING_DIM = parameters["EMBEDDING_DIM"] 
        self.dropout_rate = parameters["dropout_rate"]
        self.lstm_output_size = parameters["lstm_output_size"]
    def model(self):
        print('Build model...')

        input_ = Input(shape = (self.maxlen,))
        embedding_layer = Embedding(self.embedding_matrix.shape[0],
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.maxlen,
                                    trainable=True)(input_)
        # embedding_layer = Embedding(max_features, embedding_size, input_length=maxlen)(input)
        # embedding_layer.trainable = train_embedding
        lstm= LSTM(self.lstm_output_size)(embedding_layer)
        dropout = Dropout(self.dropout_rate)(lstm)


        # dense1 = Dense(5, activation='relu')(embedding)
        # dropout = Dropout(0.5)(dense1)
        dense2 = Dense(1)(dropout)
        activation = Activation('sigmoid')(dense2)


        self.model = Model(inputs = [input_], outputs=[activation])
        return self.model