class LSTM_model():
    def __init__(self,parameters):
        self.parameters = parameters
    
    def model():
        print('Build model...')

        input_ = Input(shape = (maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0],
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=True)(input_)
        # embedding_layer = Embedding(max_features, embedding_size, input_length=maxlen)(input)
        # embedding_layer.trainable = train_embedding
        lstm= LSTM(lstm_output_size)(embedding_layer)
        dropout = Dropout(0.5)(lstm)


        # dense1 = Dense(5, activation='relu')(embedding)
        # dropout = Dropout(0.5)(dense1)
        dense2 = Dense(1)(dropout)
        activation = Activation('sigmoid')(dense2)


        self.model = Model(inputs = [input_], outputs=[activation])
        return self.model