import numpy as np

import sys
import os

import spacy

# uncomment this line to change backend. Faster seems to be TensorFlow
#os.environ['KERAS_BACKEND']='theano'
from keras.models import load_model

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input
from keras.layers import Conv1D, MaxPooling1D, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300 # spacy has glove with 300-dimensional embeddings
VALIDATION_SPLIT = 0.2

import preprocess_data

data_train = preprocess_data.get_train_data(preprocess_data.load_expressions())

texts = []
labels = []

intents = {}
# translation from intent value (string) to int (index)
for index, value in enumerate(preprocess_data.load_intents()):
    intents[value] = index

nlp = spacy.load('en')

data = np.zeros((len(data_train), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
for idx, (text, intent) in enumerate(data_train):
    # convert from sentences to glove matrix
    # parse the sentence
    doc = nlp(text)
    for index, word in enumerate(doc):
        data[idx][index] = word.vector

    labels.append(intents[intent])

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# shuffle the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# split the data in train and validation
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Traing and validation set number of sentences for each intent')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

# sequence_input is a matrix of glove vectors (one for each input word)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
l_lstm = Bidirectional(LSTM(100))(sequence_input)
preds = Dense(len(intents), activation='softmax')(l_lstm)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)

model.save('model_bidirectional_LSTM.h5')








"""
# TODO fix dimensionality to test
# Attention GRU network		  
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],1))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    #embedding_vector = embeddings_index.get(word)
    embedding_vector = nlp(word).vector
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer()(l_gru)
preds = Dense(len(intents), activation='softmax')(l_att)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - attention GRU network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)

model.save('model_attention_gru.h5')
"""