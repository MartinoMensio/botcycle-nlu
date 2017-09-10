import numpy as np

import sys
import os

import spacy

# uncomment this line to change backend. Faster seems to be TensorFlow
# os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import load_model

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input
from keras.layers import Conv1D, MaxPooling1D, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras.engine.topology import Layer, InputSpec
from keras import initializers

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# for the plot
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300  # spacy has glove with 300-dimensional embeddings
VALIDATION_SPLIT = 0.2

import preprocess_data

data_train = preprocess_data.get_train_data(preprocess_data.load_expressions())

texts = []
labels = []

intents = preprocess_data.load_intents()
intents_lookup = {}
# translation from intent value (string) to int (index)
for index, value in enumerate(intents):
    intents_lookup[value] = index

nlp = spacy.load('en')

data = np.zeros((len(data_train), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
for idx, (text, intent) in enumerate(data_train):
    # convert from sentences to glove matrix
    # parse the sentence
    doc = nlp(text)
    for index, word in enumerate(doc):
        data[idx][index] = word.vector

    labels.append(intents_lookup[intent])

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# shuffle the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

print('Number of sentences for each intent')
print(intents)
print(labels.sum(axis=0))

def create_model():
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = Bidirectional(LSTM(100))(sequence_input)
    preds = Dense(len(intents), activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])

    model.summary()
    return model


n_folds = 3
# skf will profide indices to iterate over in each fold
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

results = []

for i, (train, test) in enumerate(skf.split(np.zeros((len(data_train), MAX_SEQUENCE_LENGTH * EMBEDDING_DIM)), np.zeros((len(data_train),)))):
    print("Running Fold", i + 1, "/", n_folds)

    model = create_model()

    model.fit(data[train], labels[train], validation_data=(data[test], labels[test]), nb_epoch=10, batch_size=50)
    
    # generate confusion matrix
    y_pred = model.predict(data[test])
    confusion = confusion_matrix(labels[test].argmax(axis=1), y_pred.argmax(axis=1))
    results.append(confusion)
    
    # plot
    # TODO add params index = intents, columns = intents when sure that 10-fold has minimum one item for class
    df_cm = pd.DataFrame(confusion, index = intents, columns = intents)
    sn.set(font_scale=1.4)#for label size
    fig = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    fig.get_figure().savefig('conf_' + str(i + 1) + '.png')
    plt.clf()

# print(results)

print("Now training on full dataset, no validation")
model.fit(data, labels, nb_epoch=10, batch_size=50)

model.save('model_bidirectional_LSTM.h5')