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
from keras.utils import plot_model

from keras.engine.topology import Layer, InputSpec
from keras import initializers

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

# for the plot
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300  # spacy has glove with 300-dimensional embeddings
VALIDATION_SPLIT = 0.2
MODEL_PATH = 'models/bidirectional_lstm/'

import preprocess_data

print('loading the data')
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

    return model


def my_confusion_matrix(y_true, y_pred, n_classes):
    """This function returns the confusion matrix tolerant to classes without true samples"""
    from scipy.sparse import coo_matrix
    CM = coo_matrix((np.ones(y_true.shape[0], dtype=np.int), (y_true, y_pred)),
                    shape=(n_classes, n_classes)
                    ).toarray()
    return CM

def plot_confusion(confusion, label_values, path):
    df_cm = pd.DataFrame(confusion, index=label_values, columns=label_values)
    #sn.set(font_scale=1.4)  # for label size
    fig = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    fig.get_figure().savefig(path + '.png')
    plt.clf()

n_folds = 10
# skf will profide indices to iterate over in each fold
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

f1_scores = np.zeros((n_folds))
confusion_sum = np.zeros((labels.shape[1], labels.shape[1]))

for i, (train, test) in enumerate(skf.split(np.zeros((data.shape[0],)), np.zeros((data.shape[0],)))):
    model = create_model()
    if i == 0:
        # first iteration
        model.summary()
        # this requires graphviz binaries also
        plot_model(model, to_file=MODEL_PATH + 'model.png', show_shapes=True)

    print("Running Fold", i + 1, "/", n_folds)

    model.fit(data[train], labels[train], validation_data=(
        data[test], labels[test]), epochs=10, batch_size=50)

    # generate confusion matrix
    y_pred = model.predict(data[test])
    confusion = my_confusion_matrix(labels[test].argmax(
        axis=1), y_pred.argmax(axis=1), labels.shape[1])

    # compute f1 score weighted by support
    f1 = f1_score(labels[test].argmax(axis=1),
                  y_pred.argmax(axis=1), average='weighted')
    print('f1 at fold ' + str(i + 1) + ': ' + str(f1))
    
    f1_scores[i] = f1
    confusion_sum = np.add(confusion_sum, confusion)

    plot_confusion(confusion, intents, MODEL_PATH + 'confusion_iteration_' + str(i + 1))


print('mean f1 score: ' + str(f1_scores.mean()))
plot_confusion(confusion_sum, intents, MODEL_PATH + 'confusion_sum')

print("Now training on full dataset, no validation")
model.fit(data, labels, nb_epoch=10, batch_size=50)

model.save(MODEL_PATH + 'model_bidirectional_LSTM.h5')
