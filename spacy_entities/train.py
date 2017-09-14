#!/usr/bin/env python
# coding: utf8
"""
This code trains the NER with entities from the exported wit_data.
The entity types found in the wit_data are added to an existing pre-trained NER
model ('en').

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.
After training the model, it is saved to a directory.
Documentation:
* Training the Named Entity Recognizer: https://spacy.io/docs/usage/train-ner
* Saving and loading models: https://spacy.io/docs/usage/saving-loading

Example adapted from https://github.com/explosion/spaCy/blob/master/examples/training/train_new_entity_type.py
"""
from __future__ import unicode_literals, print_function
import gc
import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from keras.utils import plot_model

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import random

import spacy
from spacy.gold import GoldParse
from spacy.tagger import Tagger

import tag_data

def get_entities_lookup(entities):
    """From a list of entities gives back a dictionary that maps the value to the index in the original list"""
    entities_lookup = {}
    for index, value in enumerate(entities):
        entities_lookup[value] = index
    return entities_lookup

def kfold(model_name, n_folds, data, entities_lookup):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    f1_scores = np.zeros((n_folds))
    # dimensions +1 because also no entity class (indexed last)
    extended_entities_lookup = entities_lookup.copy()
    extended_entities_lookup['NONE'] = len(entities_lookup)
    confusion_sum = np.zeros((len(extended_entities_lookup), len(extended_entities_lookup)))

    for i, (train, test) in enumerate(skf.split(np.zeros((data.shape[0],)), np.zeros((data.shape[0],)))):
        print("Running Fold", i + 1, "/", n_folds)
        nlp = spacy.load(model_name)
        # passing None because don't want to save every fold model
        train_ner(nlp, data[train], [key for key in entities_lookup], None)
        for test_data in data[test]:
            # test_data is like {'text':'', 'entities':{'entity','value','start','end'}}
            doc = nlp(test_data['text'])
            # true_ents maps 'start_index:end_index' of entity to entity name, e.g. {'10:16': 'LOCATION'}
            true_ents = {'{}:{}'.format(true_ent['start'], true_ent['end']): true_ent['entity'].upper() for true_ent in test_data['entities']}
            
            for predicted_ent in doc.ents:
                # on match an entry from true_ents is removed (see below computation of false negatives)
                true_ent = true_ents.pop('{}:{}'.format(predicted_ent.start_char, predicted_ent.end_char), 'NONE')
                # the fallback parameter is needed in case unexpected types of entities are found
                predicted_class = extended_entities_lookup.get(predicted_ent.label_, extended_entities_lookup['NONE'])
                true_class = extended_entities_lookup[true_ent]
                # actual class indexes the rows while predicted class indexes the columns
                confusion_sum[true_class, predicted_class] += 1
                if predicted_class is not true_class:
                    # TODO careful to boundaries
                    print('wrong prediction in "' + str(doc) + '". "' + str(predicted_ent.text) + '" was classified as', predicted_class, 'but was', true_class)

            for false_negative in true_ents.values():
                print('false negative found: ' + false_negative)
                confusion_sum[extended_entities_lookup[false_negative], extended_entities_lookup['NONE']] += 1

            # now also add some NONE->NONE values, one for each sentence? TODO
            confusion_sum[extended_entities_lookup['NONE'], extended_entities_lookup['NONE']] += 1
        
        # free memory before death
        del nlp
        gc.collect()
    
    print('final confusion matrix:\n', confusion_sum)
    return confusion_sum, extended_entities_lookup

def train_ner(nlp, data, entity_names, output_directory):
    train_data = list(map(lambda x: (x['text'], list(map(lambda ent: (
        ent['start'], ent['end'], ent['entity'].upper()), x['entities']))), data))

    for entity_name in entity_names:
            nlp.entity.add_label(entity_name)

    # Add new words to vocab
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]
    random.seed(0)
    # You may need to change the learning rate. It's generally difficult to
    # guess what rate you should set, especially when you have limited data.
    nlp.entity.model.learn_rate = 0.001

    # average of last iterations, for printing something
    tot_loss = 0.
    for itn in range(1000):
        random.shuffle(train_data)
        loss = 0.
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            nlp.tagger(doc)
            # As of 1.9, spaCy's parser now lets you supply a dropout probability
            # This might help the model generalize better from only a few
            # examples.
            loss += nlp.entity.update(doc, gold, drop=0.9)

        tot_loss += loss / len(train_data)

        if loss == 0:
            break

        if (itn + 1) % 50 == 0:
            print('loss: ' + str(tot_loss / 50))
            tot_loss = 0.
    # This step averages the model's weights. This may or may not be good for
    # your situation --- it's empirical.
    nlp.end_training()

    # Save to directory
    if output_directory:
        nlp.save_to_directory(output_directory)

# TODO remove duplicated code, same as intent model utils
def plot_confusion(confusion, label_values, path):
    df_cm = pd.DataFrame(confusion, index=label_values, columns=label_values)
    df_cm.columns.name = 'predict'
    df_cm.index.name = 'actual'
    #sn.set(font_scale=1.4)  # for label size
    fig = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    fig.get_figure().savefig(path + '.png')
    plt.clf()


def main(n_folds='10', model_name='en', output_directory='models'):
    print("Loading initial model", model_name)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    expressions = tag_data.load_expressions()
    data = tag_data.tag(expressions)
    data = np.array(data)

    entities = tag_data.load_entities()
    entities = list(map(str.upper,entities))
    entities_lookup = get_entities_lookup(entities)

    if True:
        n_folds = int(n_folds)
        confusion, extended_entities_lookup = kfold(model_name, n_folds, data, entities_lookup)
        plot_confusion(confusion, [key for key in extended_entities_lookup], output_directory + '/confusion_' + str(n_folds) + 'folds')

        tps = np.diagonal(confusion)
        supports = confusion.sum(axis=1)
        precisions = np.divide(tps, confusion.sum(axis=0))
        recalls = np.divide(tps, supports)
        f1s = 2*((precisions*recalls)/(precisions+recalls))
        f1 = np.average(f1s, weights=supports)
        print('f1 score: ', f1)

    # now train on full data
    nlp = spacy.load('en')
    train_ner(nlp, data, entities, output_directory)

if __name__ == '__main__':
    import plac
    plac.call(main)
