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

import random
from pathlib import Path
import random

import spacy
from spacy.gold import GoldParse
from spacy.tagger import Tagger

import tag_data


def train_ner(nlp, train_data, output_dir):
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
    if output_dir:
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.save_to_directory(output_dir)


def main(model_name='en', output_directory='model'):
    print("Loading initial model", model_name)
    nlp = spacy.load(model_name)
    if output_directory is not None:
        output_directory = Path(output_directory)

    expressions = tag_data.load_expressions()
    tagged = tag_data.tag(expressions)

    train_data = list(map(lambda x: (x['text'], list(map(lambda ent: (
        ent['start'], ent['end'], ent['entity'].upper()), x['entities']))), tagged))

    for entity in tag_data.load_entities():
        nlp.entity.add_label(entity.upper())
    train_ner(nlp, train_data, output_directory)

    # Test that the entity is recognized
    doc = nlp('I want to go from piazza statuto to via garibaldi')
    print("Ents in 'I want to go from piazza statuto to via garibaldi':")
    for ent in doc.ents:
        print(ent.label_, ent.text)
    if output_directory:
        print("Loading from", output_directory)
        nlp2 = spacy.load('en', path=output_directory)
        for entity in tag_data.load_entities():
            nlp.entity.add_label(entity.upper())
        doc2 = nlp2('I want to go from piazza statuto to via garibaldi')
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    import plac
    plac.call(main)
