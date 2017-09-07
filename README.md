# botcycle-nlu
Experiments with Natural Language Understanding

## Wit_data

This folder contains the wit.ai dataset. To download simply run the `download.sh` script (you need `WIT_TOKEN` env variable specific for your bot, leak it from browser developer tools)

## Spacy_entities

Contains code that trains the NER of spacy on the wit_data. Run the `train.py` script that will load the `'en'` model and after training will save the updated model in the subfolder `model`.
