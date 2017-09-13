import json


def load_expressions():
    """
    Returns the expressions_file loaded from JSON
    """
    with open('../wit_data/BotCycle/expressions.json') as expressions_file:
        return json.load(expressions_file)


def load_entities():
    """
    Returns a list of names of the entities (excluding intent)
    """
    # TODO dynamic look into entities folder
    return ['location']


def tag(expressions):
    """Returns a list of objects like `{'text': SENTENCE, entities: [{'entity': ENTITY_NAME, 'value': ENTITY_VALUE, 'start': INT, 'end', INT}]}`"""
    array = expressions['data']
    result = list(map(lambda x: {'text': x['text'], 'entities': [
                  ent for ent in x['entities'] if ent['entity'] != "intent"], }, array))

    return result
