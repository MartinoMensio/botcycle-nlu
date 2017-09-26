import json

with open('atis/atis.train.w-intent.iob') as f:
    content = f.readlines()

result = []

entity_types = set()

for line in content:
    element = {}
    start_text_idx = line.find('BOS ') + 4
    end_text_idx = line.find('EOS', start_text_idx)
    text = line[start_text_idx:end_text_idx]
    text = text.strip()
    element['text'] = text
    start_annotations_idx = line.find('\t') + 1
    annotations = line[start_annotations_idx:]
    annotations = annotations.split()
    entities_tags = annotations[1:-1]
    #entities_tags = iob_to_biluo(entities_tags)
    intent = annotations[-1]
    element['intent'] = intent
    # chunks are defined by the space, IOB notations correspond to this split
    chunks = text[:start_annotations_idx - 1].split()
    entities = []
    state = 'O'
    entity = {}
    for idx, tag in enumerate(entities_tags):
        tag = tag.split('-')
        next_state = tag[0]
        if len(tag) is 2:
            if next_state is 'B':
                if state is 'B':
                    # close previous entity
                    entity['end'] = sum(map(len, chunks[:idx])) + idx - 1
                    entity['value'] = element['text'][entity['start']:entity['end']]
                    entities.append(entity)
                # beginning of new entity
                entity = {'entity': tag[1], 'start': sum(map(len, chunks[:idx])) + idx}
                entity_types.add(tag[1])

        if next_state is 'O' and state is not 'O':
            # end of entity inside the sentence
            entity['end'] = sum(map(len, chunks[:idx])) + idx - 1
            entity['value'] = element['text'][entity['start']:entity['end']]
            entities.append(entity)
            entity = {}

        #update state
        state = next_state

    if state is not 'O':
        # last entity at the end of the sentence
        idx = len(entities_tags)
        #print('idx:',idx,'previous chunks:', chunks[:idx])
        entity['end'] = sum(map(lambda c: len(c), chunks[:idx])) + idx - 1
        entity['value'] = element['text'][entity['start']:entity['end']]
        entities.append(entity)
        #print (element, entities)
        entity = {}

    element['entities'] = entities

    result.append(element)
    

print(result)
with open('atis_parsed_texts.json', 'w') as outfile:
    json.dump(result, outfile)

with open('atis_entities.json', 'w') as outfile:
    json.dump(list(entity_types), outfile)