import sys
from transformers import pipeline

# example usage: python ner.py < sample_ner_input.txt 
# based on https://huggingface.co/transformers/task_summary.html

tagger = pipeline("ner")

def annotation_to_str(annotation):
    print(annotation)
    return annotation['word'] + '/' + annotation['entity']

def get_NER_at(text, tagged, i):
    res = [tagged[i]]
    for i in range(i+1, len(tagged)):
        prev = tagged[i-1]
        cur = tagged[i]
        if prev['entity'] != cur['entity'] or prev['index'] + 1 != cur['index']:
            return res
        else: res.append(cur)
    return res

def get_NERs(text, tagged):
    i=0
    res = []
    while i < len(tagged):
        ner = get_NER_at(text, tagged, i)
        i += len(ner)
        res.append(ner)
    return res

def ner2str(ner, text):
    if len(ner) <= 0: return ''
    left = ner[0]['start']
    right = ner[-1]['end']
    res = text[left:right] + '/' + str(ner[0]['entity'])
    return res

for line in sys.stdin:
    text = line.rstrip()
    if len(text) > 0:
        tagged = tagger(text)
        print('input: ' + text)
        NERs = get_NERs(text, tagged)
        res = '|'.join([ner2str(ner, text) for ner in NERs])
        print('NERs:' + res)
        print('')
