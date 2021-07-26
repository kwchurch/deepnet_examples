from transformers import pipeline
import sys

unmasker = pipeline('fill-mask', model='bert-base-uncased')

def maskify(tokens, i):
    left = right = []
    if i > 0:
        left = tokens[0:i]
    if i < len(tokens):
        right = tokens[(i+1):]
    return ' '.join(left) + ' [MASK] ' + ' '.join(right)

for line in sys.stdin:
    rline = line.rstrip()
    tokens = rline.split()
    for i,token in enumerate(tokens):
        masked = maskify(tokens, i)
        candidates = ['%s|%0.3f' % (candidate['token_str'], candidate['score']) for candidate in unmasker(masked)]
        print(token + '\t' + '\t'.join(candidates))
    sys.stdout.flush()
