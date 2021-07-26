from transformers import pipeline
import sys,argparse

# Input from stdin: One sentence per line
# Output to stdout: One word per line, followd by top_k candidates with scores

# Method: Mask each word in the input sentence,
# and run the model to predict each word in the input from the rest of the input sentence.

# echo 'This is a test of the emergency broadcast system .' | python unmask.py
#
# this	this|0.595, it|0.375, there|0.004, that|0.002, here|0.001
# is	was|0.628, is|0.334, included|0.007, includes|0.004, involved|0.003
# a	a|0.935, another|0.029, the|0.023, one|0.005, first|0.000
# test	part|0.253, subset|0.125, variation|0.082, feature|0.072, component|0.051
# of	of|0.886, for|0.070, on|0.017, in|0.010, against|0.004
# the	the|0.883, an|0.088, our|0.005, your|0.003, their|0.002
# emergency	radio|0.257, television|0.048, satellite|0.031, digital|0.031, microwave|0.029
# broadcast	braking|0.620, response|0.070, management|0.024, medical|0.016, lighting|0.014
# system	system|0.244, technique|0.077, capability|0.059, technology|0.033, method|0.028
# .	.|0.969, ;|0.029, !|0.001, ?|0.000, ||0.000

# If there are multiple instances of the same input word, they will
# have the same vector (and the same neighbors), under static
# embedding (but not under contextual embeddings such as BERT and
# ERNIE).  The following examples have multiple instances of "to",
# "be" and "can".  Note that the candidates are different across
# multiple instances of these input words.

# echo 'to be or not to be' | python unmask.py
#
# to	to|0.995, will|0.001, whether|0.000, either|0.000, from|0.000
# be	be|0.906, know|0.012, have|0.008, do|0.006, .|0.003
# or	or|0.451, .|0.128, and|0.090, is|0.090, ,|0.083
# not	not|0.755, never|0.059, ,|0.015, is|0.013, only|0.008
# to	to|0.958, .|0.014, will|0.004, would|0.003, must|0.003
# be	?|0.593, .|0.392, ;|0.009, !|0.006, ...|0.000

# echo 'can you can cans ?' | python unmask.py
#
# can	do|0.227, "|0.196, and|0.120, so|0.086, but|0.048
# you	-|0.320, can|0.103, ,|0.063, you|0.035, cans|0.027
# can	open|0.091, see|0.047, get|0.043, smell|0.040, use|0.038
# cans	fly|0.136, walk|0.106, see|0.060, talk|0.040, run|0.031
# ?	?|0.748, .|0.154, !|0.094, ;|0.003, ||0.000

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help='defaults to bert-base-uncased', default='bert-base-uncased')
parser.add_argument("-K", "--top_k", type=int, help='number of candidates to output [default to 5]', default=5)
args = parser.parse_args()

unmasker = pipeline('fill-mask', model=args.model_string, top_k=args.top_k)

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
        print(token + '\t' + ', '.join(candidates))
    sys.stdout.flush()
