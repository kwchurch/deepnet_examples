import numpy as np
import gensim.downloader,sys,argparse
from gensim.models import KeyedVectors

# Input from stdin: One word per line
# Output to stdout: The input word, followd by top_k candidates with scores

# echo 'this is a test of the emergency broadcast system .' | tr ' ' '\n' | python near.py
#
# this	it:0.91, same:0.89, one:0.87, that:0.87, but:0.87
# is	this:0.86, it:0.85, now:0.85, has:0.83, be:0.83
# a	another:0.92, one:0.84, an:0.82, this:0.81, same:0.79
# test	tests:0.90, testing:0.83, tested:0.68, match:0.64, cricket:0.61
# of	the:0.83, from:0.77, part:0.77, all:0.76, for:0.76
# the	this:0.86, part:0.85, one:0.85, of:0.83, same:0.83
# emergency	relief:0.70, aid:0.69, assistance:0.67, evacuation:0.67, medical:0.66
# broadcast	broadcasts:0.87, television:0.87, aired:0.84, broadcasting:0.83, radio:0.81
# system	systems:0.86, control:0.72, program:0.70, which:0.70, using:0.69
# .	but:0.90, although:0.88, however:0.88, ,:0.88, when:0.87

# If there are multiple instances of the same input word, they will
# have the same vector (and the same neighbors), under static
# embedding (but not under contextual embeddings such as BERT and
# ERNIE).  The following examples have multiple instances of "to",
# "be" and "can".  Note that the candidates are the same across
# multiple instances of these input words.

# echo 'to be or not to be' | tr ' ' '\n' | python near.py
#
# to	would:0.87, take:0.87, help:0.86, make:0.86, could:0.84
# be	not:0.93, would:0.90, should:0.90, could:0.90, being:0.89
# or	either:0.80, instead:0.80, other:0.79, rather:0.77, those:0.77
# not	be:0.93, would:0.93, could:0.92, should:0.91, they:0.91
# to	would:0.87, take:0.87, help:0.86, make:0.86, could:0.84
# be	not:0.93, would:0.90, should:0.90, could:0.90, being:0.89

# echo 'can you can cans ?' | tr ' ' '\n' | python near.py
#
# can	could:0.88, should:0.87, be:0.86, will:0.86, might:0.86
# you	'll:0.93, n't:0.91, know:0.91, i:0.90, do:0.90
# can	could:0.88, should:0.87, be:0.86, will:0.86, might:0.86
# cans	bottles:0.84, cartons:0.72, buckets:0.71, bottle:0.71, containers:0.69
# ?	you:0.89, '':0.87, maybe:0.87, know:0.87, thing:0.86


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--list", type=int, help='list available models', default=0)
parser.add_argument("-m", "--model_string", help='defaults to glove-wiki-gigaword-100 (faster when model_string ends with .annoy)', default='glove-wiki-gigaword-100')
parser.add_argument("-K", "--top_k", type=int, help='number of candidates to output [default to 5]', default=5)
args = parser.parse_args()

if args.list != 0:
    print('\n'.join(gensim.downloader.info()['models'].keys()))
    sys.exit()

if args.model_string.endswith('.annoy'):
    M = KeyedVectors.load(args.model_string,  mmap='r')
else:
    M = gensim.downloader.load(args.model_string)

for line in sys.stdin:
    words = line.rstrip().split()
    if len(words) > 0:
        try: 
            candidates = M.most_similar(words, topn=args.top_k)
            candidates_string = ', '.join('%s:%0.2f' % (candidate[0], candidate[1]) for candidate in candidates)
            print(line.rstrip() + '\t' + candidates_string)
        except:
            print(line.rstrip() + '\t*** ERROR ***')
