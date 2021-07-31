# based on https://github.com/jessevig/bertviz
# and https://huggingface.co/transformers/main_classes/output.html
# and https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

# look at /mnt/scratch/kwc/word2vec/random_classes/BERT_similarity_reorder.py
# and /mnt/scratch/kwc/word2vec/random_classes/ernie/encoder_kwc.py

import os,torch,sys,argparse,scipy
import numpy as np
from scipy.cluster.hierarchy import linkage,leaves_list,to_tree,optimal_leaf_ordering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification

# Inputs sentences on stdin
# outputs 14 copies, permuted by all combinations of 13 layers + orig (layer -1)

# Method: for each layer, compute an embedding of N by 768 hidden units

# echo 'to be or not to be' | python reorder.py
# layer	reorderd
# -1	[CLS] to be or not to be [SEP]
# 0	be:0.597 be:0.321 to:0.586 to:0.254 [SEP]:0.293 [CLS]:0.313 or:0.259 not:0.000
# 1	[SEP]:0.305 not:0.322 be:0.611 be:0.313 [CLS]:0.379 or:0.374 to:0.592 to:0.000
# 2	[SEP]:0.375 [CLS]:0.422 or:0.380 not:0.335 to:0.586 to:0.371 be:0.624 be:0.000
# 3	be:0.616 be:0.396 to:0.584 to:0.385 not:0.421 or:0.426 [CLS]:0.389 [SEP]:0.000
# 4	[SEP]:0.369 not:0.414 be:0.618 be:0.409 to:0.590 to:0.435 [CLS]:0.417 or:0.000
# 5	[SEP]:0.389 not:0.430 [CLS]:0.477 to:0.613 to:0.419 or:0.424 be:0.614 be:0.000
# 6	be:0.624 be:0.443 [CLS]:0.505 to:0.649 to:0.460 not:0.449 or:0.422 [SEP]:0.000
# 7	not:0.471 be:0.634 be:0.498 to:0.672 to:0.529 [CLS]:0.488 [SEP]:0.468 or:0.000
# 8	or:0.466 [CLS]:0.505 [SEP]:0.471 to:0.678 to:0.506 be:0.642 be:0.493 not:0.000
# 9	or:0.496 to:0.686 to:0.471 not:0.500 [CLS]:0.547 [SEP]:0.488 be:0.629 be:0.000
# 10	or:0.506 [CLS]:0.554 [SEP]:0.522 not:0.516 to:0.689 to:0.563 be:0.639 be:0.000
# 11	or:0.516 not:0.533 [SEP]:0.565 [CLS]:0.573 to:0.693 to:0.548 be:0.630 be:0.000
# 12	be:0.643 be:0.543 [CLS]:0.553 [SEP]:0.499 to:0.687 to:0.561 not:0.547 or:0.000

parser = argparse.ArgumentParser()
parser.add_argument("-L", "--max_length", type=int, help='max_length [defaults to 510]', default=510)
parser.add_argument("-m", "--model_string", help='defaults to bert-base-uncased', default='bert-base-uncased')
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model_string)
model = AutoModelForSequenceClassification.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(args.model_string)

def truncate_tokens(tokens, T):
    if len(tokens) < T:
        return tokens
    else:
        return tokens[0:T]

def reorder(tokens,layer):
    z = layer.detach().numpy()[0,:,:]
    dendrogram = scipy.cluster.hierarchy.linkage(z, method='complete', metric='cosine', optimal_ordering=True) 
    leaves = leaves_list(dendrogram)
    zz = cosine_similarity(z)
    scores = np.pad(np.array([zz[leaves[i-1], leaves[i]] for i in range(1,len(leaves))]), (0,1))
    words = [tokens[j] for j in leaves] 
    return words,scores

def all_reorderings(tokens,layers):
    print('layer\treorderd')
    print ('-1\t%s' % (' '.join(tokens)))
    for l,layer in enumerate(layers):
        words,scores = reorder(tokens, layer)
        print ('%d\t%s' % (l, ' '.join(['%s:%0.3f' % (word,score) for word,score in zip(words,scores)])))

for line in sys.stdin:
    cline = line.rstrip()
    inputs = tokenizer(cline, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].detach().numpy().reshape(-1))
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    layers = model(**inputs, labels=labels, output_hidden_states=True)[2]
    all_reorderings(tokens,layers)
