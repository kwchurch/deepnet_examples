# based on https://github.com/jessevig/bertviz

# look at /mnt/scratch/kwc/word2vec/random_classes/BERT_similarity_reorder.py
# and /mnt/scratch/kwc/word2vec/random_classes/ernie/encoder_kwc.py
# and https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

import os,torch,sys,argparse,scipy
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
import numpy as np
from scipy.cluster.hierarchy import linkage,leaves_list,to_tree,optimal_leaf_ordering
from sklearn.metrics.pairwise import cosine_similarity

# Inputs sentences on stdin
# outputs 145 copies, permuted by all combinations of 12 layers and 12 attention heads + orig (layer -1 and attention head -1)

# Method: for each layer and attention head, compute an embedding of N by 

# echo 'to be or not to be' | python reorder.py
# layer	attention_head	reorderd
# -1	-1	[CLS] to be or not to be [SEP]
# 0	0	[CLS]:0.930 or:0.966 not:0.941 to:0.985 to:0.884 be:0.996 be:0.838 [SEP]:0.000
# 0	1	[CLS]:0.171 to:0.993 to:0.892 or:0.753 be:0.996 be:0.790 not:0.593 [SEP]:0.000
# 0	2	to:0.435 be:0.388 or:0.596 not:0.109 be:0.160 [SEP]:0.782 [CLS]:0.186 to:0.000
# 0	3	to:0.434 be:0.525 or:0.403 not:0.707 to:0.252 be:0.332 [SEP]:0.310 [CLS]:0.000
# 0	4	[SEP]:0.290 be:0.993 be:0.926 not:0.941 or:0.967 to:0.994 to:0.434 [CLS]:0.000
# 0	5	or:0.942 [SEP]:0.822 not:0.907 be:0.994 be:0.980 to:0.989 to:0.518 [CLS]:0.000
# 0	6	[CLS]:0.637 to:0.991 to:0.952 or:0.987 not:0.942 be:0.993 be:0.595 [SEP]:0.000

parser = argparse.ArgumentParser()
parser.add_argument("-L", "--max_length", type=int, help='max_length [defaults to 510]', default=510)
parser.add_argument("-m", "--model_string", help='defaults to bert-base-uncased', default='bert-base-uncased')
args = parser.parse_args()

model_version = 'bert-base-uncased'
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version)

def truncate_tokens(tokens, T):
    if len(tokens) < T:
        return tokens
    else:
        return tokens[0:T]

def get_attention(model, tokenizer, input):
    tokens = [tokenizer.cls_token] + truncate_tokens(tokenizer.tokenize(input), args.max_length) + [tokenizer.sep_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_type_ids = torch.LongTensor([[0] * len(tokens)])
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
    output = model(tokens_tensor, token_type_ids=token_type_ids)
    attn = output[-1]
    return tokens,attn

def attn_shape(attn):
    nlayers = len(attn)
    nheads = attn[0]['attn'].shape[1]
    ntokens = attn[0]['attn'].shape[2]
    return nlayers,nheads,ntokens

def reorder(tokens,attn,layer,head):
    z = attn[layer]['attn'][0,head,:,:].detach().numpy()
    zz = cosine_similarity(z)
    print('z.shape: ' + str(z.shape))
    dendrogram = scipy.cluster.hierarchy.linkage(z.T, method='complete', metric='cosine', optimal_ordering=True) 
    leaves = leaves_list(dendrogram)
    scores = np.pad(np.array([zz[leaves[i-1], leaves[i]] for i in range(1,len(leaves))]), (0,1))
    words = [tokens[j] for j in leaves] 
    return words,scores

def all_reorderings(tokens,attn):
    L,H,N = attn_shape(attn)
    print('layer\tattention_head\treorderd')
    print ('-1\t-1\t%s' % (' '.join(tokens)))
    for l in range(L):
        for h in range(H):
            words,scores = reorder(tokens, attn, l, h)
            print ('%d\t%d\t%s' % (l, h, ' '.join(['%s:%0.3f' % (word,score) for word,score in zip(words,scores)])))

for line in sys.stdin:
    cline = line.rstrip()
    tokens,attn = get_attention(model, tokenizer, cline)
    all_reorderings(tokens,attn)
    
