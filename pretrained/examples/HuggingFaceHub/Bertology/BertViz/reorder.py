import os,torch,sys,argparse,scipy
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
import numpy as np
from scipy.cluster.hierarchy import linkage,leaves_list,to_tree,optimal_leaf_ordering
from sklearn.metrics.pairwise import cosine_similarity

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
    
