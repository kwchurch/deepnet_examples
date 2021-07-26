import os,torch,sys,argparse,scipy
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
import numpy as np
from scipy.cluster.hierarchy import linkage,leaves_list,to_tree,optimal_leaf_ordering
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("-L", "--max_length", type=int, help='max_length [defaults to 510]', default=510)
parser.add_argument("-m", "--model_string", help='defaults to bert-base-uncased', default='bert-base-uncased')
parser.add_argument("-l", "--layer", type=int, help='layer [defaults to 10]', default=10)
parser.add_argument("-O", "--reorder", type=int, help='reorder', default=0)
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
    dendrogram = scipy.cluster.hierarchy.linkage(z.T, method='complete', metric='cosine', optimal_ordering=True) 
    leaves = leaves_list(dendrogram)
    N = z.shape[0]
    scores = [z[leaves[i-1], leaves[i]] for i in range(1,len(leaves))]
    words = [tokens[j] for j in leaves] 
    return words,scores,leaves

def do_ticks(ax, labs):
    ax.set_xticks(np.arange(len(labs)))
    ax.set_yticks(np.arange(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_yticklabels(labs)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

def do_text(ax, sim):
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            text = ax.text(j, i, round(sim[i, j],1), ha="center", va="center")

def visualize_reordering(tokens,attn,layer,head,ax):
    z = attn[layer]['attn'][0,head,:,:].detach().numpy()

    sim=cosine_similarity(z)
    np.fill_diagonal(sim, np.min(sim)) # The diagonal is not interesting
    most_sim = np.argmax(sim, axis=1)
    print('Layer: %d, Head: %s, %s' % (layer, head, '; '.join([ tokens[i] + '->' + tokens[j] + ':' + str(round(sim[i,j],2)) for i,j in enumerate(most_sim)])))

    if args.reorder != 0:
        words,scores,leaves = reorder(tokens,attn,layer,head)
        sim2=sim[leaves,:][:,leaves]
        ax.set_title('Layer: %d, Head: %d' % (layer, head))
        do_text(ax, sim2)
        do_ticks(ax, words)
        return ax.imshow(sim2)
    else:
        ax.set_title('Layer: %d, Head: %d' % (layer, head))
        do_text(ax, sim)
        do_ticks(ax, tokens)
        return ax.imshow(sim)

def visualize_all_reorderings(tokens,attn):
    L,H,N = attn_shape(attn)
    fig,axs = plt.subplots(3,4)
    for head,ax in zip(range(H), axs.reshape(-1)):
        img = visualize_reordering(tokens,attn,args.layer,head,ax)
    fig.colorbar(img)
    plt.show()

for line in sys.stdin:
    cline = line.rstrip()
    tokens,attn = get_attention(model, tokenizer, cline)
    visualize_all_reorderings(tokens,attn)
    
