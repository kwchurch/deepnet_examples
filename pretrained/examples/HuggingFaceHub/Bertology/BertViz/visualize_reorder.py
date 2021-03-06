# based on https://github.com/jessevig/bertviz

# look at /mnt/scratch/kwc/word2vec/random_classes/BERT_similarity_reorder.py
# and /mnt/scratch/kwc/word2vec/random_classes/ernie/encoder_kwc.py
# and https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

import os,torch,sys,argparse,scipy
import numpy as np
from scipy.cluster.hierarchy import linkage,leaves_list,to_tree,optimal_leaf_ordering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-L", "--max_length", type=int, help='max_length [defaults to 510]', default=510)
parser.add_argument("-m", "--model_string", help='defaults to bert-base-uncased', default='bert-base-uncased')
parser.add_argument("-l", "--layer", type=int, help='layer [defaults to 10]', default=10)
parser.add_argument("-O", "--reorder", type=int, help='reorder', default=0)
# parser.add_argument("-o", "--output", help='output file', required=True)
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
    return words,scores,leaves

def all_reorderings(tokens,layers):
    print('layer\treorderd')
    print ('-1\t%s' % (' '.join(tokens)))
    for l,layer in enumerate(layers):
        words,scores = reorder(tokens, layer)
        print ('%d\t%s' % (l, ' '.join(['%s:%0.3f' % (word,score) for word,score in zip(words,scores)])))

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

def visualize_reordering(tokens,layer,l,ax):
    z = layer.detach().numpy()[0,:,:]
    sim=cosine_similarity(z)
    # print('sim.shape: ' + str(sim.shape))
    np.fill_diagonal(sim, np.min(sim)) # The diagonal is not interesting

    if args.reorder != 0:
        words,scores,leaves = reorder(tokens,layer)
        sim2=sim[leaves,:][:,leaves]
        ax.set_title('Layer: %d with reordering' % l)
        do_text(ax, sim2)
        do_ticks(ax, words)
        return ax.imshow(sim2)
    else:
        ax.set_title('Layer: %d without reordering' % l)
        do_text(ax, sim)
        do_ticks(ax, tokens)
        return ax.imshow(sim)

def visualize_all_reorderings(tokens,layers):
    fig,axs = plt.subplots(3,4)
    for l,layer in enumerate(layers[0:12]):
        ax = axs.reshape(-1)[l]
        img = visualize_reordering(tokens,layer, l, ax)
    fig.colorbar(img)
    # plt.savefig(args.output, metadata='pdf')
    plt.show()

for line in sys.stdin:
    cline = line.rstrip()
    inputs = tokenizer(cline, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].detach().numpy().reshape(-1))
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    layers = model(**inputs, labels=labels, output_hidden_states=True)[2]
    visualize_all_reorderings(tokens,layers)
