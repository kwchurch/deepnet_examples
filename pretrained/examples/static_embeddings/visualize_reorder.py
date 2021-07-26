import os,torch,sys,argparse,scipy,gensim.downloader,gensim
import numpy as np
from scipy.cluster.hierarchy import linkage,leaves_list,to_tree,optimal_leaf_ordering
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--list", type=int, help='list available models', default=0)
parser.add_argument("-m", "--model_string", help='defaults to glove-wiki-gigaword-100', default='glove-wiki-gigaword-100')
parser.add_argument("-K", "--top_k", type=int, help='number of candidates to output [default to 5]', default=5)
args = parser.parse_args()

if args.list != 0:
    print('\n'.join(gensim.downloader.info()['models'].keys()))
    sys.exit()

M = gensim.downloader.load(args.model_string)

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
            text = ax.text(j, i, round(sim[i, j],2), ha="center", va="center")

for line in sys.stdin:
    tokens = line.rstrip().split()
    found = [token for token in tokens if token in M]
    z = np.array([M[token] for token in found])
    dendrogram = scipy.cluster.hierarchy.linkage(z, method='complete', metric='cosine', optimal_ordering=True) 
    leaves = leaves_list(dendrogram)
    permuted = [ found[leaf] for leaf in leaves ]

    sim = cosine_similarity(z)
    sim2=sim[leaves,:][:,leaves]
    
    fig,axs = plt.subplots(1,2)

    do_ticks(axs[0], found)
    do_ticks(axs[1], permuted)

    axs[0].set_title('without reordering')
    axs[1].set_title('with reordering')
    axs[0].imshow(sim)
    img = axs[1].imshow(sim2)
    fig.colorbar(img)

    do_text(axs[0], sim)
    do_text(axs[1], sim2)

    plt.show()
