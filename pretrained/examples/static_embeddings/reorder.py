import numpy as np
import gensim.downloader,sys,argparse
import scipy
from scipy.cluster.hierarchy import linkage,leaves_list,to_tree,optimal_leaf_ordering
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

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

def reorder(tokens):
    found = [token for token in tokens if token in M]
    if len(found) < 2:
        return found,None
    z = np.array([M[token] for token in found])
    dendrogram = scipy.cluster.hierarchy.linkage(z, method='complete', metric='cosine', optimal_ordering=True) 
    leaves = leaves_list(dendrogram)
    zz = cosine_similarity(z)
    scores = np.pad(np.array([zz[leaves[i-1], leaves[i]] for i in range(1,len(leaves))]), (0,1))
    permuted = [found[j] for j in leaves] 
    return permuted,scores

for line in sys.stdin:
    cline = line.rstrip()
    words,scores = reorder(cline.split())
    if scores is None:
        print(cline  + '\t*** ERROR ****')
    else:
        print(cline  + '\t' +  ', '.join(['%s:%0.2f' %(word, score) for word,score in zip(words,scores)]))
    
    
