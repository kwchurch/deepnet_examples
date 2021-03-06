import sys,argparse
import os,torch,sys,argparse,scipy
import numpy as np
from scipy.cluster.hierarchy import linkage,leaves_list,to_tree,optimal_leaf_ordering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help='defaults to bert-base-uncased', default='bert-base-uncased')
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model_string)
model = AutoModelForSequenceClassification.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(args.model_string)

for line in sys.stdin:
    cline = line.rstrip()
    inputs = tokenizer(cline, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].detach().numpy().reshape(-1))
    print(' '.join(tokens))
