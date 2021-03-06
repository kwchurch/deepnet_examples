
import sys,argparse,datasets

# Example usage:

#  List all datasets
#    python cat_dataset.py --list 1

# Output Penn Treebank on stdout
# Warning, most of the content words have been replaced with <unk>
#    python cat_dataset.py -d ptb_text_only

# Output book corpors on stdout
# Warning, this will take a long time to download (and the output will be large)
#    python cat_dataset.py --dataset bookcorpus

# Output wikitext-2-raw-v1 test set on stdout
#    python cat_dataset.py --dataset wikitext --dataset_config wikitext-2-raw-v1 --split test

# The standard HuggingFaceHub interface to datasets requires the user
# to know quite a bit about how the dataset is organized: names of
# splits, configuations, as well as the json datastructure.  To make
# it easier for users to figure this out, there are database explorer
# links under https://huggingface.co/datasets/bookcorpus and
# https://huggingface.co/datasets/wikitext But this is still a lot of
# work, considering that there are quite a few datasets (more than
# 1k), and there will be more in the future.

# To make it easier for users, the following provides a standard
# interface to most datasets.  

# Each record in the dataset is output on stdout with a prefix that
# specifies config and split

# Under this design, one can used Unix pipes to combine datasetswith inference models
#     python cat_datasets.py <args> | python inference.py <args>

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help='see https://huggingface.co/datasets', default=None)
parser.add_argument("-l", "--list", type=int, help='list datasets', default=0)
parser.add_argument("-s", "--split", help='Specify only if you do not want to output all splits.', default=None)
parser.add_argument("-c", "--dataset_config", help='usually not necessary, but required for some datasets such as xglue', default=None)
args = parser.parse_args()

if args.list != 0:
    print('\n'.join(datasets.list_datasets()))
    sys.exit()

def dict2str(d):
    return '\t'.join([ str(k) + '|' + str(d[k]) for k in d.keys() ])

# Some datasets have annotations for each token.
# Treat this as a special case.

def data_output_by_tokens(prefix, split, d):
    for sent in d:
        if 'labels' in sent:
            print(prefix  + ' '.join([tok + '/' + str(l) for tok,l in zip(sent['tokens'], sent['labels'])]))
        else:
            print(prefix + ' '.join(sent['tokens']))

def data_output(split, d):
    prefix = str(args.dataset) + '\t' + str(args.dataset_config) + '\t' + str(split) + '\t'
    if len(d) > 0 and 'tokens' in d[0]:
        data_output_by_tokens(prefix, split, d)
    else:
        for sent in d:
            print(prefix + dict2str(sent))

def my_load_dataset():
    if args.dataset_config is None:
        return datasets.load_dataset(args.dataset)
    else: return datasets.load_dataset(args.dataset, args.dataset_config)

if not args.dataset is None:
    ds = my_load_dataset()
    if args.split is None:
        for split in ds:
            data_output(split, ds[split])
    else:
        data_output(args.split, ds[args.split])

