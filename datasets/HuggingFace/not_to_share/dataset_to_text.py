
import sys,argparse,datasets

# Example usage:
#    python dataset_to_text.py -d ptb_text_only -o text/ptb_text_only
# To list supported datasets: 
#    python dataset_to_text.py -l 1

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help='see https://huggingface.co/datasets', default=None)
parser.add_argument("-o", "--output", help='output file [defaults --> dataset]', default=None)
parser.add_argument("-l", "--list", type=int, help='list supported datasets', default=0)
parser.add_argument("-c", "--config", help='usually not necessary, but required for some datasets such as xglue', default=None)
args = parser.parse_args()

if args.list != 0:
    print('\n'.join(datasets.list_datasets()))

def dict2str(d):
    return '\t'.join([ str(k) + '|' + str(d[k]) for k in d.keys() ])

def data_output_by_tokens(split, d):
    with open(args.output + '.' + split, 'w') as fd:
        with open(args.output + '.' + split + '.txt', 'w') as fd2:
            for sent in d:
                print(' '.join(sent['tokens']), file=fd2)
                if 'labels' in sent:
                    print(' '.join([tok + '/' + str(l) for tok,l in zip(sent['tokens'], sent['labels'])]), file=fd)
                else:
                    print(' '.join(sent['tokens']), file=fd)

def data_output(split, d):
    if len(d) > 0 and 'tokens' in d[0]:
        data_output_by_tokens(split, d)
    else:
        with open(args.output + '.' + split, 'w') as fd:
            for sent in d:
                print(dict2str(sent), file=fd)

def my_load_dataset():
    if args.config is None:
        return datasets.load_dataset(args.dataset)
    else: return datasets.load_dataset(args.dataset, args.config)

if not args.dataset is None:
    if args.output is None:
        args.output = args.dataset
    ds = my_load_dataset()
    for split in ds:
        data_output(split, ds[split])

