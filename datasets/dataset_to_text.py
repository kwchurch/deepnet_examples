
import paddlenlp,sys,argparse,paddlenlp.datasets

# example usage: python dataset_to_text.py -d ptb -o text/ptb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help='see https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/datasets.md', required=True)
parser.add_argument("-o", "--output", help='output file', required=True)
args = parser.parse_args()

def dict2str(d):
    return '\t'.join([ str(k) + '|' + str(d[k]) for k in d.keys() ])

def data_output(ext, d):
    assert not d is None, 'unexpected dataset: ' + args.dataset
    with open(args.output + '.' + ext, 'w') as fd:
        for sent in d:
            print(dict2str(sent), file=fd)

train=dev=test=None

if args.dataset in ['msra_ner']:
    splits = ['train', 'test']
elif args.dataset in ['cmrc2018']:
    splits = ['train', 'dev', 'trial']
elif args.dataset in ['squad']:
    splits = ['train_v1', 'dev_v1', 'train_v2', 'dev_v2']
elif args.dataset in ['mnli']:
    splits = ['train', 'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched']
elif args.dataset in ['ptb', 'yahoo_answer_100k']:
    splits = ['train', 'valid', 'test']
else:
    splits = ['train', 'dev', 'test']

if args.dataset in ['cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
    res = paddlenlp.datasets.load_dataset('glue', args.dataset, splits=splits)
else:
    res = paddlenlp.datasets.load_dataset(args.dataset, splits=splits)

for lab,d in zip(splits,res):
    data_output(lab, d)

