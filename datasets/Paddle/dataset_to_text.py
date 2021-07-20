
import paddlenlp,sys,argparse,paddlenlp.datasets

# Example usage: python dataset_to_text.py -d ptb -o text/ptb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help='see https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/datasets.md', required=True)
parser.add_argument("-o", "--output", help='output file', required=True)
parser.add_argument("-l", "--list", type=int, help='list supported datasets', default=0)
args = parser.parse_args()


if args.list != 0:
    print('see https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/datasets.md')
    print('\n'.join(['ptb', 'cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'qnli', 'rte', 'wnli', 'lcqmc', 'chnsenticorp',
                     'squad', 'dureader_yesno', 'cmrc2018', 'drcd', 'msra_ner', 'peoples_daily_ner', 'iwslt15', 'wmt14ende', 
                     'poetry', 'couplet', 'yahoo_answer_100k']))

def dict2str(d):
    return '\t'.join([ str(k) + '|' + str(d[k]) for k in d.keys() ])

def data_output(split, d):
    with open(args.output + '.' + split, 'w') as fd:
        for sent in d:
            print(dict2str(sent), file=fd)

def data_output_by_tokens(split, d, label_dict):
    assert not d is None, 'unexpected dataset: ' + args.dataset
    with open(args.output + '.' + split, 'w') as fd:
        with open(args.output + '.' + split + '.txt', 'w') as fd2:
            for sent in d:
                print(''.join(sent['tokens']), file=fd2)
                if 'labels' in sent:
                    print(' '.join([tok + '/' + label_dict[l] for tok,l in zip(sent['tokens'], sent['labels'])]), file=fd)
                else:
                    print(' '.join(sent['tokens']), file=fd)

train=dev=test=None
splits = ('train', 'dev', 'test')

# need to test this
def special_case_for_msra_ner():
    splits=('train', 'test')
    res = paddlenlp.datasets.load_dataset(args.dataset, splits=splits)
    labels = {i:label for i,label in enumerate(res[0].label_list)}
    for lab,d in zip(splits,res):
        data_output_by_tokens(lab, d, labels)

if args.dataset in ['msra_ner']:
    special_case_for_msra_ner()
    sys.exit(0)
elif args.dataset in ['cmrc2018']:
    splits = ['train', 'dev', 'trial']
elif args.dataset in ['squad']:
    splits = ['train_v1', 'dev_v1', 'train_v2', 'dev_v2']
elif args.dataset in ['mnli']:
    splits = ['train', 'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched']
elif args.dataset in ['ptb', 'yahoo_answer_100k']:
    splits = ['train', 'valid', 'test']

if args.dataset in ['cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
    res = paddlenlp.datasets.load_dataset('glue', args.dataset, splits=splits)
elif args.dataset in ['msra_ner']:
    res = paddlenlp.datasets.load_dataset(args.dataset, splits=splits, lazy=False)
else:
    res = paddlenlp.datasets.load_dataset(args.dataset, splits=splits)

for split,d in zip(splits,res):
    data_output(split, d)

