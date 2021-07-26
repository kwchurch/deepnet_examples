
import paddlenlp,sys,argparse,paddlenlp.datasets

# Example usage: python cat_dataset.py -d ptb 

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help='see https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/datasets.md', default=None)
parser.add_argument("-l", "--list", type=int, help='list supported datasets', default=0)
parser.add_argument("-s", "--split", help='Specify if you want just one split.', default=None)
args = parser.parse_args()


if args.list != 0:
    print('see https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/datasets.md')
    print('\n'.join(['ptb', 'cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'qnli', 'rte', 'wnli', 'lcqmc', 'chnsenticorp',
                     'squad', 'dureader_yesno', 'cmrc2018', 'drcd', 'msra_ner', 'peoples_daily_ner', 'iwslt15', 'wmt14ende', 
                     'poetry', 'couplet', 'yahoo_answer_100k']))
    sys.exit()

assert not args.dataset is None, '--dataset arg is required (unless --list is nonzero)'

def dict2str(d):
    return '\t'.join([ str(k) + '|' + str(d[k]) for k in d.keys() ])

def data_output( split, d):
    for sent in d:
        print(str(split) + '\t' + dict2str(sent))

def data_output_by_tokens(split, d, label_dict):
    assert not d is None, 'unexpected dataset: ' + args.dataset
    for sent in d:
        if 'labels' in sent:
            print(str(split) + '\t' + ' '.join([tok + '/' + label_dict[l] for tok,l in zip(sent['tokens'], sent['labels'])]))
        else:
            print(str(split) + '\t' + ' '.join(sent['tokens']))

# need to test this
def special_case_for_msra_ner(splits):
    res = paddlenlp.datasets.load_dataset(args.dataset, splits=splits)
    labels = {i:label for i,label in enumerate(res[0].label_list)}
    for lab,d in zip(splits,res):
        data_output_by_tokens(lab, d, labels)


def get_splits():
    if args.dataset in ['msra_ner']:
        return ['train', 'test']
    if not args.split == None:
        return [args.split]
    splits = ('train', 'dev', 'test')
    if args.dataset in ['cmrc2018']:
        return ['train', 'dev', 'trial']
    if args.dataset in ['squad']:
        return ['train_v1', 'dev_v1', 'train_v2', 'dev_v2']
    if args.dataset in ['mnli']:
        return ['train', 'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched']
    if args.dataset in ['ptb', 'yahoo_answer_100k']:
        return ['train', 'valid', 'test']
    return splits

splits = get_splits()

if args.dataset in ['msra_ner']:
    special_case_for_msra_ner(splits)
    sys.exit()

if args.dataset in ['cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
    res = paddlenlp.datasets.load_dataset('glue', args.dataset, splits=splits)
else:
    res = paddlenlp.datasets.load_dataset(args.dataset, splits=splits)

for split,d in zip(splits,res):
    data_output(split, d)

