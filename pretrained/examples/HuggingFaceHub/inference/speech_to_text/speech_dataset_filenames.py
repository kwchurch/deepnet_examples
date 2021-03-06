from datasets import load_dataset
import argparse,pdb

# example of usage:
#   python speech_dataset_filenames.py -d patrickvonplaten/librispeech_asr_dummy -c clean -s validation

# outputs filename + dataset + split, one per line, with tabs between fields

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="e.g., patrickvonplaten/librispeech_asr_dummy", required=True)
parser.add_argument("-c", "--config", help="e.g., clean", default=None)
parser.add_argument("-s", "--split", help="e.g., validation", default=None)
args = parser.parse_args()

def my_load_dataset(dataset, config, split):
    if not config is None and not split is None:
        return load_dataset(dataset, config, split)
    if not config is None and split is None:
        return load_dataset(dataset, config)
    if config is None and not split is None:
        return load_dataset(dataset, split)
    if config is None and split is None:
        return load_dataset(dataset)
    assert False, 'should not get here'

ds = my_load_dataset(args.dataset, args.config, args.split)
for split in ds:
    vals = args.dataset + '\t' + str(split)
    for d in ds[split]:
        print(d['file'] + '\t' + vals)

