import fairseq,torch,sys,argparse

# Based on https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md
# Install: pip install fairseq fastBPE sacremoses subword_nmt
# Usage: echo 'Hello World' | python translate.py -m transformer.wmt19.en-de

help_msg='See https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md for models that are likely to work.\nModels will download (if necessary).\nIf you provide an unknown model, you will see a list of known models (though not all of them will work).'

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help=help_msg, default=None)
parser.add_argument("-s", "--source_language", help="two letter language code such as en, de, zh, etc.", default=None)
parser.add_argument("-t", "--target_language", help="two letter language code such as en, de, zh, etc.", default=None)
args = parser.parse_args()

tested_model_strings = [
    'transformer.wmt14.en-fr',
    'transformer.wmt16.en-de',
    'transformer.wmt19.de-en',
    'transformer.wmt19.de-en.single_model',
    'transformer.wmt19.en-de',
    'transformer.wmt19.en-de.single_model',
    'transformer.wmt19.en-ru',
    'transformer.wmt19.en-ru.single_model',
    'transformer.wmt19.ru-en',
    'transformer.wmt19.ru-en.single_model']

if args.model_string is None:
    args.model_string = 'transformer.wmt19.' + args.source_language + '-' + args.target_language + '.single_model'

print('model_string: ' + str(args.model_string))

model_strings = torch.hub.list('pytorch/fairseq') # Our code is known to fail on some of these

assert args.model_string in model_strings, help_msg + '\n%s should be one of the following (though not all of them will work): %s' % (args.model_string, '\n\t'.join(model_strings))

if not args.model_string in tested_model_strings:
    print('warning: %s may not work; the following have been tested: %s' % (args.model_string, '\n\t'.join(tested_model_strings)), file=sys.stderr)

# Warning, there are some nasty interactions between models and arguments

bpe='subword_nmt'
checkpoint_file='model.pt'

if args.model_string == 'transformer.wmt18.en-de':
    checkpoint_file='nc_model/model.pt'
if args.model_string.find('wmt19') >= 0:
    bpe='fastbpe'
    if not args.model_string.endswith('single_model'):
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt'

model = torch.hub.load('pytorch/fairseq', args.model_string, tokenizer='moses', bpe=bpe, checkpoint_file=checkpoint_file)
model.cuda()
model.eval()

assert isinstance(model.models[0], fairseq.models.transformer.TransformerModel)

for line in sys.stdin:
    sent = line.rstrip()
    print(sent + '\t' + model.translate(sent))
