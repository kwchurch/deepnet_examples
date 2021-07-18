import sys,argparse
from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer

# example usage: python translate.py --source_language en --target_language de < sample_translate_input.txt 
# based on https://huggingface.co/transformers/task_summary.html

# Example of back translation via Chinese:
#   cat sample_English_input.txt | 
#   python translate.py --source_language en --target_language zh | 
#   python translate.py --source_language zh --target_language en

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_language", help="defaults to en", required=True)
parser.add_argument("-t", "--target_language", help="defaults to de", required=True)
args = parser.parse_args()

# HuggingFace pipelines support a few language pairs
if args.source_language in ['en'] and args.target_language in ['fr', 'de', 'ro']:
    translator = pipeline('translation_' + args.source_language + '_to_' + args.target_language)
    for line in sys.stdin:
        sent = line.rstrip()
        if len(sent) > 0:
            res = translator(sent, max_length=400)
            print(res[0]['translation_text'])
else:
    # There are pretrained models for many more language pairs 
    # See https://huggingface.co/Helsinki-NLP and https://www.aclweb.org/anthology/2020.eamt-1.pdf#page=499
    model_string = 'Helsinki-NLP/opus-mt-' + args.source_language + '-' + args.target_language
    model = AutoModelWithLMHead.from_pretrained(model_string)
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    for line in sys.stdin:
        sent = line.rstrip()
        if len(sent) > 0:
            res = model.generate(tokenizer.encode(sent, return_tensors="pt"), 
                                 max_length=400, num_beams=40, early_stopping=True)
            print(tokenizer.decode(res[0]))
