import sys
from transformers import pipeline

# example usage: python sentiment.py < sample_sentiment_input.txt 
# based on https://huggingface.co/transformers/task_summary.html

classifier = pipeline("sentiment-analysis")

texts = [line.rstrip() for line in sys.stdin.readlines()]

for text,res in zip(texts,classifier(texts)):
    print('\t'.join([text, res['label'], str(res['score'])]))
