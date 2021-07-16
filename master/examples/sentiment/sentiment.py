import sys
import paddlehub as hub

# example usage: python sentiment.py < sample_input.txt 

senta = hub.Module(name="senta_bilstm")

texts = [line.rstrip() for line in sys.stdin.readlines()]

for res in senta.sentiment_classify(texts=texts, use_gpu=False, batch_size=1):
    print('\t'.join([res['text'], res['sentiment_key'], str(res['positive_probs'])]))
