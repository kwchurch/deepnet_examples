
import fairseq,torch,sys,argparse

# Based on https://www.paddlepaddle.org.cn/hubdetail?name=transformer_zh-en&en_category=MachineTranslation
# Usage: python translate.py -m 'transformer_zh-en' < sample_Chinese_input.txt

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help="transformer_zh-en|transformer_en-de", required=True)
args = parser.parse_args()

import sys
import paddlehub as hub

model = hub.Module(name=args.model_string, beam_size=5)
src_texts = sys.stdin.read().split('\n')

n_best = 3  # 每个输入样本的输出候选句子数量
trg_texts = model.predict(src_texts, n_best=n_best)
for idx, st in enumerate(src_texts):
    if len(st) > 0:
        print('-'*30)
        print(f'src: {st}')
        for i in range(n_best):
            print(f'trg[{i+1}]: {trg_texts[idx*n_best+i]}')
    
