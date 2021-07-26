import torch,pdb,argparse,sys
from tqdm import tqdm
import numpy as np

# based on https://huggingface.co/transformers/perplexity.html

from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help='defaults to bert-base-uncased; see https://huggingface.co/datasets', default='bert-base-uncased')
parser.add_argument("-D", "--device", help='cpu|cuda', default='cpu')
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model_string)
model = AutoModelWithLMHead.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

from datasets import load_dataset
test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

try:
    max_length = model.config.n_positions
except:
    max_length = 512

stride = 512

lls = []
for i in tqdm(range(0, encodings.input_ids.shape[1], stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i    # may be different from stride on last loop
    input_ids = encodings.input_ids[:,begin_loc:end_loc].to(args.device)
    target_ids = input_ids.clone()
    target_ids[:,:-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len
        lls.append(log_likelihood)

ppl = torch.exp(torch.stack(lls).sum() / end_loc)
print(ppl)
