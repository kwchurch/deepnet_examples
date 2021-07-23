import torch,pdb
from tqdm import tqdm

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
device = 'cuda'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

print('point 1')

from datasets import load_dataset
test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

max_length = model.config.n_positions
stride = 512

print('point 2')

# pdb.set_trace()
print(encodings.input_ids.size(1))
# pdb.set_trace()
lls = []
for i in tqdm(range(0, encodings.input_ids.shape[1], stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i    # may be different from stride on last loop
    input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:,:-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len

    print(log_likelihood)
    lls.append(log_likelihood)

ppl = torch.exp(torch.stack(lls).sum() / end_loc)

print(ppl)
