import paddle
from paddlenlp.transformers import ErnieForQuestionAnswering
from paddlenlp.transformers import ErnieTokenizer
import sys

# example usage: python ernie_SQuAD.py < sample_SQuAD_input.txt

ernie = 'ernie-2.0-en-finetuned-squad'
model = ErnieForQuestionAnswering.from_pretrained(ernie)
tokenizer = ErnieTokenizer.from_pretrained(ernie)

def untokenize(tokens):
    return ' '.join(tokens).replace(' ##', '')

def answer_question(question, doc):
    tokenized_input = tokenizer(question, doc)
    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    start_logits,end_logits = model(paddle.to_tensor([input_ids]), token_type_ids=paddle.to_tensor([token_type_ids]))
    return untokenize(tokens[paddle.argmax(start_logits):paddle.argmax(end_logits)+1])

for line in sys.stdin:
    fields = line.rstrip().split('\t')
    if len(fields) >= 2:
        question,context = fields[0:2]
        answer = answer_question(question,context)
        print('question:\t' + question)
        print('answer:\t' + answer)
        print('context:\t' + context)
        print('')
