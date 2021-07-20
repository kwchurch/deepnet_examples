import sys
from transformers import pipeline

# example usage: python question_answering.py < sample_SQuAD_input.txt 
# based on https://huggingface.co/transformers/task_summary.html

QA = pipeline("question-answering")

for line in sys.stdin:
    fields = line.rstrip().split('\t')
    if len(fields) >= 2:
        question,context=fields[0:2]
        print('context: ' + context)
        print('question: ' + question)
        res = QA(question=question, context=context)
        print('answer: ' + res['answer'] + '\tscore: %0.4f' % res['score'] + ' span: %d-%d' % (res['start'], res['end']))
