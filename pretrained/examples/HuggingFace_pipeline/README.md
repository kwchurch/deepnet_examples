# HuggingFace Pipelines

Super-simple examples of HuggingFace pipelines, based on <a href="https://huggingface.co/transformers/task_summary.html">documentation from HuggingFace</a>.

```shell 
  pip install -r requirements.txt

  # sentiment analysis
  python sentiment.py < sample_sentiment_input.txt 

  # Named Entity Recognition (NER)
  python ner.py < sample_ner_input.txt

  # Question Answering (Q&A)
  python question_answering.py < sample_SQuAD_input.txt 

  # translation
  python translate.py --source_language en --target_language de < sample_translate_input.txt 
  python translate.py --source_language en --target_language zh < sample_English_input.txt 
  python translate.py --source_language zh --target_language en < sample_Chinese_input.txt 

  # back translation
  cat sample_Chinese_input.txt |
    python translate.py --source_language en --target_language en | 
    python translate.py --source_language zh --target_language en
```

Huggingface pipelines provides support for a number of important cases including:
<ol>
<li>sentiment-analysis</li>
<li>ner (Named Entity Recognition)</li>
<li>question-answering (such as SQuAD)</li>
<li>translation</li>
</ol>

For a more comprehensive list:

```python
   from transformers import pipeline
   help(pipeline)
```
