# HuggingFace Pipelines

Super-simple examples of HuggingFace pipelines, based on <a href="https://huggingface.co/transformers/task_summary.html">doc</a>.

```shell 
  pip install -r requirements.txt
  python sentiment.py < sample_sentiment_input.txt 
  python ner.py < sample_ner_input.txt
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
