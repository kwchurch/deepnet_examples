# Application: Sentiment Analysis, based on HuggingFace Pipelines

```shell 
  pip install -r requirements.txt

  # sentiment analysis
  python sentiment.py < sample_sentiment_input.txt 
```

Huggingface pipelines provide support for a number of important applications including:
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

See <a href="https://huggingface.co/transformers/task_summary.html">documentation from HuggingFace</a>.
