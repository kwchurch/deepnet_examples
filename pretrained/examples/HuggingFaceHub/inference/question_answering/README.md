# Application: Q&A (Question Answering), based on HuggingFace Pipelines

```shell 
  pip install -r requirements.txt

  # Question Answering (Q&A)
  python question_answering.py < sample_SQuAD_input.txt 
```

Huggingface pipelines provides support for a number of important applications including:
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
