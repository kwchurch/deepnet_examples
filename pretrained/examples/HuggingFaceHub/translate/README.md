# Application: MT (Machine Translation), based on HuggingFace Pipelines

```shell 
  pip install -r requirements.txt

  # translation
  # supports many language pairs (see https://huggingface.co/Helsinki-NLP)
  python translate.py --source_language en --target_language de < sample_translate_input.txt 
  python translate.py --source_language en --target_language zh < sample_English_input.txt 
  python translate.py --source_language zh --target_language en < sample_Chinese_input.txt 

  # back translation
  cat sample_Chinese_input.txt |
    python translate.py --source_language en --target_language zh | 
    python translate.py --source_language zh --target_language en
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
