# Application: Q&A (Question Answering), based on PaddleNLP

Very simple programs to use pretrained SQuAD models based on https://huggingface.co/transformers

```shell 
  pip install -r requirements.txt
  python ernie_SQuAD.py < sample_SQuAD_input.txt
```

We provide an alternative based on HuggingFace

```shell 
  pip install -r requirements.txt
  python bert_SQuAD.py < sample_SQuAD_input.txt
```

Many <a href="https://rajpurkar.github.io/SQuAD-explorer/">SQuAD</a>
questions involve acronyms: <i>What does NFL stand for?</i>.  The
following examples show how to use these pretrained SQuAD models to
find LF (long form) expansions of SFs (short forms):


```shell
   python bert_acronym.py < sample_acronym_input.txt
   python ernie_acronym.py < ernie_acronym_input.txt
```

Although these models work surprisingly well on acronyms, there are
better solutions such as <a
href="https://github.com/ncbi-nlp/Ab3P">Ab3P</a>.

We have tested on python 3.6.9.  Other versions may work, though they have not been tested.

See <a href="https://github.com/kwchurch/bert_acronym.py">here</a> for examples like bert_acronym.py and ernie_acronym.py, but
with more models.


