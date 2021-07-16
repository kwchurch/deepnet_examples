# SQuAD


Very simple programs  to use BERT-SQuAD and ERNIE-SQuAD to find long forms (LFs) for short forms (SFs)
```shell 
  pip install -r requirements.txt
  python bert_SQuAD.py < sample_input.txt 
  python ernie_SQuAD.py < sample_input.txt 
```

Some SQuAD questions involve acronyms, and therefore,
these pre-trained models can be used to find expansions of SFs (short forms).
The input is a SF and a context.  Output is a LF (long form).

```shell 
  python ernie_acronym.py < sample_acronym_input.txt
  python bert_acronym.py < sample_acronym_input.txt
```

We have tested on python 3.6.9.  Other versions may work but not tested.

If this does not work, you may try to upgrade pip.  Also, this may help:
```shell
  pip install --upgrade paddlenlp -i https://pypi.org/simple
  ```
