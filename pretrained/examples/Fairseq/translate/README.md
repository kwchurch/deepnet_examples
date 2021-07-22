# Application: MT (Machine Translation), based on Fairseq

Super-simple example of pre-trained translation models from <a href="https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md">fairseq</a>

Install:
   ```pip install -r requirements.txt```

Usage:
   ```echo 'Hello World' | python translate.py -m transformer.wmt19.en-de```

Reads sentences from stdin (one per line), and outputs: 
   ```< sent> \t <translation>```
on stdout

For help:
   ```python translate.py --help```

If you specify an unknown model, you will see a list of known models.
      
See also: https://github.com/kwchurch/deepnet_examples/tree/main/pretrained/examples/HuggingFace_pipeline
for translation of more language pairs, using HuggingFace piplines

More examples:
   ```echo 'Hello World' | python translate.py -m transformer.wmt14.en-fr
      echo 'Hello World' | python translate.py -m transformer.wmt16.en-de
      echo 'Hello World' | python translate.py -m transformer.wmt19.de-en.single_model
      echo 'Hello World' | python translate.py -m transformer.wmt19.en-de.single_model
      echo 'Hello World' | python translate.py -m transformer.wmt19.en-ru.single_model
      echo 'Hello World' | python translate.py -m transformer.wmt19.ru-en.single_model
```
