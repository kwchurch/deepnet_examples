# translate

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
      
