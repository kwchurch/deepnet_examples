# Datasets

Simple program to load datasets based on <a href="https://huggingface.co/datasets">datasets in HuggingFace</a>

```shell 
  pip install -r requirements.txt
  # download dataset and output as text files to text/ptb_text_only
  python dataset_to_text.py -d ptb_text_only -o text/ptb_text_only*
  # list datasets
  python dataset_to_text.py -l 1
```