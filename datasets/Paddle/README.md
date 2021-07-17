# Datasets

Simple program to load datasets based on <a href="https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/datasets.md">datasets in PaddleNLP</a>

```shell 
  pip install -r requirements.txt
  mkdir -p text
  for d in ptb cola sst-2 mrpc sts-b qqp mnli qnli rte wnli lcqmc chnsenticorp squad dureader_yesno cmrc2018 drcd msra_ner peoples_daily_ner iwslt15 wmt14ende poetry couplet yahoo_answer_100k
  do
  echo $d
  python dataset_to_text.py -d $d -o text/$d
  done
```
