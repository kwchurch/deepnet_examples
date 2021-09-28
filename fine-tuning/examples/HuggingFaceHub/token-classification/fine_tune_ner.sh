#!/bin/sh

python run_ner.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name conll2003 \
  --output_dir tmp.$1/test-ner \
  --do_train \
  --do_eval
