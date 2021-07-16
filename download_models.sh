#!/bin/sh

for model in `cat all_models.sh`
do

    echo 'Hello world.' | python  ~kwc/papers/deepnets_tutorial/translate.py -m $model
done
