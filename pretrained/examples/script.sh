#!/bin/sh

# Each example is in a subdirectory by itself
# There should be no dependencies across examples
# Each subdirectory should be independent and self-explanatory

# Standard interfaces
# Most programs conform to the following conventions:
#   read input from stdout
#   write output to stdout
#   consistent argument names (when possible)
#     --help
#     --list
#     --source_language
#     --target_language
#     --model_string
#     --top_k
#     --max_length

# install
# echo 'installing'
# find . -name requirements.txt -exec pip install -r {} \;

# image classification
echo Inference Task: image classification
dir=HuggingFaceHub/inference/image_classification
ls $dir/images/*jpg | python $dir/OCR.py

# OCR
echo Inference Task: OCR
dir=PaddleHub/inference/OCR
echo "$dir/Sample_input_for_OCR.png" | python $dir/OCR.py

# sentiment
echo Inference Task: sentiment
dir=PaddleHub/inference/sentiment
python $dir/paddlehub_sentiment.py < $dir/sample_Chinese_input.txt
dir=HuggingFaceHub/inference/sentiment
python $dir/sentiment.py < $dir/sample_sentiment_input.txt 

# NER (named entity recognition)
echo Inference Task: NER
dir=HuggingFaceHub/inference/ner
python $dir/ner.py < $dir/sample_ner_input.txt

# QA (question answering)
echo Inference Task: QA
dir=PaddleNLP/inference/question_answering
python $dir/ernie_SQuAD.py < $dir/sample_SQuAD_input.txt
dir=HuggingFaceHub/inference/question_answering
python $dir/question_answering.py < $dir/sample_SQuAD_input.txt

# MT (machine translation)
echo Inference Task: MT
dir=PaddleHub/inference/translate
python $dir/translate.py -m 'transformer_zh-en' < $dir/sample_Chinese_input.txt
dir=HuggingFaceHub/inference/translate
python $dir/translate.py --source_language en --target_language de < $dir/sample_translate_input.txt 
python $dir/translate.py --source_language en --target_language zh < $dir/sample_English_input.txt 
python $dir/translate.py --source_language zh --target_language en < $dir/sample_Chinese_input.txt

# back translation (from English to English via Chinese)
  cat sample_Chinese_input.txt |
    python $dir/translate.py --source_language en --target_language zh | 
    python $dir/translate.py --source_language zh --target_language en

dir=Fairseq/inference/translate
echo 'Hello World' | python $dir/translate.py --model_string transformer.wmt16.en-de
echo 'Hello World' | python $dir/translate.py --model_string transformer.wmt19.de-en.single_model
echo 'Hello World' | python $dir/translate.py --model_string transformer.wmt19.en-de.single_model
echo 'Hello World' | python $dir/translate.py --model_string transformer.wmt19.en-ru.single_model
echo 'Hello World' | python $dir/translate.py --model_string transformer.wmt19.ru-en.single_model

# Combinations of OCR and translation
OCRdir=PaddleHub/inference/OCR
MTdir=HuggingFaceHub/inference/translate
echo "$OCRdir/Sample_input_for_OCR.png" | 
    python $OCRdir/OCR.py | 
    python $MTdir/translate.py --source_language zh --target_language en
