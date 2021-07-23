
# example is borrowed from https://huggingface.co/facebook/wav2vec2-base-960h
# see also: https://huggingface.co/transformers/model_doc/speech_to_text.html

# example of usage:
# ls wavs/lj_speech/*wav | python speech_to_text.py

import pdb
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from datasets import load_dataset
import soundfile as sf
import torch
import sys
import scipy.signal
import numpy as np

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# ASR = Automatic Speech Recognition (aka Speech-to-Text)
def ASR(filename):
    speech, sampling_rate = sf.read(filename)

    # assuming the models are trained for 16k samples per second
    if sampling_rate != 16000:
        sec = len(speech)/sampling_rate
        speech = scipy.signal.resample(speech, int(sec * 16000))

    tokens = tokenizer(speech, return_tensors="pt", padding="longest").input_values
    logits = model(tokens).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    return transcription

# assumes that lines from stdin start with filenames,
# and each of these files contain audio in a format that can be converted to 16k samples per second

for line in sys.stdin:
    fields = line.rstrip().split('\t')
    if len(fields) > 0:
        filename = fields[0]
        print(line + '\t' + '|'.join(ASR(filename)) + '\n')
