# Application of ASR (Automatic Speech Recognition, aka Speech to Text), based on ESPNet

Based on <a href="https://github.com/espnet/espnet_model_zoo">ESPnet Model Zoo</a>


```shell
   pip install -r requirements.txt
   find . -name '*.wav' | python speech_to_text.py -l en --corpus librispeech
```

Reads input from stdin and writes transcriptions to stdout.
Input lines start with filenames.  Each of these files contain audio.
