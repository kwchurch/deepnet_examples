# Application of ASR (Automatic Speech Recognition, aka Speech to Text), based on HuggingFaceHub

This example is borrowed from https://huggingface.co/facebook/wav2vec2-base-960h

```shell
   pip install -r requirements.txt
   ls wavs/lj_speech/*wav | python speech_to_text.py
   python speech_dataset_filenames.py -d lj_speech | head | python speech_to_text.py
```

Reads input from stdin and writes transcriptions to stdout.
Input lines start with filenames.  Each of these files contain audio.
