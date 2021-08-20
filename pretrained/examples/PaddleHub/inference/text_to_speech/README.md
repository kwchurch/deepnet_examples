# Application: TTS (Text to Speech), based on PaddleHub

<b>UNDER CONSTRUCTION</b>

Super-simple example from <a href="https://www.paddlepaddle.org.cn/hubdetail?name=fastspeech_ljspeech&en_category=TextToSpeech">PaddleHub</a>

Install:
   ```shell
   # Warning, there is a dependency on soundfile.
   # That may require root permission;
   # see https://pysoundfile.readthedocs.io/en/latest/
   pip install -r requirements.txt
   git clone https://github.com/PaddlePaddle/Parakeet && cd Parakeet && pip install -e .
```

Usage:
   ```shell
   python text_to_speech.py -m transformer_tts_ljspeech -o audio < sample_input.txt
```

Reads sentences from stdin (one per line), and outputs wav files.


