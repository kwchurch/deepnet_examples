# Application: TTS (Text to Speech), based on PaddleHub

Super-simple example from <a href="https://www.paddlepaddle.org.cn/hubdetail?name=fastspeech_ljspeech&en_category=TextToSpeech">PaddleHub</a>

Install:
   ```shell
   # Warning, there is a dependency on soundfile.
   # That may require root permission;
   # see https://pysoundfile.readthedocs.io/en/latest/

   # parakeet is a bit tricky to install
   git clone https://github.com/PaddlePaddle/Parakeet
   cd Parakeet
   git checkout release/v0.1
   pip install -e .

   # The other packages are easier
   pip install -r requirements.txt
```

Usage:
   ```shell
   python text_to_speech.py -o audio < sample_input.txt
```

Reads sentences from stdin (one per line), and outputs wav files.


