# Application: TTS (Text to Speech), based on PaddleHub

Super-simple example of pre-trained translation models from <a href="https://www.paddlepaddle.org.cn/hubdetail?name=fastspeech_ljspeech&en_category=TextToSpeech">PaddleHub</a>

Install:
   ```pip install -r requirements.txt```

Usage:
   ```python text_to_speech.py -m transformer_tts_ljspeech -o audio < sample_input```

Reads sentences from stdin (one per line), and outputs wav files.


