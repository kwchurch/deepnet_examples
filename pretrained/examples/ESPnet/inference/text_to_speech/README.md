# Application: TTS (Text to Speech), based on ESPnet

Based on <a href="https://github.com/espnet/espnet_model_zoo">ESPnet Model Zoo</a>

Install:
   ```shell
   # Warning, there is a dependency on soundfile.
   # That may require root permission;
   # see https://pysoundfile.readthedocs.io/en/latest/
   pip install -r requirements.txt
```

Usage:
   ```shell
   python text_to_speech.py -o audio < sample_input.txt
```

Reads sentences from stdin (one per line), and outputs wav files.
