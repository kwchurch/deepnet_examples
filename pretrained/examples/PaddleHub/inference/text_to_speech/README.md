# Application: TTS (Text to Speech), based on PaddleHub

<b>UNDER CONSTRUCTION</b>

<p>
<b style="background-color:yellow;">
All 3 models below produce errors; see <a href="20210819.log">log</a>
<ol>
<li>python text_to_speech.py -m transformer_tts_ljspeech
<br>
from parakeet.g2p.en import text_to_sequence
ModuleNotFoundError: No module named 'parakeet.g2p
</li>
<li>python text_to_speech.py -m deepvoice3_ljspeech
<br>
ImportError: cannot import name 'io' from 'parakeet.utils' (/mnt/home/kwc/venv/deepnet_examples10/lib/python3.7/site-packages/parakeet/utils/__init__.py)
</li>
<li>python text_to_speech.py -m fastspeech_ljspeech
<br>
from parakeet.models.fastspeech.fastspeech import FastSpeech as FastSpeechModel
ModuleNotFoundError: No module named 'parakeet.models.fastspeech'
</li>
</ol>
</b>

Super-simple example from <a href="https://www.paddlepaddle.org.cn/hubdetail?name=fastspeech_ljspeech&en_category=TextToSpeech">PaddleHub</a>

Install:
   ```shell
   # Warning, there is a dependency on soundfile.
   # That may require root permission;
   # see https://pysoundfile.readthedocs.io/en/latest/
   pip install -r requirements.txt
```

Usage:
   ```shell
   python text_to_speech.py -m transformer_tts_ljspeech -o audio < sample_input.txt
   python text_to_speech.py -m deepvoice3_ljspeech -o audio < sample_input.txt
   python text_to_speech.py -m fastspeech_ljspeech -o audio < sample_input.txt
```

Reads sentences from stdin (one per line), and outputs wav files.


