# Application: MT (Machine Translation), based on PaddleHub

Super-simple example of pre-trained translation models from <a href="https://www.paddlepaddle.org.cn/hubdetail?name=transformer_zh-en&en_category=MachineTranslation">PaddleHub</a>

Install:
   ```pip install -r requirements.txt```

Usage:
   ```python translate.py -m 'transformer_zh-en' < sample_Chinese_input.txt```

Reads sentences from stdin (one per line), and outputs n_best translations on stdout

