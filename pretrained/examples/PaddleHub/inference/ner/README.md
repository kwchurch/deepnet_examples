# Application named editity recognition (NER), based on PaddleHub

Super-simple example of named entity recognition from <a
href="https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.1/demo/sequence_labeling">PaddleHub</a>.
See <a href="https://www.paddlepaddle.org.cn/hublist">PaddleHub
home</a> for many more pre-trained models.


Install:
   ```shell
   pip install -r requirements.txt
```

Example of Usage:
   ```shell
   python ner.py
```

To fine tune:

   ```shell
   python fine_tune_ner.py
```

WARNING: this will take a long time.

After fine tuning you can say <i> to be completed </i>

   ```shell
   python ner.py
```

