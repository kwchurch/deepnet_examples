# Application OCR, based on PaddleHub

Super-simple example of pre-trained resnet model from <a
href="https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.1/demo/image_classification">PaddleHub</a>.
See <a href="https://www.paddlepaddle.org.cn/hublist">PaddleHub
home</a> for many more pre-trained models.


Install:
   ```shell
   pip install -r requirements.txt
```

Example of Usage:
   ```shell
   find flower_photos -name '*.jpg' | python resnet.py --labels flower_photos/label_list.txt
```

Input: images (bitmaps) on stdin such as <a href="flower_photos/daisy/134409839_71069a95d1_m.jpg">this</a>.

Outputs labels and scores for each input image.

Click <a href="https://bj.bcebos.com/paddlehub-dataset/flower_photos.tar.gz">here</a> for more pictures of flowers.
