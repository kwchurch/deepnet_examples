# Application image_classification, based on PaddleHub

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
   find . -name '*.jpg' | python resnet.py
   find . -name '*.jpg' | python resnet.py --labels flower_photos/label_list.txt
```

Input: images (bitmaps) on stdin such as <a href="flower_photos/daisy/134409839_71069a95d1_m.jpg">this</a>.

Outputs labels and scores for each input image.  By default, labels
are the Imagenet2012 categories.  Fine tuning will be necessary for
custom labels.

To fine tune for flowers:

   ```shell
   python fine_tune_flowers.py
```

WARNING: this will take a long time.

After fine tuning you can say

   ```shell
   M=img_classification_ckpt/best_model/model.pdopt
   find . -name '*.jpg' | python resnet.py -c $M -l flower_photos/label_list.txt
```

Click <a href="https://bj.bcebos.com/paddlehub-dataset/flower_photos.tar.gz">here</a> for more pictures of flowers.
