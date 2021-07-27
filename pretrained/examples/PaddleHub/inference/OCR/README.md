# Application OCR, based on PaddleHub

Super-simple example of pre-trained OCR model from <a
href="https://www.paddlepaddle.org.cn/hubdetail?name=chinese_text_detection_db_server&en_category=TextRecognition">PaddleHub</a>.
See <a href="https://www.paddlepaddle.org.cn/hublist">PaddleHub
home</a> for many more pre-trained models.


Install:
   ```shell
   pip install -r requirements.txt
```

Example of Usage:
   ```shell
   echo 'Sample_input_for_OCR.png' | python OCR.py
```

Input: images (bitmaps) on stdin such as <a href="Sample_input_for_OCR.png">this</a>.

Outputs for each input image:
<ol>
<li>text with boxes such as <a href="Sample_input_for_OCR.txt">this</a>.</li>
<li>image with boxes such as <a href="Sample_input_for_OCR.with_boxes.png">this</a>.</li>
</ol>
