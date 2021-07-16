# OCR

Super-simple example of pre-trained OCR model from <a
href="https://www.paddlepaddle.org.cn/hubdetail?name=chinese_text_detection_db_server&en_category=TextRecognition">PaddleHub</a>.
See <a href="https://www.paddlepaddle.org.cn/hublist>PaddleHub
home</a> for many more pre-trained models.


Install:
   ```pip install -r requirements.txt```

Example of Usage:
   ```python add_boxes_to_images.py Sample_input_for_OCR.png > Sample_input_for_OCR.txt```

Input: images (bitmaps) on command line such as <a href="Sample_input_for_OCR.png">this</a>.

Outputs for each image on command line:
<ol>
<li>text with boxes such as <a href="Sample_input_for_OCR.png">this</a>.</li>
<li>image with boxes such as <a href="Sample_input_for_OCR_with_boxes.png">this</a>.</li>
</ol>
