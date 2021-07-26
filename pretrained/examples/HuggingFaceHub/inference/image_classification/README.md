# Application: image classification based on HuggingFaceHub

```shell 
  pip install -r requirements.txt

  # inputs images (urls and/or filenames)
  # outputs class labels
  head COCO_val2017.txt | python VIT.py
  ls images/*.jpg | python VIT.py
```

Based on <a href="https://huggingface.co/google/vit-base-patch16-224">Vision Transformer</a>.
