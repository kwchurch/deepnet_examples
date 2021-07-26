import sys
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

for line in sys.stdin:
    filename = line.rstrip()
    try:
        image = Image.open(requests.get(filename, stream=True).raw)
    except:
        image = Image.open(filename)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print(filename +  '\tpredicted_label: ' + model.config.id2label[predicted_class_idx])
