import paddle,sys,argparse
import paddlehub as hub
import paddlehub.vision.transforms as T

transforms = T.Compose([T.Resize((256, 256)),
                        T.CenterCrop(224),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])],
                        to_rgb=True)

import paddlehub.vision.transforms as T

transforms = T.Compose([T.Resize((256, 256)),
                        T.CenterCrop(224),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])],
                        to_rgb=True)

from paddlehub.datasets import Flowers

flowers = Flowers(transforms)

flowers_validate = Flowers(transforms, mode='val')

model = hub.Module(name="resnet50_vd_imagenet_ssld", label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"])

# 更换name参数即可无缝切换efficientnet模型, 代码示例如下
# model = hub.Module(name="efficientnetb7_imagenet")

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt', use_gpu=True)

trainer.train(flowers, epochs=100, batch_size=32, eval_dataset=flowers_validate, save_interval=1)
