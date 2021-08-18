# from https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.1/demo/sequence_labeling

import paddlehub as hub
import paddle,sys,argparse

label_list = hub.datasets.MSRA_NER.label_list
label_map = {
    idx: label for idx, label in enumerate(label_list)
}

model = hub.Module(name='ernie_tiny', version='2.0.1', task='token-cls', label_map=label_map)

train_dataset = hub.datasets.MSRA_NER(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='train')
dev_dataset = hub.datasets.MSRA_NER(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='dev')

optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='test_ernie_token_cls', use_gpu=False)

trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset)

# 在测试集上评估当前训练模型
trainer.evaluate(test_dataset, batch_size=32)
