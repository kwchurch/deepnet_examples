# from https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.1/demo/sequence_labeling

import paddlehub as hub

split_char = "\002"
label_list = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]
text_a = [
    '去年十二月二十四日，市委书记张敬涛召集县市主要负责同志研究信访工作时，提出三问：『假如上访群众是我们的父母姐妹，你会用什么样的感情对待他们？',
    '新华社北京5月7日电国务院副总理李岚清今天在中南海会见了美国前商务部长芭芭拉·弗兰克林。',
    '根据测算，海卫1表面温度已经从“旅行者”号探测器1989年造访时的零下236摄氏度上升到零下234摄氏度。',
    '华裔作家韩素音女士曾三次到大足，称“大足石窟是一座未被开发的金矿”。',
]
data = [[split_char.join(text)] for text in text_a]
label_map = {
    idx: label for idx, label in enumerate(label_list)
}

model = hub.Module(
    name='ernie_tiny',
    version='2.0.1',
    task='token-cls',
    load_checkpoint='./token_cls_save_dir/best_model/model.pdparams',
    label_map=label_map,
)

results = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=False)
for idx, text in enumerate(text_a):
    print(f'Data: {text} \t Lable: {", ".join(results[idx][1:len(text)+1])}')
