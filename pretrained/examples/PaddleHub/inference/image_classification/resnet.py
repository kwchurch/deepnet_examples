import paddle,sys
import paddlehub as hub

# example usage
# find . -name '*.jpg' | python resnet.py

# for more flower pictures, download: https://bj.bcebos.com/paddlehub-dataset/flower_photos.tar.gz
def dict2str(d):
    return '\t'.join([ str(key) + ':' + str(d[key]) for key in d.keys() ])

model = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"])

input_file_names = [ f for f in  sys.stdin.read().split('\n') if len(f) > 0 ]
results = model.predict(input_file_names)
for f,r in zip(input_file_names,results):
    print(f + '\t' + dict2str(r))
