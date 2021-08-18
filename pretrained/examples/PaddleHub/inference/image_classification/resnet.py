import paddle,sys,argparse
import paddlehub as hub

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--labels", help="filename containing labels [defaults to Imagenet2012 categories]", default=None)
args = parser.parse_args()


# example usage
# find flower_photos -name '*.jpg' | python resnet.py
# find flower_photos -name '*.jpg' | python resnet.py --labels flower_photos/label_list.txt

# for more flower pictures, download: https://bj.bcebos.com/paddlehub-dataset/flower_photos.tar.gz
def dict2str(d):
    return '\t'.join([ str(key) + ':' + str(d[key]) for key in d.keys() ])

def lines(fd):
    return fd.read().rstrip().split('\n')


if args.labels is None:    
    model = hub.Module(name='resnet50_vd_imagenet_ssld')
else:
    with open(args.labels, 'r') as fd: 
        model = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=lines(fd))

input_file_names = lines(sys.stdin)
results = model.predict(input_file_names)
for f,r in zip(input_file_names,results):
    print(f + '\t' + dict2str(r))
