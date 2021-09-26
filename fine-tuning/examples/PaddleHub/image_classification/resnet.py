import paddle,sys,argparse
import paddlehub as hub

# example usage
# find flower_photos -name '*.jpg' | python resnet.py
# find flower_photos -name '*.jpg' | python resnet.py --labels flower_photos/label_list.txt

# for more flower pictures, download: https://bj.bcebos.com/paddlehub-dataset/flower_photos.tar.gz

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--labels", help="filename containing labels [defaults to Imagenet2012 categories]", default=None)
parser.add_argument("-c", "--checkpoint", help="checkpoint from fine-tuning", default=None)
args = parser.parse_args()

if not args.checkpoint is None:
    assert not args.labels is None, '--labels are required if --checkpoint is provided'

def dict2str(d):
    return '\t'.join([ str(key) + ':' + str(d[key]) for key in d.keys() ])

def lines(fd):
    return fd.read().rstrip().split('\n')

if args.labels is None:    
    model = hub.Module(name="resnet50_vd_imagenet_ssld")
else:
    with open(args.labels, 'r') as fd: 
        if not args.checkpoint is None:
            model = hub.Module(
                name='resnet50_vd_imagenet_ssld',
                label_list=lines(fd),
                load_checkpoint=args.checkpoint)
        else:
            model = hub.Module(name="resnet50_vd_imagenet_ssld", label_list=lines(fd))

input_file_names = lines(sys.stdin)
results = model.predict(input_file_names)
for f,r in zip(input_file_names,results):
    print(f + '\t' + dict2str(r))
