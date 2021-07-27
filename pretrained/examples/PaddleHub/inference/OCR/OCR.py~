import sys,cv2
import paddlehub as hub
ocr = hub.Module(name="chinese_ocr_db_crnn_server")

# example usage:
#  echo 'Sample_input_for_OCR.png' | python add_boxes_to_images.py > Sample_input_for_OCR.txt

# input: one or more files on stdin, where each file is a bitmap (.jpg, .png, etc)
# output: for each input file, output positions, text, copy of bitmap with red boxes

color=(0,0,255)                 # red
thickness=2

def box2points(box):
    X = [ p[0] for p in box ]
    Y = [ p[1] for p in box ]
    return (min(X),min(Y)), (max(X),max(Y))

def output_filename(f):
    pieces = f.split('.');
    return '.'.join(pieces[0:-1] + ['with_boxes', pieces[-1]])

files = []
for line in sys.stdin:
    fields = line.rstrip().split()
    if len(fields) > 0:
        files.append(fields[0])

images = [cv2.imread(f) for f in files]
boxes_per_image = ocr.recognize_text(images)
        
for f,image,boxes in zip(files, images, boxes_per_image):
    for b in boxes['data']:
        start_point,end_point = box2points(b['text_box_position'])
        print(f + '\t' + '\t'.join(map(str,[f, start_point,end_point,b['text']])))
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        outf = output_filename(f)
        try:
            cv2.imwrite(outf, image)
        except:
            print('cannot write: ' + str(outf), file=sys.stderr)
