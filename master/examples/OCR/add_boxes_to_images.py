import sys,cv2
import paddlehub as hub
ocr = hub.Module(name="chinese_ocr_db_crnn_server")

# example usage:
#  python add_boxes_to_images.py Sample_input_for_OCR.png > Sample_input_for_OCR.txt

# input: one or more bitmaps on command line
# output: for each bitmap, output positions, text, copy of bitmap with red boxes



color=(0,0,255)                 # red
thickness=2
images = [cv2.imread(f) for f in sys.argv[1:]]
boxes_per_image = ocr.recognize_text(images)

def box2points(box):
    X = [ p[0] for p in box ]
    Y = [ p[1] for p in box ]
    return (min(X),min(Y)), (max(X),max(Y))

def output_filename(f):
    pieces = f.split('.');
    return '.'.join(pieces[0:-1] + ['with_boxes', pieces[-1]])

for f,image,boxes in zip(sys.argv[1:], images, boxes_per_image):
    for b in boxes['data']:
        start_point,end_point = box2points(b['text_box_position'])
        print('\t'.join(map(str,[f, start_point,end_point,b['text']])))
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.imwrite(output_filename(f), image)
        
