# based on https://github.com/espnet/espnet_model_zoo

# example usage:
#   python text_to_speech.py -l en -c ljspeech < sample_input.en.txt -o audio.en < sample_input.en.txt
#   python text_to_speech.py -l zh -c csmsc < sample_input.zh.txt -o audio.zh < sample_input.zh.txt

import sys,argparse,scipy.signal
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
d = ModelDownloader()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help="see https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv", default=None)
parser.add_argument("-c", "--corpus", help="vctk|ljspeech|libritts|csmsc...", default=None)
parser.add_argument("-l", "--language", help="en|zh|...", default=None)
parser.add_argument("-o", "--output", help="prefix for output filenames", default='audio')
args = parser.parse_args()

# The following will attempt to find an appropriate model from whatever the user provides (language, corpus, etc)

if not args.model_string is None:
    m = d.download_and_unpack(args.model_string)
else:
    kwds={}
    if not args.corpus is None:
        kwds['corpus'] = args.corpus
    if not args.language is None:
        kwds['lang'] = args.language
    m = d.download_and_unpack(task="tts", version=-1, **kwds),

text2speech = Text2Speech(**m[0])

for i,line in enumerate(sys.stdin):
    if len(line) > 1:
        speech, *_ = text2speech(line.rstrip())
        output_filename = args.output + '.' + str(i) + '.wav'
        print(output_filename)
        soundfile.write(output_filename, speech.numpy(), text2speech.fs, "PCM_16")
