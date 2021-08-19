
# from https://github.com/espnet/espnet_model_zoo

# example usage:
# find . -name '*.wav' | sed 1q | python speech_to_text.py -l en --corpus librispeech

import sys,argparse,scipy.signal
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
d = ModelDownloader()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help="see https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv", default=None)
parser.add_argument("-c", "--corpus", help="wsj|librispeech|aishell|...", default=None)
parser.add_argument("-l", "--language", help="en|zh|...", default=None)
parser.add_argument("-r", "--sampling_rate", type=int, help="defaults to 16000", default=16000)
parser.add_argument("-N", "--nbest", type=int, help="nbest", default=1)
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
    m = d.download_and_unpack(task="asr", version=-1, **kwds),


speech2text = Speech2Text(
    **m[0],
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=args.nbest
)

for line in sys.stdin:
    filename = line.rstrip()
    speech, sampling_rate = soundfile.read(filename)

    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    if sampling_rate != args.sampling_rate:
        sec = len(speech)/sampling_rate
        speech = scipy.signal.resample(speech, int(sec * args.sampling_rate))

    nbests = speech2text(speech)
    for i,path in enumerate(nbests):
        print('\t'.join([filename, str(i), str(path[0])]))
